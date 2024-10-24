
#include <Converter.h>
#include <ImuTypes.h>
#include <MapPoint.h>
#include <System.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>

#include "ndarray_converter.h"

namespace py = pybind11;
using namespace ORB_SLAM3;

class Debug
{
public:
    Debug(){};
    ~Debug(){};
    std::string getPID()
    {
        pid_t pid = getpid();
        return to_string(pid);
    }
};

// Added to handle IMU data
struct IMUData
{
    // accelerometer
    float acc_x;
    float acc_y;
    float acc_z;

    // gyroscope
    float ang_vel_x;
    float ang_vel_y;
    float ang_vel_z;

    double timestamp;
};

class PyOrbSlam
{
public:
    // Create SLAM system. It initializes all system threads and gets ready to
    // process frames.
    std::unique_ptr<ORB_SLAM3::System> slam;
    std::unique_ptr<ORB_SLAM3::Converter> conv;
    // PyOrbSlam(){};
    PyOrbSlam(std::string path_to_vocabulary, std::string path_to_settings,
              std::string sensorType, bool useViewer)
    {
        ORB_SLAM3::System::eSensor sensor;
        if (sensorType.compare("Mono") == 0)
        {
            sensor = ORB_SLAM3::System::MONOCULAR;
        }
        if (sensorType.compare("Stereo") == 0)
        {
            sensor = ORB_SLAM3::System::STEREO;
        }
        if (sensorType.compare("RGBD") == 0)
        {
            sensor = ORB_SLAM3::System::RGBD;
        }
        if (sensorType.compare("MonoIMU") == 0)
        {
            sensor = ORB_SLAM3::System::IMU_MONOCULAR;
        }
        if (sensorType.compare("StereoIMU") == 0)
        {
            sensor = ORB_SLAM3::System::IMU_STEREO;
        }
        if (sensorType.compare("RGBDIMU") == 0)
        {
            sensor = ORB_SLAM3::System::IMU_RGBD;
        }
        try
        {
            slam = std::make_unique<ORB_SLAM3::System>(
                path_to_vocabulary, path_to_settings, sensor, useViewer);
            conv = std::make_unique<ORB_SLAM3::Converter>();
        }
        catch (const std::exception& e)
        {
            ofstream myfile;
            myfile.open("errorLog.txt");
            myfile << e.what();
            myfile.close();
            // std::cerr << e.what() << std::endl;
            throw runtime_error(e.what());
        }
    };

    ~PyOrbSlam()
    {
        slam->Shutdown();
        this_thread::sleep_for(chrono::milliseconds(5000));
        // delete &slam;
        // delete &conv;
    };

    void Shutdown()
    {
        try
        {
            slam->Shutdown();
        }
        catch (const std::exception& e)
        {
            cout << e.what() << endl;
            ofstream myfile;
            myfile.open("errorLog.txt");
            myfile << e.what();
            myfile.close();
            // std::cerr << e.what() << std::endl;
            throw runtime_error(e.what());
        }
    };

    void saveTrajectory(string filePath)
    {
        // Save camera trajectory
        slam->SaveKeyFrameTrajectoryTUM(filePath);
    };

    void ActivateLocalizationMode() { slam->ActivateLocalizationMode(); };

    void DeactivateLocalizationMode() { slam->DeactivateLocalizationMode(); };

    void Reset() { slam->Reset(); };

    void ResetActiveMap() { slam->ResetActiveMap(); };

    int GetTrackingState() { return slam->GetTrackingState(); };

    bool IsLost() { return slam->isLost(); };

    int getNrOfMaps()
    {
        vector<ORB_SLAM3::Map*> vpMaps = slam->mpAtlas->GetAllMaps();
        return vpMaps.size();
    }

    int getCurrentMapId()
    {
        ORB_SLAM3::Map* pActiveMap = slam->mpAtlas->GetCurrentMap();
        return pActiveMap->GetId();
    }

    cv::Mat GetTrackedMapReferencePointsOfMap(int mapNr)
    {
        try
        {
            ORB_SLAM3::Map* pActiveMap;
            if (mapNr == -1)
            {
                pActiveMap = slam->mpAtlas->GetCurrentMap();
            }
            else
            {
                vector<ORB_SLAM3::Map*> vpMaps = slam->mpAtlas->GetAllMaps();
                pActiveMap = vpMaps[mapNr];
            }
            if (!pActiveMap)
                return cv::Mat(1, 3, CV_32FC1, 0.0f);
            const vector<ORB_SLAM3::MapPoint*>& vpRefMPs =
                pActiveMap->GetReferenceMapPoints();
            set<ORB_SLAM3::MapPoint*> spRefMPs(vpRefMPs.begin(),
                                               vpRefMPs.end());
            if (vpRefMPs.empty())
                return cv::Mat(1, 3, CV_32FC1, 0.0f);
            cv::Mat positions = cv::Mat(vpRefMPs.size(), 40, CV_32FC1, 0.0f);
            for (size_t i = 0, iend = vpRefMPs.size(); i < iend; i++)
            {
                if (vpRefMPs[i]->isBad())
                    continue;
                Eigen::Matrix<float, 3, 1> pos = vpRefMPs[i]->GetWorldPos();
                // glVertex3f(pos(0),pos(1),pos(2));
                positions.at<float>(i, 0) = pos(0);
                positions.at<float>(i, 1) = pos(1);
                positions.at<float>(i, 2) = pos(2);
                Eigen::Matrix<float, 3, 1> norm = vpRefMPs[i]->GetNormal();
                // glVertex3f(pos(0),pos(1),pos(2));
                positions.at<float>(i, 3) = norm(0);
                positions.at<float>(i, 4) = norm(1);
                positions.at<float>(i, 5) = norm(2);
                positions.at<float>(i, 6) = float(vpRefMPs[i]->Observations());
                positions.at<float>(i, 7) = vpRefMPs[i]->GetFoundRatio();
                cv::Mat descr = vpRefMPs[i]->GetDescriptor();
                for (int z = 0; z < 32; z++)
                {
                    positions.at<float>(i, 8 + z) =
                        float(descr.at<unsigned char>(z));
                }
            }
            return positions;
        }
        catch (const std::exception& e)
        {
            cout << e.what() << endl;
            ofstream myfile;
            myfile.open("errorLog.txt");
            myfile << e.what();
            myfile.close();
            // std::cerr << e.what() << std::endl;
            throw runtime_error(e.what());
        }
    }

    cv::Mat GetTrackedMapPointsOfMap(int mapNr)
    {
        try
        {
            ORB_SLAM3::Map* pActiveMap;
            if (mapNr == -1)
            {
                pActiveMap = slam->mpAtlas->GetCurrentMap();
            }
            else
            {
                vector<ORB_SLAM3::Map*> vpMaps = slam->mpAtlas->GetAllMaps();
                pActiveMap = vpMaps[mapNr];
            }
            if (!pActiveMap)
                return cv::Mat(1, 3, CV_32FC1, 0.0f);

            const vector<ORB_SLAM3::MapPoint*>& vpMPs =
                pActiveMap->GetAllMapPoints();
            const vector<ORB_SLAM3::MapPoint*>& vpRefMPs =
                pActiveMap->GetReferenceMapPoints();

            set<ORB_SLAM3::MapPoint*> spRefMPs(vpRefMPs.begin(),
                                               vpRefMPs.end());

            if (vpMPs.empty())
                return cv::Mat(1, 3, CV_32FC1, 0.0f);
            cv::Mat positions = cv::Mat(vpMPs.size(), 40, CV_32FC1, 0.0f);
            for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
            {
                if (vpMPs[i]->isBad())  //|| spRefMPs.count(vpMPs[i]))
                    continue;
                Eigen::Matrix<float, 3, 1> pos = vpMPs[i]->GetWorldPos();
                // glVertex3f(pos(0),pos(1),pos(2));
                positions.at<float>(i, 0) = pos(0);
                positions.at<float>(i, 1) = pos(1);
                positions.at<float>(i, 2) = pos(2);
                Eigen::Matrix<float, 3, 1> norm = vpMPs[i]->GetNormal();
                // glVertex3f(pos(0),pos(1),pos(2));
                positions.at<float>(i, 3) = norm(0);
                positions.at<float>(i, 4) = norm(1);
                positions.at<float>(i, 5) = norm(2);
                positions.at<float>(i, 6) = float(vpMPs[i]->Observations());
                positions.at<float>(i, 7) = vpMPs[i]->GetFoundRatio();
                cv::Mat descr = vpMPs[i]->GetDescriptor();
                for (int z = 0; z < 32; z++)
                {
                    positions.at<float>(i, 8 + z) =
                        float(descr.at<unsigned char>(z));
                }
            }
            return positions;
        }
        catch (const std::exception& e)
        {
            cout << e.what() << endl;
            ofstream myfile;
            myfile.open("errorLog.txt");
            myfile << e.what();
            myfile.close();
            // std::cerr << e.what() << std::endl;
            throw runtime_error(e.what());
        }
    }

    cv::Mat getKeyFramesOfMap(int mapNr, bool withIMU)
    {
        try
        {
            ORB_SLAM3::Map* pActiveMap;
            if (mapNr == -1)
            {
                pActiveMap = slam->mpAtlas->GetCurrentMap();
            }
            else
            {
                vector<ORB_SLAM3::Map*> vpMaps = slam->mpAtlas->GetAllMaps();
                pActiveMap = vpMaps[mapNr];
            }
            if (!pActiveMap)
                return cv::Mat(1, 3, CV_32FC1, 0.0f);
            vector<ORB_SLAM3::KeyFrame*> vpKFs = pActiveMap->GetAllKeyFrames();
            sort(vpKFs.begin(), vpKFs.end(), ORB_SLAM3::KeyFrame::lId);
            cv::Mat keyPositions = cv::Mat(vpKFs.size(), 7, CV_32FC1, 0.0f);
            for (size_t i = 0; i < vpKFs.size(); i++)
            {
                ORB_SLAM3::KeyFrame* pKF = vpKFs[i];

                // pKF->SetPose(pKF->GetPose()*Two);

                if (!pKF || pKF->isBad())
                    continue;
                if (withIMU)
                {
                    Sophus::SE3f Twb = pKF->GetImuPose();
                    Eigen::Quaternionf q = Twb.unit_quaternion();
                    Eigen::Vector3f twb = Twb.translation();
                    keyPositions.at<float>(i, 0) = twb(0);
                    keyPositions.at<float>(i, 1) = twb(1);
                    keyPositions.at<float>(i, 2) = twb(2);
                    keyPositions.at<float>(i, 3) = q.x();
                    keyPositions.at<float>(i, 4) = q.y();
                    keyPositions.at<float>(i, 5) = q.z();
                    keyPositions.at<float>(i, 6) = q.w();
                }
                else
                {
                    Sophus::SE3f Twc = pKF->GetPoseInverse();
                    Eigen::Quaternionf q = Twc.unit_quaternion();
                    Eigen::Vector3f t = Twc.translation();
                    keyPositions.at<float>(i, 0) = t(0);
                    keyPositions.at<float>(i, 1) = t(1);
                    keyPositions.at<float>(i, 2) = t(2);
                    keyPositions.at<float>(i, 3) = q.x();
                    keyPositions.at<float>(i, 4) = q.y();
                    keyPositions.at<float>(i, 5) = q.z();
                    keyPositions.at<float>(i, 6) = q.w();
                }
            }
            return keyPositions;
        }
        catch (const std::exception& e)
        {
            cout << e.what() << endl;
            ofstream myfile;
            myfile.open("errorLog.txt");
            myfile << e.what();
            myfile.close();
            // std::cerr << e.what() << std::endl;
            throw runtime_error(e.what());
        }
    }

    cv::Mat processMonocular(cv::Mat& in_image, const double& seconds)
    {
        cv::Mat camPose;
        Sophus::SE3f camPoseReturn;
        g2o::SE3Quat g2oQuat;
        try
        {
            camPoseReturn = slam->TrackMonocular(in_image, seconds);
            g2oQuat = conv->toSE3Quat(camPoseReturn);
            camPose = conv->toCvMat(g2oQuat);
        }
        catch (const std::exception& e)
        {
            ofstream myfile;
            myfile.open("errorLog.txt");
            myfile << e.what();
            myfile.close();
            throw runtime_error(e.what());
            // std::cerr << e.what() << std::endl;
        }
        return camPose;
    }

    cv::Mat processMonocularIMU(cv::Mat& in_image, const double& seconds,
                                const vector<IMUData>& imu_data)
    {
        cv::Mat camPose;
        Sophus::SE3f camPoseReturn;
        g2o::SE3Quat g2oQuat;
        try
        {
            vector<ORB_SLAM3::IMU::Point> v_imu_meas;
            for (const auto& item : imu_data)
            {
                ORB_SLAM3::IMU::Point p(item.acc_x, item.acc_y, item.acc_z,
                                        item.ang_vel_x, item.ang_vel_y,
                                        item.ang_vel_z, item.timestamp);
                v_imu_meas.push_back(p);
            }
            camPoseReturn = slam->TrackMonocular(in_image, seconds, v_imu_meas);
            g2oQuat = conv->toSE3Quat(camPoseReturn);
            camPose = conv->toCvMat(g2oQuat);
        }
        catch (const std::exception& e)
        {
            ofstream myfile;
            myfile.open("errorLog.txt");
            myfile << e.what();
            myfile.close();
            throw runtime_error(e.what());
            // std::cerr << e.what() << std::endl;
        }
        return camPose;
    }

    cv::Mat processRGBD(cv::Mat& in_image, const cv::Mat& depth_map,
                        const double& seconds)
    {
        cv::Mat camPose;
        Sophus::SE3f camPoseReturn;
        g2o::SE3Quat g2oQuat;

        try
        {
            camPoseReturn = slam->TrackRGBD(in_image, depth_map, seconds);
            g2oQuat = conv->toSE3Quat(camPoseReturn);
            camPose = conv->toCvMat(g2oQuat);
        }
        catch (const std::exception& e)
        {
            ofstream myfile;
            myfile.open("errorLog.txt");
            myfile << e.what();
            myfile.close();
            throw runtime_error(e.what());
            // std::cerr << e.what() << std::endl;
        }
        return camPose;
    }

    cv::Mat GetTrackedMapPointsOrig()
    {
        try
        {
            const vector<ORB_SLAM3::MapPoint*>& vpMPs =
                slam->GetTrackedMapPoints();

            if (vpMPs.empty())
                return cv::Mat(1, 3, CV_32FC1, 0.0f);

            cv::Mat positions = cv::Mat(vpMPs.size(), 40, CV_32FC1, 0.0f);
            for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
            {
                if (vpMPs[i]->isBad())  //|| spRefMPs.count(vpMPs[i]))
                    continue;

                Eigen::Matrix<float, 3, 1> pos = vpMPs[i]->GetWorldPos();
                // glVertex3f(pos(0),pos(1),pos(2));
                positions.at<float>(i, 0) = pos(0);
                positions.at<float>(i, 1) = pos(1);
                positions.at<float>(i, 2) = pos(2);
                Eigen::Matrix<float, 3, 1> norm = vpMPs[i]->GetNormal();
                // glVertex3f(pos(0),pos(1),pos(2));
                positions.at<float>(i, 3) = norm(0);
                positions.at<float>(i, 4) = norm(1);
                positions.at<float>(i, 5) = norm(2);
                positions.at<float>(i, 6) = float(vpMPs[i]->Observations());
                positions.at<float>(i, 7) = vpMPs[i]->GetFoundRatio();
                cv::Mat descr = vpMPs[i]->GetDescriptor();
                for (int z = 0; z < 32; z++)
                {
                    positions.at<float>(i, 8 + z) =
                        float(descr.at<unsigned char>(z));
                }
            }
            return positions;
        }
        catch (const std::exception& e)
        {
            cout << e.what() << endl;
            ofstream myfile;
            myfile.open("errorLog.txt");
            myfile << e.what();
            myfile.close();
            // std::cerr << e.what() << std::endl;
            throw runtime_error(e.what());
        }
    }

	cv::Mat getLastKeyFrameMapPoints()
    {
        try
        {
            ORB_SLAM3::Map* pActiveMap = slam->mpAtlas->GetCurrentMap(); 

            int numDims = 4; // x, y, z, point_idx
            if (!pActiveMap)
                return cv::Mat(1, numDims, CV_32FC1, 0.0f);
            vector<ORB_SLAM3::KeyFrame*> vpKFs = pActiveMap->GetAllKeyFrames();
			// std::cout << "vpKFs.size() " << vpKFs.size() << std::endl;
			if (vpKFs.empty())
                return cv::Mat(1, numDims, CV_32FC1, 0.0f);
			
            sort(vpKFs.begin(), vpKFs.end(), ORB_SLAM3::KeyFrame::lId);

			ORB_SLAM3::KeyFrame* pKF = vpKFs.back();
			vector<ORB_SLAM3::MapPoint*> vpMPs = pKF->GetMapPointMatches();
			// std::cout << "vpMPs.size() " << vpMPs.size() << std::endl;

			if (vpMPs.empty())
                return cv::Mat(1, numDims, CV_32FC1, 0.0f);

            cv::Mat positions = cv::Mat(vpMPs.size(), numDims, CV_32FC1, 0.0f);
            for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
            {
				if (vpMPs[i] == NULL)
					continue;
                if (vpMPs[i]->isBad())  //|| spRefMPs.count(vpMPs[i]))
                    continue;

                Eigen::Matrix<float, 3, 1> pos = vpMPs[i]->GetWorldPos();
                // glVertex3f(pos(0),pos(1),pos(2));
                positions.at<float>(i, 0) = pos(0);
                positions.at<float>(i, 1) = pos(1);
                positions.at<float>(i, 2) = pos(2);
                positions.at<float>(i, 3) = static_cast<float>(vpMPs[i]->mnId);
            }
            return positions;
            
        }
        catch (const std::exception& e)
        {
            cout << e.what() << endl;
            ofstream myfile;
            myfile.open("errorLog.txt");
            myfile << e.what();
            myfile.close();
            // std::cerr << e.what() << std::endl;
            throw runtime_error(e.what());
        }
    }

    void SaveAtlas(string filename)
    {
        slam->SaveAtlas(filename);
    }
};

PYBIND11_MODULE(pyorbslam3, m)
{
    NDArrayConverter::init_numpy();

    py::class_<IMUData>(m, "IMUData")
        .def(py::init<float, float, float, float, float, float, double>())
        .def_readwrite("acc_x", &IMUData::acc_x)
        .def_readwrite("acc_y", &IMUData::acc_y)
        .def_readwrite("acc_z", &IMUData::acc_z)
        .def_readwrite("ang_vel_x", &IMUData::ang_vel_x)
        .def_readwrite("ang_vel_y", &IMUData::ang_vel_y)
        .def_readwrite("ang_vel_z", &IMUData::ang_vel_z)
        .def_readwrite("timestamp", &IMUData::timestamp);

    // py::class_<KeyFrame>(m, "KeyFrame")
    //     .def(py::init<Frame&, Map*, KeyFrameDatabase*>(), py::arg("F"),
    //          py::arg("pMap"), py::arg("pKFDB"))
    //     .def("GetPose", &KeyFrame::GetPose)
    //     .def("GetPoseInverse", &KeyFrame::GetPoseInverse)
    //     .def("GetCameraCenter", &KeyFrame::GetCameraCenter)
    //     .def("GetRotation", &KeyFrame::GetRotation)
    //     .def("GetTranslation", &KeyFrame::GetTranslation)
    //     .def("GetConnectedKeyFrames", &KeyFrame::GetConnectedKeyFrames)
    //     .def("GetVectorCovisibleKeyFrames",
    //          &KeyFrame::GetVectorCovisibleKeyFrames)
    //     .def("GetBestCovisibilityKeyFrames",
    //          &KeyFrame::GetBestCovisibilityKeyFrames, py::arg("N"))
    //     .def("GetCovisiblesByWeight", &KeyFrame::GetCovisiblesByWeight,
    //          py::arg("w"))
    //     .def("GetWeight", &KeyFrame::GetWeight, py::arg("w"))
    //     .def("GetChilds", &KeyFrame::GetChilds)
    //     .def("GetParent", &KeyFrame::GetParent)
    //     .def("hasChild", &KeyFrame::hasChild)
    //     .def("GetLoopEdges", &KeyFrame::GetLoopEdges)
    //     .def("GetMapPoints", &KeyFrame::GetMapPoints)
    //     .def("GetMapPointMatches", &KeyFrame::GetMapPointMatches)
    //     .def("TrackedMapPoints", &KeyFrame::TrackedMapPoints, py::arg("minObs"))
    //     .def("GetMapPoint", &KeyFrame::GetMapPoint, py::arg("idx"))
    //     .def("IsInImage", &KeyFrame::IsInImage, py::arg("x"), py::arg("y"))
    //     .def("ComputeSceneMedianDepth", &KeyFrame::ComputeSceneMedianDepth,
    //          py::arg("q") = 2);

    // py::class_<MapPoint>(m, "MapPoint")
    //     .def(py::init<const Eigen::Vector3f&, KeyFrame*, Map*>(),
    //          py::arg("Pos"), py::arg("pRefKF"), py::arg("pMap"))
    //     .def(py::init<const Eigen::Vector3f&, Map*, Frame*, const int&>(),
    //          py::arg("Pos"), py::arg("pMap"), py::arg("pFrame"),
    //          py::arg("idxF"))
    //     .def("GetWorldPos", &MapPoint::GetWorldPos)
    //     .def("GetNormal", &MapPoint::GetNormal)
    //     .def("GetReferenceKeyFrame", &MapPoint::GetReferenceKeyFrame)
    //     .def("GetObservations", &MapPoint::GetObservations)
    //     .def("GetIndexInKeyFrame", &MapPoint::GetIndexInKeyFrame,
    //          py::arg("pKF"))
    //     .def("IsInKeyFrame", &MapPoint::IsInKeyFrame, py::arg("pKF"))
    //     .def("isBad", &MapPoint::isBad)
    //     .def("GetReplaced", &MapPoint::GetReplaced)
    //     .def("GetFoundRatio", &MapPoint::GetFoundRatio)
    //     .def("GetFound", &MapPoint::GetFound)
    //     .def("GetDescriptor", &MapPoint::GetDescriptor)
    //     .def("GetMinDistanceInvariance", &MapPoint::GetMinDistanceInvariance)
    //     .def("GetMaxDistanceInvariance", &MapPoint::GetMaxDistanceInvariance);

    py::class_<Debug>(m, "Debug").def(py::init()).def("getPID", &Debug::getPID);

    py::class_<PyOrbSlam>(m, "System")
        //.def(py::init())
        .def(py::init<string, string, string, bool>(),
             py::arg("path_to_vocabulary"), py::arg("path_to_settings"),
             py::arg("sensorType") = "Mono", py::arg("useViewer") = false)
        .def("saveTrajectory", &PyOrbSlam::saveTrajectory, py::arg("filePath"))
        .def("processMonocular", &PyOrbSlam::processMonocular, py::arg("in_image"),
             py::arg("seconds"))
        .def("processMonocularIMU", &PyOrbSlam::processMonocularIMU, py::arg("in_image"),
             py::arg("seconds"), py::arg("imu_data"))
        .def("processRGBD", &PyOrbSlam::processRGBD, py::arg("in_image"),
             py::arg("depth_map"), py::arg("seconds"))
        .def("ActivateLocalizationMode", &PyOrbSlam::ActivateLocalizationMode)
        .def("DeactivateLocalizationMode",
             &PyOrbSlam::DeactivateLocalizationMode)
        .def("Reset", &PyOrbSlam::Reset)
        .def("ResetActiveMap", &PyOrbSlam::ResetActiveMap)
        .def("GetTrackingState", &PyOrbSlam::GetTrackingState)
        .def("IsLost", &PyOrbSlam::IsLost)
        .def("GetTrackedMapPoints", &PyOrbSlam::GetTrackedMapPointsOfMap,
             py::arg("mapNr") = -1)
        .def("GetTrackedMapReferencePoints",
             &PyOrbSlam::GetTrackedMapReferencePointsOfMap,
             py::arg("mapNr") = -1)
        .def("GetTrackedMapPointsOrig", &PyOrbSlam::GetTrackedMapPointsOrig)
        .def("getLastKeyFrameMapPoints", &PyOrbSlam::getLastKeyFrameMapPoints)
        .def("getNrOfMaps", &PyOrbSlam::getNrOfMaps)
        .def("getKeyFramesOfMap", &PyOrbSlam::getKeyFramesOfMap,
             py::arg("mapNr") = -1, py::arg("withIMU") = false)
        .def("getCurrentMapId", &PyOrbSlam::getCurrentMapId)
        .def("SaveAtlas", &PyOrbSlam::SaveAtlas, py::arg("filename"))
        .def("shutdown", &PyOrbSlam::Shutdown);
};
