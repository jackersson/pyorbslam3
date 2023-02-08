cmake_minimum_required(VERSION 3.4...3.18)

project(pyOrbSlam)

LIST(APPEND CMAKE_MODULE_PATH
    ${PROJECT_SOURCE_DIR}/cmake
)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -march=native  -O0 -g3 -ggdb")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -march=native  -O0 -g3 -ggdb")

add_subdirectory(modules/pybind11)

find_package(OpenCV REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/modules/ORB_SLAM3/cmake_modules)

SET(ORB_SLAM3_SOURCE_DIR $ENV{ORB_SLAM3_SOURCE_DIR})


include_directories(
    ${PROJECT_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}/
    ${ORB_SLAM3_SOURCE_DIR}/include
    ${ORB_SLAM3_SOURCE_DIR}/include/CameraModels
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus/sophus
    ${PROJECT_SOURCE_DIR}/modules/pybind11_opencv_numpy
    ${Pangolin_INCLUDE_DIRS}
)

set(LIBS
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    -lboost_system
)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message("Python_FOUND:${Python_FOUND}")
message("Python_VERSION:${Python_VERSION}")
message("Python_Development_FOUND:${Python_Development_FOUND}")
message("Python_LIBRARIES:${Python_LIBRARIES}")

pybind11_add_module(pyorbslam3
    src/pyOrbSlam.cpp
    ${PROJECT_SOURCE_DIR}/modules/pybind11_opencv_numpy/ndarray_converter.cpp
)

add_executable(testOrbSlamVideo
        ${PROJECT_SOURCE_DIR}/src/orbSlamTest.cc)
target_link_libraries(testOrbSlamVideo
PRIVATE
    ${OpenCV_LIBS}
    ${LIBS}
)


target_link_libraries(pyorbslam3
PRIVATE
    ${OpenCV_LIBS}
    ${LIBS}
    ${PYTHON_LIBRARIES} Python3::NumPy
)

target_compile_definitions(pyorbslam3
PRIVATE
    VERSION_INFO=${EXAMPLE_VERSION_INFO}
)
#string(REPLACE "libpython${PY_LIB}" /libORB_SLAM3.so PY_LIB ${Python_LIBRARIES})
message("copying libORB_SLAM3.so to python libraries")

string(REGEX REPLACE "[^\/]+$" "libORB_SLAM3.so" PY_LIB ${Python_LIBRARIES})

set(COM
   "cp ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so ${PY_LIB}")

message(${COM})

execute_process (
    COMMAND bash -c ${COM})


