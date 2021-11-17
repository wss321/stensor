project(public)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS thread REQUIRED)
#find_package(Boost 1.54 REQUIRED COMPONENTS system thread filesystem)

set(CMAKE_MODULE_PATH ../cmake)
include(../cmake/FindcuBLAS.cmake)
find_package(cuBLAS REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -gencode arch=compute_30,code=sm_30;
        -gencode arch=compute_35,code=sm_35;
        -gencode arch=compute_50,code=sm_50;
        -gencode arch=compute_52,code=sm_52;
        -gencode arch=compute_60,code=sm_60;
        -gencode arch=compute_61,code=sm_61;
        -gencode arch=compute_75,code=sm_75;
        -std=c++11;)

set(CMAKE_MODULE_PATH ../cmake)
include(../cmake/utils.cmake)
file(GLOB tmp *.cpp)
set(CORE_SRCS ${CORE_SRCS} ${tmp})
file(GLOB TEST_SRCS *_test.cpp)

exclude(CORE_SRCS "${CORE_SRCS}" ${TEST_SRCS})

add_library(${PROJECT_NAME} SHARED ${CORE_SRCS})
add_library(lib::public ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}
        PUBLIC ${PROJECT_SOURCE_DIR}
        PUBLIC ${CUDA_INCLUDE_DIRS}
        PUBLIC ${CUBLAS_INCLUDE_DIRS}
        PUBLIC ${Boost_INCLUDE_DIRS}
        )
target_link_libraries(${PROJECT_NAME}
        glog::glog
        ${GTEST_LIBRARIES}
        ${GTEST_MAIN_LIBRARIES}
        ${CUDA_LIBRARIES}
        Boost::thread
        ${CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
)

#GTEST
enable_testing()
include(GoogleTest)
foreach(test_src ${TEST_SRCS})
    STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" test_target ${test_src})  #文件名字符串获取
    #    message(${test_target})
    add_executable(${test_target}  ${test_src})
    target_include_directories(${test_target}
            PUBLIC ${PROJECT_SOURCE_DIR}
            PUBLIC ${stensor_SOURCE_DIR}
            )
    target_link_libraries(${test_target}
            lib::public
            )
    gtest_discover_tests(${test_target})
endforeach()