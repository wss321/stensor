project(public)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Boost COMPONENTS thread REQUIRED)
#find_package(Boost 1.54 REQUIRED COMPONENTS system thread filesystem)

find_package(CUDA REQUIRED)
#get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)
#set(CUDNN_LIB_NAME "libcudnn.so")
#find_library(CUDNN_LIBRARY NAMES ${CUDNN_LIB_NAME}
#        PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE} ${__libpath_hist} ${__libpath_hist}/../lib
#        DOC "Path to cuDNN library.")

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
#        PUBLIC ${CUDNN_INCLUDE_DIR}
        )
target_link_libraries(${PROJECT_NAME}
        glog::glog
        ${CUDA_LIBRARIES}
        Boost::thread
        ${CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
#        ${CUDNN_LIBRARY}
        )

#GTEST
enable_testing()
include(GoogleTest)
foreach (test_src ${TEST_SRCS})
    STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" test_target ${test_src})  #文件名字符串获取
    message(${test_target})
    add_executable(${test_target} ${test_src})
    target_include_directories(${test_target}
            PRIVATE ${PROJECT_SOURCE_DIR}
            PRIVATE ${stensor_SOURCE_DIR}
            )
    target_link_libraries(${test_target}
            lib::public
            ${GTEST_LIBRARIES}
            ${GTEST_MAIN_LIBRARIES}
            )
    gtest_discover_tests(${test_target})
endforeach ()