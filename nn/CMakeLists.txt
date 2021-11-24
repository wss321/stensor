cmake_minimum_required(VERSION 2.6)
project(nn)

set(CMAKE_CXX_STANDARD 11)
file(GLOB tmp *.cpp)
set(CORE_SRCS ${CORE_SRCS} ${tmp})

file(GLOB TEST_SRCS *_test.cpp)
file(GLOB CU_SRCS *.cu)
cuda_add_library(nn_cuda SHARED ${CU_SRCS})
target_include_directories(nn_cuda
        PRIVATE ${PROJECT_SOURCE_DIR}
        PRIVATE ${stensor_SOURCE_DIR}/public
        #        PRIVATE ${CUBLAS_INCLUDE_DIRS}
        )
target_link_libraries(nn_cuda
        lib::public
        lib::tensor
        ${CUBLAS_LIBRARIES}
        #        ${CUDA_curand_LIBRARY}
        )

add_library(${PROJECT_NAME} STATIC
        ${CORE_SRCS}
        )
add_library(lib::nn ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}
        PUBLIC ${PROJECT_SOURCE_DIR}
        PUBLIC ${stensor_SOURCE_DIR}
        PUBLIC ${stensor_BINARY_DIR}# proto.pb.h
        )

target_link_libraries(${PROJECT_NAME}
        lib::public
        lib::tensor
        nn_cuda
        )
enable_testing()
include(GoogleTest)
include(GoogleTest)
foreach(test_src ${TEST_SRCS})
    STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" test_target ${test_src})  #文件名字符串获取
    #    message(${test_target})
    add_executable(${test_target}  ${test_src})
    target_include_directories(${test_target}
            PRIVATE ${PROJECT_SOURCE_DIR}
            PRIVATE ${stensor_SOURCE_DIR}
            )
    target_link_libraries(${test_target}
            lib::public
            lib::tensor
            lib::nn
            nn_cuda
            )
    gtest_discover_tests(${test_target})
endforeach()
