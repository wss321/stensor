cmake_minimum_required(VERSION 2.6)
project (tensor)
set(CMAKE_CXX_STANDARD 11)
# Dependency
set(CMAKE_MODULE_PATH ../cmake)
include(../cmake/utils.cmake)
include(../cmake/FindcuBLAS.cmake)
find_package(cuBLAS REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(GTest REQUIRED)

INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})
file(GLOB tmp *.cpp)
set(CORE_SRCS ${CORE_SRCS} ${tmp})
file(GLOB TEST_SRCS *_test.cpp)
file(GLOB CU_SRCS *.cu)
message(${CUBLAS_LIBRARIES})
exclude(CORE_SRCS "${CORE_SRCS}" ${TEST_SRCS})
cuda_add_library(core_cuda SHARED ${CU_SRCS})

add_library(${PROJECT_NAME} SHARED
        ${CORE_SRCS}
        )
add_library(lib::tensor ALIAS ${PROJECT_NAME})
target_include_directories(core_cuda
        PRIVATE ${PROJECT_SOURCE_DIR}
        PRIVATE ${stensor_SOURCE_DIR}/public
#        PRIVATE ${CUBLAS_INCLUDE_DIRS}
        )
target_link_libraries(core_cuda
        lib::public
        ${CUBLAS_LIBRARIES}
#        ${CUDA_curand_LIBRARY}
        )
target_include_directories(${PROJECT_NAME}
        PUBLIC ${PROJECT_SOURCE_DIR}
        PUBLIC ${stensor_SOURCE_DIR}
        PUBLIC ${stensor_BINARY_DIR}# proto.pb.h
        )

target_link_libraries(${PROJECT_NAME}
        lib::proto
        lib::public
        ${OpenBLAS_LIB}
        core_cuda
        )

# Test Target Generate
enable_testing()
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
            core_cuda
            )
    gtest_discover_tests(${test_target})
endforeach()
