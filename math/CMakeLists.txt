cmake_minimum_required(VERSION 3.0)
project (math)
set(CMAKE_CXX_STANDARD 11)
# Dependency
set(CMAKE_MODULE_PATH ../cmake)
include(../cmake/utils.cmake)
include(../cmake/FindcuBLAS.cmake)

find_package(CUDA REQUIRED)
find_package(cuBLAS REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(GTest REQUIRED)

INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})
file(GLOB tmp *.cpp)
set(CORE_SRCS ${CORE_SRCS} ${tmp})
file(GLOB TEST_SRCS *_test.cpp)
file(GLOB CU_SRCS *.cu)
file(GLOB HEADER *.hpp)
message(${CUBLAS_LIBRARIES})
exclude(CORE_SRCS "${CORE_SRCS}" ${TEST_SRCS})

cuda_add_library(math_cuda SHARED ${CU_SRCS} math_base_cuda.hpp)
target_include_directories(math_cuda
        PRIVATE ${PROJECT_SOURCE_DIR}
        PUBLIC ${stensor_SOURCE_DIR}
        PRIVATE ${CUBLAS_INCLUDE_DIRS}
        )
target_link_libraries(math_cuda
        ${CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        )

add_library(${PROJECT_NAME} SHARED
        ${CORE_SRCS}
        ${HEADER}
        )
add_library(lib::math ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}
        PUBLIC ${PROJECT_SOURCE_DIR}
        PUBLIC ${stensor_SOURCE_DIR}
        )
target_link_libraries(${PROJECT_NAME}
        ${OpenBLAS_LIB}
        math_cuda
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
            lib::math
            lib::tensor
            ${GTEST_LIBRARIES}
            ${GTEST_MAIN_LIBRARIES}
            )
    gtest_discover_tests(${test_target})
endforeach()
