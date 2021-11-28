cmake_minimum_required(VERSION 3.0)
PROJECT(io)
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)
file(GLOB tmp *.cpp)
set(CPP_SRCS ${CPP_SRCS} ${tmp})

file(GLOB TEST_SRCS *_test.cpp)
file(GLOB CU_SRCS *.cu)
cuda_add_library(io_cuda SHARED ${CU_SRCS})
target_include_directories(io_cuda
        PUBLIC ${PROJECT_SOURCE_DIR}
        PRIVATE ${stensor_SOURCE_DIR}
        )
target_link_libraries(io_cuda
        lib::public
        lib::tensor
        )
add_library(${PROJECT_NAME} STATIC
        ${CPP_SRCS}
        )
add_library(lib::io ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}
        PRIVATE ${PROJECT_SOURCE_DIR}
        PRIVATE ${stensor_SOURCE_DIR}
        )
target_link_libraries(${PROJECT_NAME}
        lib::public
        lib::tensor
        ${OpenCV_LIBS}
        io_cuda
        )
enable_testing()
include(GoogleTest)
foreach(test_src ${TEST_SRCS})
    STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" test_target ${test_src})  #文件名字符串获取
    add_executable(${test_target}  ${test_src})
    target_include_directories(${test_target}
            PRIVATE ${PROJECT_SOURCE_DIR}
            PRIVATE ${stensor_SOURCE_DIR}
            )
    target_link_libraries(${test_target}
            lib::public
            lib::tensor
            lib::io
            io_cuda
            ${GTEST_LIBRARIES}
            ${GTEST_MAIN_LIBRARIES}
            )
    gtest_discover_tests(${test_target})
endforeach()
