project(public)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
set(CMAKE_MODULE_PATH ../cmake)
include(../cmake/utils.cmake)
file(GLOB tmp *.cpp)
set(CORE_SRCS ${CORE_SRCS} ${tmp})
file(GLOB TEST_SRCS *_test.cpp)

exclude(CORE_SRCS "${CORE_SRCS}" ${TEST_SRCS})

add_library(${PROJECT_NAME} ${CORE_SRCS})
add_library(lib::public ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}
        PUBLIC ${PROJECT_SOURCE_DIR}
        )
target_link_libraries(${PROJECT_NAME}
        glog::glog
        ${GTEST_LIBRARIES}
        ${GTEST_MAIN_LIBRARIES}
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