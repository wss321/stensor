project(public)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

add_library(${PROJECT_NAME} common.cpp synmem.cpp)
add_library(lib::public ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}
        PUBLIC ${PROJECT_SOURCE_DIR}
        )

#GTEST
enable_testing()
file(GLOB TEST_SRCS synmem_test.cpp)
add_executable(
        synmem_test
        ${TEST_SRCS}
)
target_include_directories(synmem_test
        PUBLIC ${PROJECT_SOURCE_DIR}
        )
target_link_libraries(
        synmem_test
        lib::public
        glog::glog
        ${GTEST_LIBRARIES}
        ${GTEST_MAIN_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(synmem_test)