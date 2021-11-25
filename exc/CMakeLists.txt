cmake_minimum_required(VERSION 3.5)
project(train_mnist)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -gencode arch=compute_30,code=sm_30;
        -gencode arch=compute_35,code=sm_35;
        -gencode arch=compute_50,code=sm_50;
        -gencode arch=compute_52,code=sm_52;
        -gencode arch=compute_60,code=sm_60;
        -gencode arch=compute_61,code=sm_61;
        -gencode arch=compute_75,code=sm_75;
        -std=c++11;)

add_executable(${PROJECT_NAME} train_mnist.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${stensor_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME}
        lib::tensor
        lib::proto
        lib::public
        lib::optim
        )