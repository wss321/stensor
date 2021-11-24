project(proto)
find_package(Protobuf REQUIRED)
message("${PROTOBUF_INCLUDE_DIR}")
include_directories(${PROTOBUF_INCLUDE_DIR})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS tensor.proto module.proto)
add_library(${PROJECT_NAME} SHARED ${PROTO_SRCS} ${PROTO_HDRS})
add_library(lib::proto ALIAS ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} ${Protobuf_LIBRARIES})

