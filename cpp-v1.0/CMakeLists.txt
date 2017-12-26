CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

project(naturalHHD)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

include_directories(./src)
file(GLOB_RECURSE Incs "./src/*h")

add_executable(naturalHHD ${Incs} src/main.cpp)

