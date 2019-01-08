cmake_minimum_required(VERSION 3.13)
project(Pathfinder)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA QUIET REQUIRED)

add_executable(Pathfinder main.cpp kernel.cu)

include_directories(inc .)