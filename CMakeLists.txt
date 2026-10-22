cmake_minimum_required(VERSION 3.15)
project(mapper-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)


add_library(mapper
    src/filter.cpp
    src/cover.cpp
    src/cluster.cpp
    src/graph.cpp
    src/embedding.cpp
    src/mapper.cpp
    src/two_step_mapper.cpp
)

target_include_directories(mapper PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanoflann
)
target_link_libraries(mapper PUBLIC Eigen3::Eigen)

add_executable(two_step_mapper_cli apps/two_step_mapper_cli.cpp)
target_link_libraries(two_step_mapper_cli PRIVATE mapper)