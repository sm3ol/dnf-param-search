cmake_minimum_required(VERSION 3.15)

project(dnf_optimizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Correct include paths
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/param_data_structure/include
    ${CMAKE_SOURCE_DIR}/search
)

# Collect source files
file(GLOB PARAM_SRC
    ${CMAKE_SOURCE_DIR}/param_data_structure/src/*.cpp
)

file(GLOB SEARCH_SRC
    ${CMAKE_SOURCE_DIR}/search/*.cpp
)

add_executable(dnf_optimizer
    main.cpp
    ${PARAM_SRC}
    ${SEARCH_SRC}
)
