cmake_minimum_required(VERSION 3.6)
project(sop6)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES main.c)
add_executable(sop6 ${SOURCE_FILES})

target_link_libraries(sop6 lapacke lapack blas gfortran m)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/coefficients.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
