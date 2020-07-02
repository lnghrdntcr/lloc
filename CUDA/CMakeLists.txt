cmake_minimum_required(VERSION 3.17)
project(CUDA CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS "-lboost_program_options  --optimize -Wall")

add_executable(CUDA main.cu utils/types.h utils/rapidcsv.h utils/format_dataset.h utils/config.h lloc/lloc.h utils/utils.h)

set_target_properties(
        CUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)