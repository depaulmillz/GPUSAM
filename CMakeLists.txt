cmake_minimum_required(VERSION 3.18)

include(cmake/config.cmake)

project(IterativeMethods LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

setup_conan()

#set(BLA_VENDOR Apple)
#find_package(BLAS REQUIRED)

enable_testing()

add_library(iterativeMethods INTERFACE)
target_include_directories(iterativeMethods INTERFACE include ${CONAN_INCLUDE_DIRS})
target_link_libraries(iterativeMethods INTERFACE ${CONAN_LIBS})

add_library(gpuIterativeMethods INTERFACE)
#target_compile_options(gpuIterativeMethods INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler '-fsanitize=address'>)
#target_link_libraries(gpuIterativeMethods INTERFACE asan)
target_link_libraries(gpuIterativeMethods INTERFACE iterativeMethods cublas cusparse cusolver)


add_executable(gmres src/gmres.cc)
target_link_libraries(gmres PUBLIC iterativeMethods)

add_executable(testLinalg src/testLinalg.cu)
target_link_libraries(testLinalg PUBLIC gpuIterativeMethods)
add_test(testLinalg testLinalg)

add_executable(testSam src/testSam.cu)
target_link_libraries(testSam PUBLIC gpuIterativeMethods)
add_test(testSam testSam)

add_executable(gpuGmres src/gpuGmres.cu)
target_link_libraries(gpuGmres PUBLIC gpuIterativeMethods)

add_executable(gpuSam src/gpuSam.cu)
target_link_libraries(gpuSam PUBLIC gpuIterativeMethods)

add_executable(gpuSamProf src/gpuSam.cu)
target_link_libraries(gpuSamProf PUBLIC gpuIterativeMethods)
target_compile_options(gpuSamProf PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler "-g -pg" -fno-builtin>)
target_compile_options(gpuSamProf PUBLIC $<$<COMPILE_LANGUAGE:CXX>: -g -pg -fno-builtin>)

