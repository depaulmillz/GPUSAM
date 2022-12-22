function(CHECK_CUDA_VERSION CUDA_MIN_VERSION CUDA_MAX_VERSION)
    if(NOT CUDA_VERSION)
        set(CUDA_VERSION ${CMAKE_CUDA_COMPILER_VERSION} PARENT_SCOPE)
    endif()
    if(NOT CUDA_TOOLKIT_ROOT_DIR)
        get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}/../.." ABSOLUTE)
    endif()
    
    if(CUDA_VERSION VERSION_LESS CUDA_MIN_VERSION AND CUDA_VERSION VERSION_GREATER CUDA_MAX_VERSION)
        message(FATAL_ERROR "CUDA ${CUDA_MIN_VERSION}-${CUDA_MAX_VERSION} required, found ${CUDA_VERSION}")
    endif()
endfunction()

function(SET_CONSISTENT_STANDARD STANDARD)
    set(CMAKE_CXX_STANDARD ${STANDARD})
    set(CMAKE_CUDA_STANDARD ${STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endfunction()

macro(SETUP_CONAN)
    message(STATUS "Running conan install ${CMAKE_SOURCE_DIR} -if ${CMAKE_BINARY_DIR} --build=missing")
    execute_process(
        COMMAND conan install ${CMAKE_SOURCE_DIR} -if ${CMAKE_BINARY_DIR} --build=missing
        RESULT_VARIABLE conan_code)
    if(NOT "${conan_code}" STREQUAL "0")
        message(FATAL_ERROR "Conan failed ${conan_code}")
    endif()
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
endmacro()
