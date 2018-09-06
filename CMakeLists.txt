cmake_minimum_required(VERSION 3.8)

# path separator
if(WIN32)
  set(SEP "\\;")
else()
  set(SEP ":")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

option(ENABLE_FAST_MATH "Enables fast math" OFF)
option(ENABLE_MICROPROFILE "Enables microprofile profiler" OFF)
option(ENABLE_NVTOOLSEXT "Enables nvtoolsext profiler" ON)

if (ENABLE_MICROPROFILE AND ENABLE_NVTOOLSEXT)
    message(FATAL_ERROR "Can't enable multiple profilers at the same time")
endif()

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting default build type: 'Release'")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

project(gpu-registration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (MSVC)
    add_definitions( -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_DEPRECATE )
    set(EXTRA_FLAGS "/FS")

    string(REGEX REPLACE "/W[1-3]" "/W4" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REGEX REPLACE "/W[1-3]" "/W4" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})

    set(EXTRA_FLAGS_RELEASE "/Zi")
    set(EXTRA_LINK_FLAGS_RELEASE "/DEBUG")

    if (ENABLE_FAST_MATH)
        set(EXTRA_FLAGS "${extra_flags} /fp:fast")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(EXTRA_FLAGS "-Wall -fPIC -march=native")
    set(EXTRA_FLAGS_RELEASE "-g1")

    if (ENABLE_FAST_MATH)
        set(EXTRA_FLAGS "${extra_flags} -ffast-math")
    endif()
endif()

add_definitions(-DSTK_USE_CUDA)
find_package(CUDA REQUIRED)

if (ENABLE_FAST_MATH)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math")
endif()

if (ENABLE_NVTOOLSEXT)
    find_package(NvToolsExt REQUIRED)
    add_definitions(-DENABLE_NVTOOLSEXT)
endif()

if (ENABLE_MICROPROFILE)
    add_definitions(-DENABLE_MICROPROFILE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_FLAGS_RELEASE}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${EXTRA_FLAGS_RELEASE}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_LINK_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_LINK_FLAGS_RELEASE}")

add_subdirectory(third_party)
add_subdirectory(src)

