set(MAIN_SRCS
    "main.cpp"
    "block_change_flags.cpp"
    "block_change_flags.h"
    "registration_cpu.cpp"
    "registration_gpu.cpp"
)

if (ENABLE_MICROPROFILE)
    set(MAIN_SRCS 
        ${MAIN_SRCS}
        "profiler/microprofile_html.h"
        "profiler/microprofile.cpp"
        "profiler/microprofile.h"
    )
endif()

CUDA_ADD_EXECUTABLE(gpu-reg ${MAIN_SRCS})

if (ENABLE_NVTOOLSEXT)
    target_link_libraries(gpu-reg ${NvToolsExt_LIBRARIES})
    target_include_directories(gpu-reg PUBLIC ${NvToolsExt_INCLUDE_DIRS})
endif()

target_include_directories(gpu-reg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS})
target_link_libraries(gpu-reg stk gco)

if(WIN32)
  target_link_libraries(gpu-reg wsock32 ws2_32)
endif()

