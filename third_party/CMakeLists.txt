add_library(gco INTERFACE)
target_include_directories(gco INTERFACE gco-v3.0)

set(STK_USE_CUDA ${DF_USE_CUDA} CACHE BOOL "")
set(STK_BUILD_WITH_DEBUG_INFO ON CACHE BOOL "")
set(STK_ENABLE_FAST_MATH ${ENABLE_FAST_MATH} CACHE BOOL "")

add_subdirectory(stk)
