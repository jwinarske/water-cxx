# SPDX-FileCopyrightText: (c) 2026 Joel Winarske
# SPDX-License-Identifier: MIT
include(EmbedShaders)

# ---- SP-SIM: compute vs fragment heightfield sim (self-contained, any Vulkan device) ----
water_embed_shaders(
  HEADER ${CMAKE_CURRENT_BINARY_DIR}/gen/sp_sim_shaders.h
  OUTVAR SP_SIM_SHADER_HEADER
  SHADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/sp_sim/fullscreen.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/sp_sim/sim_update.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/sp_sim/sim_update_naive.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/sp_sim/sim_update_tiled.comp)

add_executable(sp_sim sp_sim/sp_sim.cpp ${SP_SIM_SHADER_HEADER})
target_include_directories(sp_sim PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_link_libraries(sp_sim PRIVATE Vulkan::Vulkan)
add_test(NAME sp_sim COMMAND sp_sim)
# Compute may legitimately be slower on a given device (KILL=exit 2); don't fail the
# suite on that — it's a recorded decision, not a regression.
set_tests_properties(sp_sim PROPERTIES SKIP_RETURN_CODE 2)

# ---- SP-DMABUF: VkImage -> dma-buf -> drm-cxx -> vkms readback (needs prebuilt drm-cxx) ----
set(DRMCXX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../drm-cxx" CACHE PATH "sibling drm-cxx checkout")
# drm-cxx's detail/span.hpp + detail/expected.hpp shim selects std:: vs tcb/tl by the
# consumer's C++ standard (__cpp_lib_span / __cpp_lib_expected), and its CMake mirrors
# that into the build's ABI (src/CMakeLists.txt DRM_CXX_HAS_STD_SPAN). So the rule is:
# link the drm-cxx build that matches your standard. We are C++23, so link the C++23
# (std::span ABI) build — the same path the real plugin will use.
set(DRMCXX_BUILD "audit-cxx23" CACHE STRING "drm-cxx build dir (must match our C++ std)")

set(_drmcxx_so "${DRMCXX_ROOT}/${DRMCXX_BUILD}/src/libdrm-cxx.so")
set(_drmcxx_inc "${DRMCXX_ROOT}/${DRMCXX_BUILD}")  # holds the drm-cxx -> ../src symlink
if(EXISTS "${_drmcxx_so}" AND IS_DIRECTORY "${_drmcxx_inc}/drm-cxx")
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBDRM REQUIRED libdrm)
  add_executable(sp_dmabuf sp_dmabuf/sp_dmabuf.cpp)
  target_include_directories(sp_dmabuf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    "${_drmcxx_inc}"     # <drm-cxx/...> via the build's symlink
    "${DRMCXX_ROOT}/src" # resolves intra-header quote-includes ("core/...")
    ${LIBDRM_INCLUDE_DIRS})
  target_link_libraries(sp_dmabuf PRIVATE "${_drmcxx_so}" ${LIBDRM_LIBRARIES} Vulkan::Vulkan)
  # Same DRM_CXX_HAS_* feature set the .so was built with, so #if-guarded class layouts
  # agree. The span/expected flavor is handled by the shim at our C++23 default — no
  # manual matching needed (this was the user's point).
  set_target_properties(sp_dmabuf PROPERTIES BUILD_RPATH "${DRMCXX_ROOT}/${DRMCXX_BUILD}/src")
  target_compile_definitions(sp_dmabuf PRIVATE
    DRM_CXX_HAS_VULKAN=1 DRM_CXX_HAS_BLEND2D=1 DRM_CXX_HAS_EGL=1 DRM_CXX_HAS_EGL_STREAMS=1
    DRM_CXX_HAS_GSTREAMER=1 DRM_CXX_HAS_LIBSEAT=1 DRM_CXX_HAS_NVBUFSURFACE=0)
  add_test(NAME sp_dmabuf COMMAND sp_dmabuf)
  # Self-skips (exit 77) when no vkms/DRM node is available (e.g. headless CI without vkms).
  set_tests_properties(sp_dmabuf PROPERTIES SKIP_RETURN_CODE 77)
else()
  message(STATUS "water: drm-cxx not found at ${_drmcxx_so} — skipping sp_dmabuf "
                 "(set -DDRMCXX_ROOT / -DDRMCXX_BUILD)")
endif()
