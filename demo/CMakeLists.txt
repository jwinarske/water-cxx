# SPDX-FileCopyrightText: (c) 2026 Joel Winarske
# SPDX-License-Identifier: MIT
#
# Standalone Wayland windowed demo. Presents the engine's rendered frames to a
# Wayland compositor via an exported dma-buf (zwp_linux_dmabuf_v1) on an XDG
# toplevel. Demo-only: it pulls in the wayland-cxx-scanner submodule for the
# type-safe Wayland C++ bindings and is gated behind WATER_BUILD_WAYLAND_DEMO.

set(_wlcxx ${CMAKE_SOURCE_DIR}/external/wayland-cxx-scanner)
if(NOT EXISTS ${_wlcxx}/CMakeLists.txt)
  message(FATAL_ERROR "wayland-cxx-scanner submodule missing — run: "
                      "git submodule update --init external/wayland-cxx-scanner")
endif()

# Build only the scanner tool + header-only framework from the submodule.
set(WAYLAND_CXX_SCANNER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WAYLAND_CXX_SCANNER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WAYLAND_CXX_WERROR OFF CACHE BOOL "" FORCE)
add_subdirectory(${_wlcxx} ${CMAKE_BINARY_DIR}/wayland-cxx-scanner EXCLUDE_FROM_ALL)
# Make the codegen function resolve the just-built scanner in this scope.
set(WAYLAND_CXX_SCANNER_EXECUTABLE "$<TARGET_FILE:wayland-cxx-scanner>")

find_package(PkgConfig REQUIRED)
pkg_check_modules(WLCLIENT REQUIRED IMPORTED_TARGET wayland-client)
pkg_check_modules(LIBDRM REQUIRED IMPORTED_TARGET libdrm)  # drm_fourcc.h
pkg_get_variable(WS_DIR wayland-scanner pkgdatadir)
pkg_get_variable(WP_DIR wayland-protocols pkgdatadir)

# Scene pipelines are created in the demo from embedded SPIR-V (same set the
# engine tests use).
include(EmbedShaders)
water_embed_shaders(
  HEADER ${CMAKE_CURRENT_BINARY_DIR}/gen/scene_shaders.h
  OUTVAR SCENE_SHADERS
  INCLUDES
    ${CMAKE_SOURCE_DIR}/shaders/surface_push.glsl
    ${CMAKE_SOURCE_DIR}/shaders/intersect.glsl
    ${CMAKE_SOURCE_DIR}/shaders/surface_common.glsl
    ${CMAKE_SOURCE_DIR}/shaders/water_common.glsl
  SHADERS
    ${CMAKE_SOURCE_DIR}/shaders/pool.vert
    ${CMAKE_SOURCE_DIR}/shaders/sphere.vert
    ${CMAKE_SOURCE_DIR}/shaders/debug_scene.frag
    ${CMAKE_SOURCE_DIR}/shaders/pool.frag
    ${CMAKE_SOURCE_DIR}/shaders/sphere.frag
    ${CMAKE_SOURCE_DIR}/shaders/water.vert
    ${CMAKE_SOURCE_DIR}/shaders/water_above.frag
    ${CMAKE_SOURCE_DIR}/shaders/water_below.frag)

add_executable(water_window water_window.cpp ${SCENE_SHADERS})
target_link_libraries(water_window PRIVATE water_core wayland-cxx::wayland-cxx
                      PkgConfig::WLCLIENT PkgConfig::LIBDRM)
target_include_directories(water_window PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_compile_definitions(water_window PRIVATE
                           WATER_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets")
target_compile_options(water_window PRIVATE -Wno-unused-parameter)

# Generate the type-safe client headers from the system protocol XMLs.
wayland_cxx_generate(PROTOCOL ${WS_DIR}/wayland.xml MODE client-header
                     OUTPUT gen/wayland_client.hpp TARGET water_window)
wayland_cxx_generate(PROTOCOL ${WP_DIR}/stable/xdg-shell/xdg-shell.xml
                     MODE client-header OUTPUT gen/xdg_shell_client.hpp TARGET water_window)
wayland_cxx_generate(PROTOCOL ${WP_DIR}/unstable/linux-dmabuf/linux-dmabuf-unstable-v1.xml
                     MODE client-header OUTPUT gen/linux_dmabuf_client.hpp TARGET water_window)
# Self-contained tables (no wl/ helper header for this protocol).
wayland_cxx_generate(PROTOCOL ${WP_DIR}/staging/linux-drm-syncobj/linux-drm-syncobj-v1.xml
                     MODE client-header OUTPUT gen/drm_syncobj_client.hpp
                     EMIT_INTERFACE_TABLES TARGET water_window)
