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
pkg_get_variable(WS_DIR wayland-scanner pkgdatadir)
pkg_get_variable(WP_DIR wayland-protocols pkgdatadir)

add_executable(water_window water_window.cpp)
target_link_libraries(water_window PRIVATE water_core wayland-cxx::wayland-cxx
                      PkgConfig::WLCLIENT)
target_compile_options(water_window PRIVATE -Wno-unused-parameter)

# Generate the type-safe client headers from the system protocol XMLs.
wayland_cxx_generate(PROTOCOL ${WS_DIR}/wayland.xml MODE client-header
                     OUTPUT gen/wayland_client.hpp TARGET water_window)
wayland_cxx_generate(PROTOCOL ${WP_DIR}/stable/xdg-shell/xdg-shell.xml
                     MODE client-header OUTPUT gen/xdg_shell_client.hpp TARGET water_window)
wayland_cxx_generate(PROTOCOL ${WP_DIR}/unstable/linux-dmabuf/linux-dmabuf-unstable-v1.xml
                     MODE client-header OUTPUT gen/linux_dmabuf_client.hpp TARGET water_window)
