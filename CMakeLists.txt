# SPDX-FileCopyrightText: (c) 2026 Joel Winarske
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.21)
project(water_ffi LANGUAGES CXX C)

# The plugin is an out-of-tree FFI module, not bound to the drm-cxx/ivi-homescreen
# C++17 floor (baseline note). C++23, Tier A unconditionally.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(WATER_BUILD_SPIKES "Build the throwaway GPU spikes" ON)
option(WATER_BUILD_NATIVE "Build the engine-slice + FFI surface" OFF)

enable_testing()

find_package(Vulkan REQUIRED)

# GLM is header-only; prefer a config package, fall back to the system include.
find_package(glm QUIET)
if(NOT glm_FOUND)
  find_path(GLM_INCLUDE_DIR glm/glm.hpp REQUIRED)
  add_library(glm::glm INTERFACE IMPORTED)
  set_target_properties(glm::glm PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${GLM_INCLUDE_DIR}")
endif()

# Vendored single-header deps (stb_image).
add_library(water_third_party INTERFACE)
target_include_directories(water_third_party INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

if(WATER_BUILD_NATIVE)
  add_subdirectory(native)
endif()

if(WATER_BUILD_SPIKES)
  enable_testing()
  add_subdirectory(test/spikes)
endif()
