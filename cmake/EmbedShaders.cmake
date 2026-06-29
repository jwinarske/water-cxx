# SPDX-FileCopyrightText: (c) 2026 Joel Winarske
# SPDX-License-Identifier: MIT
# EmbedShaders.cmake — offline GLSL -> SPIR-V -> embedded C header
#
# Pipeline per shader:  glslc  ->  spirv-opt -O (optional)  ->  spirv-val (optional)
# then all blobs are baked into one header as `static const uint32_t <name>_spv[]`.
#
# glslc is required. spirv-opt / spirv-val are looked up but optional: when absent the
# step is skipped (glslc already emits valid SPIR-V), so a bare shaderc install builds.
#
# Usage:
#   include(EmbedShaders)
#   water_embed_shaders(
#     HEADER  ${CMAKE_CURRENT_BINARY_DIR}/generated/shaders.h
#     SHADERS sim_update.comp sim_update.frag fullscreen.vert
#     OUTVAR  WATER_SHADER_HEADER)   # set to the header path
#   target_include_directories(<tgt> PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)

find_program(WATER_GLSLC glslc REQUIRED)
find_program(WATER_SPIRV_OPT spirv-opt)
find_program(WATER_SPIRV_VAL spirv-val)

if(WATER_SPIRV_OPT)
  message(STATUS "water: spirv-opt found (${WATER_SPIRV_OPT}) — SPIR-V will be -O optimized")
else()
  message(STATUS "water: spirv-opt not found — embedding unoptimized glslc output")
endif()
if(WATER_SPIRV_VAL)
  message(STATUS "water: spirv-val found (${WATER_SPIRV_VAL}) — SPIR-V will be validated")
endif()

# Vulkan env target for the optimizer; bump as drivers require.
set(WATER_SPIRV_TARGET_ENV "vulkan1.2" CACHE STRING "spirv-opt/glslc --target-env")

function(water_embed_shaders)
  # INCLUDES: extra .glsl files #include'd by the shaders; listed as DEPENDS so editing
  # a shared header triggers a recompile (glslc resolves the #include relative to source).
  cmake_parse_arguments(ARG "" "HEADER;OUTVAR" "SHADERS;INCLUDES" ${ARGN})
  if(NOT ARG_HEADER OR NOT ARG_SHADERS)
    message(FATAL_ERROR "water_embed_shaders: HEADER and SHADERS are required")
  endif()

  set(spv_files "")
  set(spv_names "")
  foreach(src ${ARG_SHADERS})
    get_filename_component(src_abs "${src}" ABSOLUTE)
    # symbol name: sim_update.comp -> sim_update_comp
    get_filename_component(fname "${src}" NAME)
    string(REPLACE "." "_" sym "${fname}")
    set(spv "${CMAKE_CURRENT_BINARY_DIR}/generated/${fname}.spv")

    # glslc -> (spirv-opt) -> (spirv-val), each step in-place on ${spv}
    set(steps COMMAND "${WATER_GLSLC}" --target-env=${WATER_SPIRV_TARGET_ENV} -O
              -c "${src_abs}" -o "${spv}")
    if(WATER_SPIRV_OPT)
      list(APPEND steps COMMAND "${WATER_SPIRV_OPT}" -O --target-env=${WATER_SPIRV_TARGET_ENV}
                                "${spv}" -o "${spv}")
    endif()
    if(WATER_SPIRV_VAL)
      list(APPEND steps COMMAND "${WATER_SPIRV_VAL}" --target-env ${WATER_SPIRV_TARGET_ENV} "${spv}")
    endif()

    add_custom_command(
      OUTPUT "${spv}"
      ${steps}
      DEPENDS "${src_abs}" ${ARG_INCLUDES}
      COMMENT "compiling shader ${fname}"
      VERBATIM)
    list(APPEND spv_files "${spv}")
    list(APPEND spv_names "${sym}")
  endforeach()

  # Bake all blobs into one header.
  string(REPLACE ";" "|" spv_files_arg "${spv_files}")
  string(REPLACE ";" "|" spv_names_arg "${spv_names}")
  add_custom_command(
    OUTPUT "${ARG_HEADER}"
    COMMAND "${CMAKE_COMMAND}"
            -DOUT=${ARG_HEADER}
            "-DSPV=${spv_files_arg}"
            "-DNAMES=${spv_names_arg}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GenShaderBlob.cmake"
    DEPENDS ${spv_files}
    COMMENT "embed SPIR-V -> ${ARG_HEADER}"
    VERBATIM)

  if(ARG_OUTVAR)
    set(${ARG_OUTVAR} "${ARG_HEADER}" PARENT_SCOPE)
  endif()
endfunction()
