# SPDX-FileCopyrightText: (c) 2026 Joel Winarske
# SPDX-License-Identifier: MIT
# Engine-slice core (STATIC) + FFI surface (SHARED, added later) and the headless
# GPU tests that exercise it.
include(EmbedShaders)

# Sim shaders (S0-S4) + shared FS-quad vert, baked into water_core.
water_embed_shaders(
  HEADER ${CMAKE_CURRENT_BINARY_DIR}/gen/sim_shaders.h
  OUTVAR SIM_SHADERS
  SHADERS
    ${CMAKE_SOURCE_DIR}/shaders/fullscreen.vert
    ${CMAKE_SOURCE_DIR}/shaders/sim_clear.frag
    ${CMAKE_SOURCE_DIR}/shaders/sim_drop.frag
    ${CMAKE_SOURCE_DIR}/shaders/sim_sphere.frag
    ${CMAKE_SOURCE_DIR}/shaders/sim_update.frag
    ${CMAKE_SOURCE_DIR}/shaders/sim_normal.frag)

water_embed_shaders(
  HEADER ${CMAKE_CURRENT_BINARY_DIR}/gen/skybox_shaders.h
  OUTVAR SKYBOX_SHADERS
  SHADERS
    ${CMAKE_SOURCE_DIR}/shaders/skybox.vert
    ${CMAKE_SOURCE_DIR}/shaders/skybox.frag)

water_embed_shaders(
  HEADER ${CMAKE_CURRENT_BINARY_DIR}/gen/caustics_shaders.h
  OUTVAR CAUSTICS_SHADERS
  INCLUDES
    ${CMAKE_SOURCE_DIR}/shaders/surface_push.glsl
    ${CMAKE_SOURCE_DIR}/shaders/intersect.glsl
  SHADERS
    ${CMAKE_SOURCE_DIR}/shaders/caustics.vert
    ${CMAKE_SOURCE_DIR}/shaders/caustics.frag)

add_library(water_core STATIC
  core/src/device.cpp
  core/src/fullscreen_pass.cpp
  core/src/heightfield_sim.cpp
  core/src/mesh.cpp
  core/src/scene_pass.cpp
  core/src/texture.cpp
  core/src/skybox.cpp
  core/src/caustics_pass.cpp
  ${SIM_SHADERS}
  ${SKYBOX_SHADERS}
  ${CAUSTICS_SHADERS})
target_include_directories(water_core
  PUBLIC core/include
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_link_libraries(water_core PUBLIC Vulkan::Vulkan glm::glm PRIVATE water_third_party)
target_compile_features(water_core PUBLIC cxx_std_23)
# Vulkan clip space: depth [0,1] (Y is flipped per-pass via a negative-height viewport).
target_compile_definitions(water_core PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)

# ----  golden test: gradient via FS-quad into RGBA8, read back, assert pixels ----
water_embed_shaders(
  HEADER ${CMAKE_CURRENT_BINARY_DIR}/gen/fill_shaders.h
  OUTVAR FILL_SHADERS
  SHADERS
    ${CMAKE_SOURCE_DIR}/shaders/fullscreen.vert
    ${CMAKE_SOURCE_DIR}/shaders/fill.frag)

add_executable(test_clear test/test_clear.cpp ${FILL_SHADERS})
target_include_directories(test_clear PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_link_libraries(test_clear PRIVATE water_core)
add_test(NAME clear COMMAND test_clear)
# Deterministic on lavapipe; CI runs it there. SKIP (77) when no Vulkan device.
set_tests_properties(clear PROPERTIES SKIP_RETURN_CODE 77)

# ---- gate: heightfield sim invariants (energy decay, unit normals, NaN trap) ----
add_executable(test_sim test/test_sim.cpp)
target_link_libraries(test_sim PRIVATE water_core)
add_test(NAME sim COMMAND test_sim)
set_tests_properties(sim PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

# ---- gate: scene infrastructure (meshes, camera/VP + Y-flip, depth occlusion) ----
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
add_executable(test_scene test/test_scene.cpp ${SCENE_SHADERS})
target_include_directories(test_scene PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_link_libraries(test_scene PRIVATE water_core)
add_test(NAME scene COMMAND test_scene)
set_tests_properties(scene PROPERTIES SKIP_RETURN_CODE 77)

# ---- gate: assets (tiles + sky cubemap, face-order verified) + skybox pass ----
add_executable(test_sky test/test_sky.cpp ${SCENE_SHADERS})
target_include_directories(test_sky PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_link_libraries(test_sky PRIVATE water_core water_third_party)
target_compile_definitions(test_sky PRIVATE WATER_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets")
add_test(NAME sky COMMAND test_sky)
set_tests_properties(sky PROPERTIES SKIP_RETURN_CODE 77)

# ---- gate: real ray-traced pool/sphere shading (getWallColor/getSphereColor) ----
add_executable(test_shading test/test_shading.cpp ${SCENE_SHADERS})
target_include_directories(test_shading PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_link_libraries(test_shading PRIVATE water_core)
target_compile_definitions(test_shading PRIVATE WATER_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets")
add_test(NAME shading COMMAND test_shading)
set_tests_properties(shading PROPERTIES SKIP_RETURN_CODE 77)

# ---- gate: water surface (above/below reflect/refract + sky) over the live sim ----
add_executable(test_water test/test_water.cpp ${SCENE_SHADERS})
target_include_directories(test_water PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_link_libraries(test_water PRIVATE water_core)
target_compile_definitions(test_water PRIVATE WATER_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets")
add_test(NAME water COMMAND test_water)
set_tests_properties(water PROPERTIES SKIP_RETURN_CODE 77)

# ---- gate: caustics pass + tracking gate (sphere shadow tracks the ball) ----
add_executable(test_caustics test/test_caustics.cpp ${SCENE_SHADERS})
target_include_directories(test_caustics PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_link_libraries(test_caustics PRIVATE water_core)
target_compile_definitions(test_caustics PRIVATE WATER_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets")
add_test(NAME caustics COMMAND test_caustics)
set_tests_properties(caustics PROPERTIES SKIP_RETURN_CODE 77)
