# water-ffi

A faithful C++23 FFI port of [WebGPU Water](https://github.com/willeastcott/webgpu-water-playcanvas)
(Evan Wallace's WebGL Water, ported to PlayCanvas) to an ivi-homescreen platform-view
plugin: a native Vulkan producer that renders the water sim into a dma-buf-exported
`VkImage`, composited by [drm-cxx](../drm-cxx) onto a KMS overlay plane, driven from
Dart over `native_comms` (control only — no pixels cross the FFI boundary).

## Status

The Vulkan engine-slice renders the full demo headless: the ping-pong heightfield
simulation, the ray-traced pool/sphere scene, the reflective/refractive water surface off
a sky cubemap, and the differential-area caustics. It runs behind a `PresentTarget`
seam and is validated on vkms / lavapipe; the ivi-homescreen embedder binding and the
dma-buf present path are deferred until the PlatformView registry lands as a homescreen
shared module.

## Layout

```
native/core/   engine-slice: device + caps, simulation, scene, water, caustics
native/test/   headless GPU tests (golden images, invariants, the tracking check)
shaders/       GLSL sources -> glslc -> spirv-opt -> embedded shaders.h blob
cmake/         EmbedShaders.cmake (offline SPIR-V build)
test/spikes/   throwaway GPU spikes: a sim benchmark and a dma-buf export check
third_party/   vendored single-header deps (stb_image)
assets/        pool tile + sky cubemap textures (MIT, from the original demo)
lib/           Dart bindings (added with the FFI surface)
```

## Building

```sh
cmake -S . -B build -G Ninja -DWATER_BUILD_NATIVE=ON -DWATER_BUILD_SPIKES=ON
cmake --build build -j
ctest --test-dir build            # headless GPU tests (skip cleanly without a device)

# force a specific Vulkan device (e.g. the software rasterizer)
WATER_VK_DEVICE=llvmpipe ctest --test-dir build
```

The dma-buf export spike additionally needs a vkms node and a prebuilt `drm-cxx`:

```sh
sudo modprobe vkms enable_overlay=1
./build/test/spikes/sp_dmabuf     # -DDRMCXX_ROOT=../drm-cxx, build dir audit-cxx23
```

## Toolchain notes

- `glslc` (shaderc) is required; `spirv-opt` / `spirv-val` are **optional** — the embed
  step degrades to glslc-only output if they are absent (see `cmake/EmbedShaders.cmake`).
- The engine-slice is C++23; the dma-buf spike links a sibling `drm-cxx` checkout built
  with the matching standard (its `std::span`/`std::expected` ABI must agree).
