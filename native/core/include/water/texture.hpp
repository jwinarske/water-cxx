// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// water/texture.hpp — image asset loading (assets.js): the repeat-wrapped
// mipmapped pool tiles (2D) and the sky cubemap (6 faces). Decoded with
// stb_image, staged to device-local RGBA8 (UNORM, no sRGB — GAMMA_NONE
// upstream), left in SHADER_READ_ONLY.
#pragma once

#include "water/device.hpp"
#include "water/result.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace water {

struct Texture {
  VkImage image = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t mip_levels = 1;
  uint32_t layers = 1;
};

// Load an RGBA8 2D texture, optionally generating a full mip chain (blit
// downsample).
[[nodiscard]] Result<Texture> load_texture_2d(const Device& dev,
                                              const char* path,
                                              bool mipmaps);

// Load a cubemap from 6 face paths in Vulkan layer order [+X,-X,+Y,-Y,+Z,-Z]
// (identical to PlayCanvas's order). The upstream asset set has no -Y face —
// pass +Y twice.
[[nodiscard]] Result<Texture> load_cubemap(
    const Device& dev,
    const std::array<const char*, 6>& faces);

// A 1x1 RGBA8 texture of a constant value. Used for the neutral caustics
// placeholder in
//  (caustic.r=1, caustic.g=1) before the real caustics pass lands in .
[[nodiscard]] Result<Texture> make_solid_texture(const Device& dev,
                                                 uint8_t r,
                                                 uint8_t g,
                                                 uint8_t b,
                                                 uint8_t a);

void destroy_texture(const Device& dev, Texture& tex) noexcept;

// Read mip 0 of one array layer back to host RGBA8 (cubemap face-order
// verification). Leaves the texture in SHADER_READ_ONLY_OPTIMAL.
[[nodiscard]] Result<std::vector<uint8_t>> readback_layer(const Device& dev,
                                                          const Texture& tex,
                                                          uint32_t layer);

// samplers: a linear-repeat-mip sampler (tiles) and a linear-clamp-no-mip
// sampler (everything else, explicit LOD 0). `max_lod` caps mip sampling (use
// VK_LOD_CLAMP_NONE).
[[nodiscard]] Result<VkSampler> make_sampler(const Device& dev,
                                             VkFilter filter,
                                             VkSamplerAddressMode address,
                                             float max_lod);

}  // namespace water
