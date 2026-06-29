// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#include "water/texture.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace water {

namespace {

struct RawImage {
  std::vector<uint8_t> rgba;  // tightly packed RGBA8
  int w = 0, h = 0;
};

Result<RawImage> decode(const char* path) {
  int w = 0, h = 0, c = 0;
  stbi_uc* p = stbi_load(path, &w, &h, &c, 4);
  if (!p)
    return fail(VK_ERROR_INITIALIZATION_FAILED, "stbi_load failed");
  RawImage img;
  img.w = w;
  img.h = h;
  img.rgba.assign(p, p + std::size_t(w) * h * 4);
  stbi_image_free(p);
  return img;
}

// Allocate a device-local image + view. Caller transitions/uploads.
Result<Texture> alloc_image(const Device& dev,
                            uint32_t w,
                            uint32_t h,
                            uint32_t mips,
                            uint32_t layers,
                            VkImageCreateFlags flags,
                            VkImageViewType view_type) {
  Texture t;
  t.width = w;
  t.height = h;
  t.mip_levels = mips;
  t.layers = layers;
  const VkImageCreateInfo ici{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = flags,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = t.format,
      .extent = {w, h, 1},
      .mipLevels = mips,
      .arrayLayers = layers,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };
  WATER_CHECK(vkCreateImage(dev.handle(), &ici, nullptr, &t.image),
              "vkCreateImage(texture)");
  VkMemoryRequirements req{};
  vkGetImageMemoryRequirements(dev.handle(), t.image, &req);
  const uint32_t mt = WATER_TRY(dev.find_memory_type(
      req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
  const VkMemoryAllocateInfo ai{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = req.size,
                                .memoryTypeIndex = mt};
  WATER_CHECK(vkAllocateMemory(dev.handle(), &ai, nullptr, &t.memory),
              "vkAllocateMemory(texture)");
  WATER_CHECK(vkBindImageMemory(dev.handle(), t.image, t.memory, 0),
              "vkBindImageMemory(texture)");

  const VkImageViewCreateInfo vci{
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = t.image,
      .viewType = view_type,
      .format = t.format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, layers},
  };
  WATER_CHECK(vkCreateImageView(dev.handle(), &vci, nullptr, &t.view),
              "vkCreateImageView(texture)");
  return t;
}

void barrier(VkCommandBuffer cmd,
             VkImage img,
             VkImageLayout from,
             VkImageLayout to,
             VkAccessFlags sa,
             VkAccessFlags da,
             VkPipelineStageFlags ss,
             VkPipelineStageFlags ds,
             uint32_t base_mip,
             uint32_t mips,
             uint32_t layers) {
  const VkImageMemoryBarrier b{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = sa,
      .dstAccessMask = da,
      .oldLayout = from,
      .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, base_mip, mips, 0,
                           layers},
  };
  vkCmdPipelineBarrier(cmd, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
}

}  // namespace

Result<Texture> load_texture_2d(const Device& dev,
                                const char* path,
                                bool mipmaps) {
  RawImage raw = WATER_TRY(decode(path));
  const auto w = uint32_t(raw.w), h = uint32_t(raw.h);
  const uint32_t mips =
      mipmaps ? uint32_t(std::floor(std::log2(float(std::max(w, h))))) + 1u
              : 1u;
  Texture t =
      WATER_TRY(alloc_image(dev, w, h, mips, 1, 0, VK_IMAGE_VIEW_TYPE_2D));

  Buffer staging = WATER_TRY(dev.create_host_buffer(
      raw.rgba.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT));
  std::memcpy(staging.mapped, raw.rgba.data(), raw.rgba.size());

  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
    // mip 0 <- staging
    barrier(cmd, t.image, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, mips, 1);
    const VkBufferImageCopy region{
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {w, h, 1}};
    vkCmdCopyBufferToImage(cmd, staging.buffer, t.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    // generate mips by successive linear blits
    auto mw = int32_t(w), mh = int32_t(h);
    for (uint32_t i = 1; i < mips; ++i) {
      barrier(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
              VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
              i - 1, 1, 1);
      const int32_t nw = std::max(mw / 2, 1), nh = std::max(mh / 2, 1);
      const VkImageBlit blit{
          .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1},
          .srcOffsets = {{0, 0, 0}, {mw, mh, 1}},
          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1},
          .dstOffsets = {{0, 0, 0}, {nw, nh, 1}},
      };
      vkCmdBlitImage(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                     VK_FILTER_LINEAR);
      barrier(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, i - 1, 1, 1);
      mw = nw;
      mh = nh;
    }
    // last mip is still TRANSFER_DST
    barrier(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, mips - 1, 1, 1);
  }));
  dev.destroy_buffer(staging);
  return t;
}

Result<Texture> load_cubemap(const Device& dev,
                             const std::array<const char*, 6>& faces) {
  std::array<RawImage, 6> raw;
  for (int i = 0; i < 6; ++i)
    raw[i] = WATER_TRY(decode(faces[i]));
  const auto w = uint32_t(raw[0].w), h = uint32_t(raw[0].h);
  const VkDeviceSize face_bytes = VkDeviceSize(w) * h * 4;
  // All six faces must share the same dimensions: the staging copy below reads
  // exactly face_bytes from each, so a mismatched face would over-read its
  // decoded buffer.
  for (int i = 1; i < 6; ++i)
    if (raw[i].w != raw[0].w || raw[i].h != raw[0].h)
      return fail(VK_ERROR_FORMAT_NOT_SUPPORTED,
                  "cubemap faces differ in size");

  Texture t = WATER_TRY(alloc_image(dev, w, h, 1, 6,
                                    VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
                                    VK_IMAGE_VIEW_TYPE_CUBE));
  Buffer staging = WATER_TRY(
      dev.create_host_buffer(face_bytes * 6, VK_BUFFER_USAGE_TRANSFER_SRC_BIT));
  for (int i = 0; i < 6; ++i)
    std::memcpy(static_cast<uint8_t*>(staging.mapped) + face_bytes * i,
                raw[i].rgba.data(), face_bytes);

  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
    barrier(cmd, t.image, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, 6);
    std::array<VkBufferImageCopy, 6> regions{};
    for (uint32_t i = 0; i < 6; ++i)
      regions[i] = {.bufferOffset = face_bytes * i,
                    .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, i, 1},
                    .imageExtent = {w, h, 1}};
    vkCmdCopyBufferToImage(cmd, staging.buffer, t.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regions.size(),
                           regions.data());
    barrier(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, 6);
  }));
  dev.destroy_buffer(staging);
  return t;
}

Result<Texture> make_solid_texture(const Device& dev,
                                   uint8_t r,
                                   uint8_t g,
                                   uint8_t b,
                                   uint8_t a) {
  Texture t = WATER_TRY(alloc_image(dev, 1, 1, 1, 1, 0, VK_IMAGE_VIEW_TYPE_2D));
  const uint8_t px[4] = {r, g, b, a};
  Buffer staging =
      WATER_TRY(dev.create_host_buffer(4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT));
  std::memcpy(staging.mapped, px, 4);
  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
    barrier(cmd, t.image, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, 1);
    const VkBufferImageCopy region{
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {1, 1, 1}};
    vkCmdCopyBufferToImage(cmd, staging.buffer, t.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    barrier(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, 1);
  }));
  dev.destroy_buffer(staging);
  return t;
}

void destroy_texture(const Device& dev, Texture& tex) noexcept {
  if (tex.view)
    vkDestroyImageView(dev.handle(), tex.view, nullptr);
  if (tex.image)
    vkDestroyImage(dev.handle(), tex.image, nullptr);
  if (tex.memory)
    vkFreeMemory(dev.handle(), tex.memory, nullptr);
  tex = {};
}

Result<std::vector<uint8_t>> readback_layer(const Device& dev,
                                            const Texture& tex,
                                            uint32_t layer) {
  const VkDeviceSize bytes = VkDeviceSize(tex.width) * tex.height * 4;
  Buffer host = WATER_TRY(
      dev.create_host_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT));
  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
    barrier(cmd, tex.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, tex.mip_levels, tex.layers);
    const VkBufferImageCopy region{
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, layer, 1},
        .imageExtent = {tex.width, tex.height, 1}};
    vkCmdCopyImageToBuffer(cmd, tex.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           host.buffer, 1, &region);
    barrier(
        cmd, tex.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
        VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, tex.mip_levels, tex.layers);
  }));
  std::vector<uint8_t> out(bytes);
  std::memcpy(out.data(), host.mapped, bytes);
  dev.destroy_buffer(host);
  return out;
}

Result<VkSampler> make_sampler(const Device& dev,
                               VkFilter filter,
                               VkSamplerAddressMode address,
                               float max_lod) {
  const VkSamplerCreateInfo sci{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = filter,
      .minFilter = filter,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = address,
      .addressModeV = address,
      .addressModeW = address,
      .maxLod = max_lod,
  };
  VkSampler s = VK_NULL_HANDLE;
  WATER_CHECK(vkCreateSampler(dev.handle(), &sci, nullptr, &s),
              "vkCreateSampler");
  return s;
}

}  // namespace water
