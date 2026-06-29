// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#include "water/heightfield_sim.hpp"

#include "sim_shaders.h"  // generated: fullscreen_vert_spv, sim_clear_frag_spv, ...

#include <cstring>
#include <span>

namespace water {

namespace {

// Push-constant layouts (std430; vec3 aligns to 16 — ).
struct DropPush {
  float center[2];
  float center2[2];
  float radius;
  float strength;
};  // 24 B
struct SpherePush {
  float old_c[3];
  float pad;
  float new_c[3];
  float radius;
};  // 32 B
struct DeltaPush {
  float delta[2];
};  // 8 B

template <class T>
std::span<const std::byte> as_bytes(const T& v) {
  return {reinterpret_cast<const std::byte*>(&v), sizeof(T)};
}

uint32_t bytes_per_texel(VkFormat f) {
  return f == VK_FORMAT_R32G32B32A32_SFLOAT ? 16u : 8u;  // 32F vs 16F (RGBA)
}

float half_to_float(uint16_t h) {
  uint32_t sign = uint32_t(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFu;
      f = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

}  // namespace

Result<HeightfieldSim> HeightfieldSim::create(const Device& dev,
                                              uint32_t size) {
  HeightfieldSim s;
  s.dev_ = &dev;
  s.size_ = size;
  s.format_ = dev.caps().heightfield_format;

  const VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                  VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  for (auto& im : s.img_) {
    im = WATER_TRY(dev.create_image(
        {.width = size, .height = size, .format = s.format_, .usage = usage}));
  }

  // Texel-aligned taps -> NEAREST is exact; clamp at the closed-pool boundary.
  const VkSamplerCreateInfo sci{
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
  };
  WATER_CHECK(vkCreateSampler(dev.handle(), &sci, nullptr, &s.sampler_),
              "vkCreateSampler");

  const VkDescriptorSetLayoutBinding b{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
  };
  const VkDescriptorSetLayoutCreateInfo dlci{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &b,
  };
  WATER_CHECK(
      vkCreateDescriptorSetLayout(dev.handle(), &dlci, nullptr, &s.dsl_),
      "vkCreateDescriptorSetLayout");

  const VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2};
  const VkDescriptorPoolCreateInfo dpci{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2,
      .poolSizeCount = 1,
      .pPoolSizes = &ps,
  };
  WATER_CHECK(vkCreateDescriptorPool(dev.handle(), &dpci, nullptr, &s.pool_),
              "vkCreateDescriptorPool");

  for (int i = 0; i < 2; ++i) {
    const VkDescriptorSetAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = s.pool_,
        .descriptorSetCount = 1,
        .pSetLayouts = &s.dsl_,
    };
    WATER_CHECK(vkAllocateDescriptorSets(dev.handle(), &ai, &s.set_[i]),
                "vkAllocateDescriptorSets");
    const VkDescriptorImageInfo dii{s.sampler_, s.img_[i].view,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkWriteDescriptorSet w{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = s.set_[i],
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &dii,
    };
    vkUpdateDescriptorSets(dev.handle(), 1, &w, 0, nullptr);
  }

  // Five passes share the FS-quad vert; the four sampling passes share the
  // source set. load_op DONT_CARE: every pass overwrites all texels; final
  // layout = sampleable.
  auto mk = [&](std::span<const uint32_t> frag, uint32_t push,
                VkDescriptorSetLayout sl) -> Result<FullscreenPass> {
    return FullscreenPass::create(
        dev, {
                 .color_format = s.format_,
                 .load_op = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                 .final_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 .vert_spirv =
                     std::span(fullscreen_vert_spv, fullscreen_vert_spv_count),
                 .frag_spirv = frag,
                 .push_constant_size = push,
                 .set_layout = sl,
             });
  };
  s.clear_ =
      WATER_TRY(mk(std::span(sim_clear_frag_spv, sim_clear_frag_spv_count), 0,
                   VK_NULL_HANDLE));
  s.drop_ = WATER_TRY(mk(std::span(sim_drop_frag_spv, sim_drop_frag_spv_count),
                         sizeof(DropPush), s.dsl_));
  s.sphere_ =
      WATER_TRY(mk(std::span(sim_sphere_frag_spv, sim_sphere_frag_spv_count),
                   sizeof(SpherePush), s.dsl_));
  s.update_ =
      WATER_TRY(mk(std::span(sim_update_frag_spv, sim_update_frag_spv_count),
                   sizeof(DeltaPush), s.dsl_));
  s.normal_ =
      WATER_TRY(mk(std::span(sim_normal_frag_spv, sim_normal_frag_spv_count),
                   sizeof(DeltaPush), s.dsl_));

  // All passes are render-pass-compatible (one attachment, same format) -> one
  // framebuffer per image, created against any pass's render pass.
  for (int i = 0; i < 2; ++i) {
    s.fb_[i] =
        WATER_TRY(s.update_.make_framebuffer(s.img_[i].view, size, size));
  }
  return s;
}

HeightfieldSim::~HeightfieldSim() {
  destroy();
}

void HeightfieldSim::destroy() noexcept {
  if (dev_) {
    VkDevice d = dev_->handle();
    for (auto fb : fb_) {
      if (fb)
        vkDestroyFramebuffer(d, fb, nullptr);
    }
    if (pool_)
      vkDestroyDescriptorPool(d, pool_, nullptr);
    if (dsl_)
      vkDestroyDescriptorSetLayout(d, dsl_, nullptr);
    if (sampler_)
      vkDestroySampler(d, sampler_, nullptr);
    for (auto& im : img_)
      dev_->destroy_image(im);
  }
  fb_ = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  pool_ = VK_NULL_HANDLE;
  dsl_ = VK_NULL_HANDLE;
  sampler_ = VK_NULL_HANDLE;
  dev_ = nullptr;
  // FullscreenPass members release themselves.
}

HeightfieldSim::HeightfieldSim(HeightfieldSim&& o) noexcept {
  *this = std::move(o);
}

HeightfieldSim& HeightfieldSim::operator=(HeightfieldSim&& o) noexcept {
  if (this != &o) {
    destroy();
    dev_ = o.dev_;
    size_ = o.size_;
    format_ = o.format_;
    read_idx_ = o.read_idx_;
    img_ = o.img_;
    fb_ = o.fb_;
    sampler_ = o.sampler_;
    dsl_ = o.dsl_;
    pool_ = o.pool_;
    set_ = o.set_;
    clear_ = std::move(o.clear_);
    drop_ = std::move(o.drop_);
    sphere_ = std::move(o.sphere_);
    update_ = std::move(o.update_);
    normal_ = std::move(o.normal_);
    o.dev_ = nullptr;
    o.img_ = {};
    o.fb_ = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    o.sampler_ = VK_NULL_HANDLE;
    o.dsl_ = VK_NULL_HANDLE;
    o.pool_ = VK_NULL_HANDLE;
    o.set_ = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  }
  return *this;
}

void HeightfieldSim::run_pass(VkCommandBuffer cmd,
                              const FullscreenPass& pass,
                              std::span<const std::byte> push,
                              bool use_source) {
  const int dst = 1 - read_idx_;
  VkDescriptorSet set = use_source ? set_[read_idx_] : VK_NULL_HANDLE;
  pass.record(cmd, fb_[dst], {size_, size_}, {0, 0, 0, 0}, push, set);

  // Ping-pong hazard fence: order this pass's color writes before the
  // next pass's source reads (RAW), and the previous source reads before this
  // write (WAR).
  const VkMemoryBarrier mb{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask =
          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
      .dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
  };
  const VkPipelineStageFlags stages =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  vkCmdPipelineBarrier(cmd, stages, stages, 0, 1, &mb, 0, nullptr, 0, nullptr);

  read_idx_ = dst;
}

void HeightfieldSim::clear(VkCommandBuffer cmd) {
  run_pass(cmd, clear_, {}, false);  // both ping-pong buffers -> defined/flat
  run_pass(cmd, clear_, {}, false);
}

void HeightfieldSim::add_line(VkCommandBuffer cmd,
                              glm::vec2 a,
                              glm::vec2 b,
                              float radius,
                              float strength) {
  const DropPush p{{a.x, a.y}, {b.x, b.y}, radius, strength};
  run_pass(cmd, drop_, as_bytes(p), true);
}

void HeightfieldSim::add_drop(VkCommandBuffer cmd,
                              glm::vec2 center,
                              float radius,
                              float strength) {
  add_line(cmd, center, center, radius, strength);
}

void HeightfieldSim::move_sphere(VkCommandBuffer cmd,
                                 glm::vec3 old_c,
                                 glm::vec3 new_c,
                                 float radius) {
  const SpherePush p{
      {old_c.x, old_c.y, old_c.z}, 0.0f, {new_c.x, new_c.y, new_c.z}, radius};
  run_pass(cmd, sphere_, as_bytes(p), true);
}

void HeightfieldSim::update(VkCommandBuffer cmd) {
  const DeltaPush p{{1.0f / float(size_), 1.0f / float(size_)}};
  run_pass(cmd, update_, as_bytes(p), true);
}

void HeightfieldSim::update_normals(VkCommandBuffer cmd) {
  const DeltaPush p{{1.0f / float(size_), 1.0f / float(size_)}};
  run_pass(cmd, normal_, as_bytes(p), true);
}

void HeightfieldSim::step(VkCommandBuffer cmd) {
  update(cmd);
  update(cmd);
  update_normals(cmd);
}

Result<std::vector<glm::vec4>> HeightfieldSim::readback(
    const Device& dev) const {
  const uint32_t bpt = bytes_per_texel(format_);
  const VkDeviceSize bytes = VkDeviceSize(size_) * size_ * bpt;
  Buffer host = WATER_TRY(
      dev.create_host_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT));
  VkImage image = img_[read_idx_].image;

  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
    auto barrier = [&](VkImageLayout from, VkImageLayout to, VkAccessFlags sa,
                       VkAccessFlags da, VkPipelineStageFlags ss,
                       VkPipelineStageFlags ds) {
      const VkImageMemoryBarrier b{
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .srcAccessMask = sa,
          .dstAccessMask = da,
          .oldLayout = from,
          .newLayout = to,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = image,
          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      vkCmdPipelineBarrier(cmd, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
    };
    barrier(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkBufferImageCopy region{
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {size_, size_, 1}};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           host.buffer, 1, &region);
    barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
  }));

  std::vector<glm::vec4> out(std::size_t(size_) * size_);
  if (format_ == VK_FORMAT_R32G32B32A32_SFLOAT) {
    std::memcpy(out.data(), host.mapped, bytes);
  } else {  // RGBA16F
    const auto* h = static_cast<const uint16_t*>(host.mapped);
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = {half_to_float(h[i * 4 + 0]), half_to_float(h[i * 4 + 1]),
                half_to_float(h[i * 4 + 2]), half_to_float(h[i * 4 + 3])};
    }
  }
  // host buffer is a local; dev owns nothing of it after copy — free via a
  // const_cast-free path:
  const_cast<Device&>(dev).destroy_buffer(host);
  return out;
}

}  // namespace water
