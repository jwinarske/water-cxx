// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// Smoke test: exercise device + caps + the format gate +
// FS-quad pass + offscreen RGBA8 target + readback as one chain. Render a UV
// gradient into an RGBA8 target, copy it back to host memory, and assert the
// golden pixels. Deterministic on lavapipe (CI). Exit 77 = SKIP (no Vulkan
// device), 1 = fail.

#include "water/device.hpp"
#include "water/fullscreen_pass.hpp"

#include "fill_shaders.h"  // fullscreen_vert_spv, fill_frag_spv (+ _count)

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>

using namespace water;

namespace {
constexpr uint32_t kW = 256, kH = 128;

const char* fmt_name(VkFormat f) {
  switch (f) {
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return "RGBA32F";
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return "RGBA16F";
    default:
      return "<other>";
  }
}

bool near(uint8_t got, int want, int tol = 2) {
  return std::abs(int(got) - want) <= tol;
}

// The whole flow returns Result so failures propagate with context; main() maps
// it.
Result<int> run() {
  DeviceConfig cfg{};
  if (const char* sub = std::getenv("WATER_VK_DEVICE"))
    cfg.device_substr = sub;
  cfg.enable_validation = std::getenv("WATER_VK_VALIDATE") != nullptr;

  Device dev = WATER_TRY(Device::create(cfg));
  const Caps& c = dev.caps();
  std::printf(
      "  device=\"%s\"  class=%s  float32_filterable=%s  heightfield=%s\n",
      c.device_name.c_str(),
      c.device_class == DeviceClass::Mobile ? "Mobile" : "Desktop",
      c.float32_filterable ? "yes" : "no", fmt_name(c.heightfield_format));

  // Offscreen RGBA8 color target (scene target format).
  Image target = WATER_TRY(dev.create_image({
      .width = kW,
      .height = kH,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
  }));

  FullscreenPass pass = WATER_TRY(FullscreenPass::create(
      dev,
      {
          .color_format = VK_FORMAT_R8G8B8A8_UNORM,
          .load_op = VK_ATTACHMENT_LOAD_OP_CLEAR,
          .final_layout =
              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,  // ready for readback copy
          .vert_spirv =
              std::span(fullscreen_vert_spv, fullscreen_vert_spv_count),
          .frag_spirv = std::span(fill_frag_spv, fill_frag_spv_count),
          .push_constant_size = sizeof(float) * 2,
      }));
  VkFramebuffer fb = WATER_TRY(pass.make_framebuffer(target.view, kW, kH));
  Buffer readback = WATER_TRY(dev.create_host_buffer(
      VkDeviceSize(kW) * kH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT));

  const std::array<float, 2> extent_px{float(kW), float(kH)};
  const auto push = std::as_bytes(std::span(extent_px));

  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
    pass.record(cmd, fb, {kW, kH}, {0, 0, 0, 1}, push);

    // Render-pass finalLayout already moved the image to TRANSFER_SRC_OPTIMAL;
    // this barrier provides the color-write -> transfer-read execution+memory
    // dependency.
    const VkImageMemoryBarrier to_copy{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = target.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &to_copy);

    const VkBufferImageCopy region{
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {kW, kH, 1},
    };
    vkCmdCopyImageToBuffer(cmd, target.image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.buffer, 1, &region);

    const VkBufferMemoryBarrier to_host{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = readback.buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &to_host,
                         0, nullptr);
  }));

  // ---- verify the golden gradient: pixel(x,y) = ((x+0.5)/W, (y+0.5)/H, 0.25,
  // 1) ----
  const auto* px = static_cast<const uint8_t*>(readback.mapped);
  auto at = [&](uint32_t x, uint32_t y) {
    return px + (std::size_t(y) * kW + x) * 4;
  };
  struct Pt {
    uint32_t x, y;
  };
  int bad = 0;
  for (Pt p : {Pt{0, 0}, Pt{kW - 1, kH - 1}, Pt{kW / 2, kH / 2},
               Pt{kW / 4, 3 * kH / 4}}) {
    const int er = int(std::lround((p.x + 0.5) / kW * 255.0));
    const int eg = int(std::lround((p.y + 0.5) / kH * 255.0));
    const uint8_t* q = at(p.x, p.y);
    const bool ok =
        near(q[0], er) && near(q[1], eg) && near(q[2], 64) && near(q[3], 255);
    std::printf("  (%3u,%3u) got=%3u,%3u,%3u,%3u  want~%3d,%3d, 64,255  %s\n",
                p.x, p.y, q[0], q[1], q[2], q[3], er, eg,
                ok ? "ok" : "MISMATCH");
    bad += !ok;
  }

  dev.destroy_image(target);
  dev.destroy_buffer(readback);
  pass.destroy_framebuffer(fb);

  std::printf("\n  VERDICT: %s\n",
              bad == 0 ? "PASS — clear+FS-quad+readback golden matches"
                       : "FAIL — golden mismatch");
  return bad == 0 ? 0 : 1;
}

}  // namespace

int main() {
  Result<int> r = run();
  if (!r) {
    std::fprintf(stderr, "SKIP/ERROR: %.*s (%s)\n", int(r.error().where.size()),
                 r.error().where.data(), to_string(r.error().code));
    // No Vulkan device available -> SKIP (77); any other failure is a hard
    // error.
    return r.error().code == VK_ERROR_INITIALIZATION_FAILED ? 77 : 1;
  }
  return *r;
}
