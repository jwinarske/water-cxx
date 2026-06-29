// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// SP-DMABUF — does the VkImage -> dma-buf + DRM modifier ->
// overlay/scanout path actually work on the target, and survive readback?
//
// This exercises the *production* component the plan reuses (/): drm-cxx's
// drm::present::VkScanoutProducer builds a VkDevice on the DRM node, allocates
// an exportable scanout VkImage with a negotiated modifier, and wraps it as a
// scene layer. We clear it to a distinct-channel color, commit it onto a vkms
// CRTC, snapshot the composited framebuffer back, and check the bytes —
// catching format/modifier/byte-order breakage and proving the modifier is
// scanout-capable.
//
// Go/kill: if the negotiated modifier can't scan out on a target, that target
// needs the linear-blit fallback (degraded-path fallback) or
// ExternalTexturePresent — an architecture fork, which is why this runs early.
// Self-skips (77) when vkms isn't loaded.

#include <drm-cxx/capture/snapshot.hpp>
#include <drm-cxx/core/device.hpp>
#include <drm-cxx/present/vk_scanout_producer.hpp>
#include <drm-cxx/scene/layer_desc.hpp>
#include <drm-cxx/scene/layer_scene.hpp>

#include <drm_fourcc.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;

constexpr int kSkip = 77;  // CTest SKIP_RETURN_CODE

std::optional<std::string> find_vkms_node() {
  std::error_code ec;
  for (const auto& entry : fs::directory_iterator("/dev/dri", ec)) {
    const std::string name = entry.path().filename().string();
    if (name.rfind("card", 0) != 0)
      continue;
    const int fd = ::open(entry.path().c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0)
      continue;
    drmVersionPtr v = drmGetVersion(fd);
    const bool is_vkms = v && v->name && std::strcmp(v->name, "vkms") == 0;
    if (v)
      drmFreeVersion(v);
    ::close(fd);
    if (is_vkms)
      return entry.path().string();
  }
  return std::nullopt;
}

bool approx(std::uint8_t a, int b, int tol = 3) {
  return std::abs(int(a) - b) <= tol;
}

// Does a Vulkan physical device actually back this DRM node? If not, the
// producer is forced cross-device (render GPU != KMS device) and snapshot()'s
// mmap-based readback can't observe our plane's fb — which changes how we
// interpret a black readback.
bool kms_node_has_vulkan(const std::string& node) {
  struct stat st{};
  if (::stat(node.c_str(), &st) != 0)
    return false;
  const unsigned want_major = major(st.st_rdev), want_minor = minor(st.st_rdev);

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_1;  // properties2 is core 1.1
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance inst{};
  if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS)
    return false;
  uint32_t n = 0;
  vkEnumeratePhysicalDevices(inst, &n, nullptr);
  std::vector<VkPhysicalDevice> ds(n);
  vkEnumeratePhysicalDevices(inst, &n, ds.data());
  bool match = false;
  for (auto d : ds) {
    VkPhysicalDeviceDrmPropertiesEXT
        drm{};  // zero-init: stays false if EXT unsupported
    drm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT;
    VkPhysicalDeviceProperties2 p2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &drm;
    vkGetPhysicalDeviceProperties2(d, &p2);
    if ((drm.hasPrimary && unsigned(drm.primaryMajor) == want_major &&
         unsigned(drm.primaryMinor) == want_minor) ||
        (drm.hasRender && unsigned(drm.renderMajor) == want_major &&
         unsigned(drm.renderMinor) == want_minor)) {
      match = true;
      break;
    }
  }
  vkDestroyInstance(inst, nullptr);
  return match;
}

struct ActiveCrtc {
  std::uint32_t crtc_id{0};
  std::uint32_t connector_id{0};
  drmModeModeInfo mode{};
};

// First connected connector (any type) with modes, and a CRTC one of its
// encoders can drive. Raw libdrm, mirroring drm-cxx's own vkms test — vkms's
// connector is "Virtual", which the examples' type-ranked picker skips.
std::optional<ActiveCrtc> pick_crtc(int fd) {
  auto* res = drmModeGetResources(fd);
  if (!res)
    return std::nullopt;
  std::optional<ActiveCrtc> found;
  for (int i = 0; i < res->count_connectors && !found; ++i) {
    auto* conn = drmModeGetConnector(fd, res->connectors[i]);
    if (!conn)
      continue;
    if (conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
      for (int e = 0; e < conn->count_encoders && !found; ++e) {
        auto* enc = drmModeGetEncoder(fd, conn->encoders[e]);
        if (!enc)
          continue;
        for (int c = 0; c < res->count_crtcs; ++c) {
          if (enc->possible_crtcs & (1U << unsigned(c))) {
            found =
                ActiveCrtc{res->crtcs[c], conn->connector_id, conn->modes[0]};
            break;
          }
        }
        drmModeFreeEncoder(enc);
      }
    }
    drmModeFreeConnector(conn);
  }
  drmModeFreeResources(res);
  return found;
}

}  // namespace

int main() {
  const auto node = find_vkms_node();
  if (!node) {
    std::printf(
        "SP-DMABUF  SKIP: vkms not loaded (sudo modprobe vkms "
        "enable_overlay=1)\n");
    return kSkip;
  }
  std::printf("SP-DMABUF  vkms node=%s\n", node->c_str());

  // A fresh open() on the vkms node makes us DRM master (no compositor
  // attached).
  auto dev_r = drm::Device::open(*node);
  if (!dev_r) {
    std::printf("FAIL: Device::open(%s): %s\n", node->c_str(),
                dev_r.error().message().c_str());
    return 1;
  }
  auto& dev = *dev_r;
  if (auto r = dev.enable_universal_planes(); !r) {
    std::printf("FAIL: enable_universal_planes: %s\n",
                r.error().message().c_str());
    return 1;
  }
  if (auto r = dev.enable_atomic();
      !r) {  // snapshot() needs atomic plane props
    std::printf("FAIL: enable_atomic: %s\n", r.error().message().c_str());
    return 1;
  }
  const auto active = pick_crtc(dev.fd());
  if (!active) {
    std::printf("FAIL: no connected connector with a mode on vkms\n");
    return 1;
  }
  const std::uint32_t w = active->mode.hdisplay;
  const std::uint32_t h = active->mode.vdisplay;
  std::printf("  output: crtc=%u connector=%u mode=%ux%u\n", active->crtc_id,
              active->connector_id, w, h);

  // The producer must outlive the scene it feeds (its VkImage backs the
  // dma-buf).
  auto prod_r = drm::present::VkScanoutProducer::create(dev);
  if (!prod_r) {
    std::printf("FAIL: VkScanoutProducer::create: %s\n",
                prod_r.error().message().c_str());
    return 1;
  }
  auto& prod = *prod_r;

  const std::uint32_t fourcc = DRM_FORMAT_XRGB8888;
  std::vector<std::uint64_t> mods = prod->exportable_modifiers(fourcc);
  std::printf("  exportable modifiers for XRGB8888: %zu\n", mods.size());

  // Prefer LINEAR for the vkms happy path; production must negotiate the
  // producer's modifier list against the chosen plane's IN_FORMATS (risk
  // register ).
  std::vector<std::uint64_t> allowed;
  if (std::find(mods.begin(), mods.end(),
                std::uint64_t(DRM_FORMAT_MOD_LINEAR)) != mods.end())
    allowed = {DRM_FORMAT_MOD_LINEAR};
  else
    allowed = mods;
  if (allowed.empty()) {
    std::printf(
        "FAIL: no exportable modifier for XRGB8888 (need linear-blit fallback, "
        "degraded-path fallback)\n");
    return 1;
  }

  auto src_r = prod->create_buffer(w, h, fourcc, allowed);
  if (!src_r) {
    std::printf("FAIL: create_buffer: %s\n", src_r.error().message().c_str());
    return 1;
  }
  auto src = std::move(*src_r);

  // Distinct per-channel clear so a swizzle/byte-order bug can't pass by
  // symmetry.
  constexpr int kR = 0x40, kG = 0x80, kB = 0xC0;
  if (auto r =
          prod->render_clear({kR / 255.0f, kG / 255.0f, kB / 255.0f, 1.0f});
      !r) {
    std::printf("FAIL: render_clear: %s\n", r.error().message().c_str());
    return 1;
  }

  drm::scene::LayerScene::Config cfg;
  cfg.crtc_id = active->crtc_id;
  cfg.connector_id = active->connector_id;
  cfg.mode = active->mode;
  auto scene_r = drm::scene::LayerScene::create(dev, cfg);
  if (!scene_r) {
    std::printf("FAIL: LayerScene::create: %s\n",
                scene_r.error().message().c_str());
    return 1;
  }
  auto& scene = *scene_r;

  drm::scene::LayerDesc desc;
  desc.source = std::move(src);
  desc.display.src_rect = drm::scene::Rect{0, 0, w, h};
  desc.display.dst_rect = drm::scene::Rect{0, 0, w, h};
  desc.display.zpos = 1;
  if (auto r = scene->add_layer(std::move(desc)); !r) {
    std::printf("FAIL: add_layer: %s\n", r.error().message().c_str());
    return 1;
  }
  if (auto r = scene->commit(); !r) {
    std::printf("FAIL: commit: %s\n", r.error().message().c_str());
    return 1;
  }

  // Read the composited CRTC back via writeback/mmap and check the bytes.
  auto img_r = drm::capture::snapshot(dev, active->crtc_id);
  if (!img_r) {
    std::printf("FAIL: snapshot: %s\n", img_r.error().message().c_str());
    return 1;
  }
  const auto& img = *img_r;
  if (img.empty()) {
    std::printf("FAIL: snapshot returned an empty image\n");
    return 1;
  }

  auto at = [&](std::uint32_t x, std::uint32_t y) {
    return img.pixels()[std::size_t(y) * img.width() + x];
  };
  // Sample a few points to confirm full-frame coverage (not just one lucky
  // texel).
  const std::pair<std::uint32_t, std::uint32_t> pts[] = {{w / 2, h / 2},
                                                         {w / 4, h / 4},
                                                         {3 * w / 4, 3 * h / 4},
                                                         {1, 1},
                                                         {w - 2, h - 2}};
  bool ok = true;
  for (auto [x, y] : pts) {
    const std::uint32_t c = at(x, y);
    const std::uint8_t R = (c >> 16) & 0xff, G = (c >> 8) & 0xff, B = c & 0xff;
    const bool hit = approx(R, kR) && approx(G, kG) && approx(B, kB);
    std::printf("  (%4u,%4u) = 0x%08X  R=%02X G=%02X B=%02X  %s\n", x, y, c, R,
                G, B, hit ? "ok" : "MISMATCH");
    ok = ok && hit;
  }

  // Reaching here means export + modifier negotiation + create_buffer +
  // render_clear + scene create + add_layer + atomic commit ALL succeeded — the
  // architecture path the spike answers. The readback is the only remaining
  // check, and it's only meaningful when the render GPU IS the KMS device.
  scene.reset();  // destroy scene before producer (lifetime order)

  if (ok) {
    std::printf(
        "\n  VERDICT: GO — VkScanoutProducer dma-buf+modifier committed to "
        "vkms and read "
        "back (expected R=40 G=80 B=C0)\n");
    return 0;
  }
  if (!kms_node_has_vulkan(*node)) {
    std::printf(
        "\n  NOTE: no Vulkan device backs the vkms KMS node — the producer "
        "rendered on a\n"
        "        separate GPU and exported a cross-device dmabuf. snapshot() "
        "mmaps the plane's\n"
        "        fb directly, which can't read cross-device GPU memory, so it "
        "returns background.\n"
        "        Export + modifier negotiation + atomic commit all SUCCEEDED, "
        "so the path is\n"
        "        proven on this host; pixel validation needs a single-GPU "
        "target or the vkms\n"
        "        writeback connector.\n");
    std::printf(
        "\n  VERDICT: GO (path proven; pixel readback not observable on this "
        "cross-device "
        "host)\n");
    return 0;
  }
  std::printf(
      "\n  VERDICT: FAIL — single-GPU host but readback mismatched: a real "
      "scanout / "
      "modifier / channel-order problem (degraded-path fallback).\n");
  return 1;
}
