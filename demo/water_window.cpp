// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
//
// Standalone Wayland windowed demo. Opens an XDG toplevel via the
// wayland-cxx-scanner framework and presents frames to the compositor through
// an exported dma-buf (zwp_linux_dmabuf_v1) rendered on the engine's Vulkan
// device.
//
// The present ring follows the drm-cxx external-rotating-producer discipline:
//   - N producer-owned, exportable dma-buf slots (BufferRing).
//   - A slot is reusable only once the compositor returns it
//   (wl_buffer.release)
//     and its GPU work has retired (per-slot fence) — no per-frame CPU stall.
//   - If no slot is free when a frame is due, the frame is deferred and served
//     from wl_buffer.release (FrameEconomy: hold, don't drop).
//   - Ownership is handed to / reclaimed from VK_QUEUE_FAMILY_FOREIGN_EXT so
//   the
//     compositor's read waits on our GPU writes via implicit dma-buf fencing —
//     the producer-side acquire fence, zero CPU wait.
// (The fully-explicit acquire fence is the linux explicit-sync protocol;
// implicit fencing via the foreign transfer is its idiomatic equivalent here.)
//
// This stage presents an animated clear colour to validate the path; the water
// engine blits its scene into each slot on top of this next.

#include "linux_dmabuf_client.hpp"
#include "wayland_client.hpp"
#include "xdg_shell_client.hpp"

extern "C" {
#include <drm_fourcc.h>  // DRM_FORMAT_ABGR8888, DRM_FORMAT_MOD_LINEAR
#include <unistd.h>      // close()
#include <wayland-client-protocol.h>  // wl_*_interface symbols
}

#include <wl/client_helpers.hpp>  // SetupHandler, BindHandler, RunEventLoop
#include <wl/display.hpp>         // DisplayHandle, RoundtripWithTimeout
#include <wl/linux_dmabuf.hpp>    // linux-dmabuf interface tables
#include <wl/registry.hpp>        // CRegistry
#include <wl/wl_ptr.hpp>          // WlPtr
#include <wl/xdg_shell.hpp>       // XDG interface tables + CRTP handlers

#include "water/device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string_view>

// Core-Wayland wl_interface tables come from libwayland-client; map the
// generated traits onto them. (xdg-shell + linux-dmabuf tables come from wl/.)
namespace wayland::client {
const wl_interface& wl_compositor_traits::wl_iface() noexcept {
  return wl_compositor_interface;
}
const wl_interface& wl_surface_traits::wl_iface() noexcept {
  return wl_surface_interface;
}
const wl_interface& wl_callback_traits::wl_iface() noexcept {
  return wl_callback_interface;
}
const wl_interface& wl_buffer_traits::wl_iface() noexcept {
  return wl_buffer_interface;
}
}  // namespace wayland::client

namespace {
constexpr int kRoundtripTimeoutMs = 2000;
constexpr int kNumBuffers = 3;  // N-slot ring; >2 gives release-gating headroom
// DRM_FORMAT_ABGR8888 (LE bytes R,G,B,A) == VK_FORMAT_R8G8B8A8_UNORM, matching
// the engine's scene colour format so the eventual blit needs no swizzle.
constexpr uint32_t kDrmFormat = DRM_FORMAT_ABGR8888;
constexpr VkFormat kVkFormat = VK_FORMAT_R8G8B8A8_UNORM;

class App;

class WlCompositorHandler
    : public wayland::client::CWlCompositor<WlCompositorHandler> {};
class WlSurfaceHandler : public wayland::client::CWlSurface<WlSurfaceHandler> {
};

// Paces frame production: a fresh wl_callback per frame; done -> render next.
class WlCallbackHandler
    : public wayland::client::CWlCallback<WlCallbackHandler> {
 public:
  App* app_ = nullptr;
  void OnDone(uint32_t time_ms) override;
};

// Returns a ring slot to the producer when the compositor is done reading it.
class WlBufferHandler : public wayland::client::CWlBuffer<WlBufferHandler> {
 public:
  App* app_ = nullptr;
  int slot_ = -1;
  void OnRelease() override;  // defined after App
};

// Checks the compositor advertises our DRM format.
class LinuxDmabufHandler
    : public linux_dmabuf_unstable_v1::client::CZwpLinuxDmabufV1<
          LinuxDmabufHandler> {
 public:
  uint32_t desired_format = kDrmFormat;
  bool has_format = false;
  void OnFormat(uint32_t format) override {
    if (format == desired_format)
      has_format = true;
  }
  void OnModifier(uint32_t format, uint32_t, uint32_t) override {
    if (format == desired_format)
      has_format = true;
  }
};

class LinuxBufferParamsHandler
    : public linux_dmabuf_unstable_v1::client::CZwpLinuxBufferParamsV1<
          LinuxBufferParamsHandler> {};

// One ring slot: an exportable, linear, dma-buf-backed present image plus its
// per-slot command buffer / fence and lifecycle state.
struct Slot {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  int dma_fd = -1;
  uint32_t stride = 0;
  uint32_t offset = 0;
  uint64_t modifier = 0;
  wl::WlPtr<WlBufferHandler> wl_buffer;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;  // signals when this slot's GPU work retires
  bool held = false;               // attached, compositor not yet released it
  bool initialized = false;        // has been rendered at least once
};

class App {
 public:
  ~App();
  int Run();

  void OnXdgSurfaceConfigure(uint32_t /*serial*/) { configured_ = true; }
  void OnToplevelConfigure(int32_t w, int32_t h) {
    if (w > 0 && h > 0) {
      width_ = w;
      height_ = h;
    }
  }
  void OnToplevelClose() { running_ = false; }
  void OnFrameReady(uint32_t time_ms);
  void OnBufferRelease(int slot);

 private:
  bool BindGlobals();
  bool CreateWindow();
  bool InitVulkan();
  bool CreatePresentRing();
  bool CreateWlBuffer(Slot& s, int index);
  int Acquire();  // a free slot (released + GPU-retired), or -1
  void RenderFrame();
  void RequestFrameCallback();

  wl::DisplayHandle display_;
  wl::CRegistry registry_;
  wl::WlPtr<WlCompositorHandler> compositor_;
  wl::WlPtr<WlSurfaceHandler> surface_;
  wl::WlPtr<wl::XdgWmBaseHandler> xdg_wm_base_;
  wl::WlPtr<wl::XdgSurfaceHandler<App>> xdg_surface_;
  wl::WlPtr<wl::XdgToplevelHandler<App>> xdg_toplevel_;
  wl::WlPtr<LinuxDmabufHandler> linux_dmabuf_;
  wl::WlPtr<WlCallbackHandler> frame_callback_;

  uint32_t compositor_name_ = 0, compositor_ver_ = 0;
  uint32_t xdg_wm_base_name_ = 0, xdg_wm_base_ver_ = 0;
  uint32_t dmabuf_name_ = 0, dmabuf_ver_ = 0;
  bool configured_ = false;
  bool running_ = true;
  bool deferred_ = false;  // a frame was due but no slot was free
  int width_ = 1024, height_ = 768;

  std::optional<water::Device> device_;
  PFN_vkGetMemoryFdKHR get_memory_fd_ = nullptr;
  PFN_vkGetImageDrmFormatModifierPropertiesEXT get_mod_props_ = nullptr;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  Slot slots_[kNumBuffers];
};

// ── Vulkan helpers ───────────────────────────────────────────────────────────

void ImageBarrier(VkCommandBuffer cmd,
                  VkImage img,
                  VkImageLayout from,
                  VkImageLayout to,
                  VkAccessFlags src_access,
                  VkAccessFlags dst_access,
                  VkPipelineStageFlags src_stage,
                  VkPipelineStageFlags dst_stage,
                  uint32_t src_qf = VK_QUEUE_FAMILY_IGNORED,
                  uint32_t dst_qf = VK_QUEUE_FAMILY_IGNORED) {
  const VkImageMemoryBarrier b{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = src_access,
      .dstAccessMask = dst_access,
      .oldLayout = from,
      .newLayout = to,
      .srcQueueFamilyIndex = src_qf,
      .dstQueueFamilyIndex = dst_qf,
      .image = img,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };
  vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1,
                       &b);
}

// ── Wayland setup ────────────────────────────────────────────────────────────

bool App::BindGlobals() {
  using namespace wayland::client;
  using namespace xdg_shell::client;
  using dmabuf_traits =
      linux_dmabuf_unstable_v1::client::zwp_linux_dmabuf_v1_traits;

  registry_.OnGlobal([this](wl::CRegistry&, const uint32_t name,
                            const std::string_view iface, const uint32_t ver) {
    if (iface == wl_compositor_traits::interface_name) {
      compositor_name_ = name;
      compositor_ver_ = ver;
    } else if (iface == xdg_wm_base_traits::interface_name) {
      xdg_wm_base_name_ = name;
      xdg_wm_base_ver_ = ver;
    } else if (iface == dmabuf_traits::interface_name) {
      dmabuf_name_ = name;
      dmabuf_ver_ = ver;
    }
  });

  if (!registry_.Create(display_.Get()) ||
      !wl::RoundtripWithTimeout(display_.Get(), kRoundtripTimeoutMs)) {
    std::fprintf(stderr, "water_window: registry scan failed\n");
    return false;
  }
  if (!compositor_name_ || !xdg_wm_base_name_ || !dmabuf_name_) {
    std::fprintf(stderr,
                 "water_window: missing wl_compositor / xdg_wm_base / "
                 "zwp_linux_dmabuf_v1\n");
    return false;
  }

  if (wl_proxy* raw = registry_.Bind<wl_compositor_traits>(
          compositor_name_,
          std::min(compositor_ver_, wl_compositor_traits::version))) {
    compositor_.Attach(raw);
  } else {
    return false;
  }
  if (!wl::BindHandler<xdg_wm_base_traits>(registry_, xdg_wm_base_,
                                           xdg_wm_base_name_, xdg_wm_base_ver_))
    return false;
  // Cap at v3: v4+ drops the legacy format/modifier events in favour of
  // dmabuf_feedback, which we don't use here.
  constexpr uint32_t kDmaBufVersion = 3;
  if (!wl::BindHandler<dmabuf_traits>(
          registry_, linux_dmabuf_, dmabuf_name_,
          std::min({dmabuf_ver_, dmabuf_traits::version, kDmaBufVersion})))
    return false;

  // Roundtrip so format/modifier advertisements arrive before we check support.
  if (!wl::RoundtripWithTimeout(display_.Get(), kRoundtripTimeoutMs))
    return false;
  if (!linux_dmabuf_.Get()->has_format) {
    std::fprintf(stderr,
                 "water_window: compositor does not advertise DRM_FORMAT "
                 "ABGR8888\n");
    return false;
  }
  return true;
}

bool App::CreateWindow() {
  using namespace wayland::client;
  using namespace xdg_shell::client;

  if (wl_proxy* raw = wl::construct<wl_surface_traits,
                                    wl_compositor_traits::Op::CreateSurface>(
          *compositor_.Get())) {
    surface_.Get()->_SetProxy(raw);
  } else {
    return false;
  }
  if (!wl::SetupHandler(xdg_surface_,
                        wl::construct<xdg_surface_traits,
                                      xdg_wm_base_traits::Op::GetXdgSurface>(
                            *xdg_wm_base_.Get(), surface_.Get()->GetProxy())))
    return false;
  xdg_surface_.Get()->app_ = this;
  if (!wl::SetupHandler(xdg_toplevel_,
                        wl::construct<xdg_toplevel_traits,
                                      xdg_surface_traits::Op::GetToplevel>(
                            *xdg_surface_.Get())))
    return false;
  xdg_toplevel_.Get()->app_ = this;
  xdg_toplevel_.Get()->SetTitle("water");
  xdg_toplevel_.Get()->SetAppId("org.water-ffi.water-window");

  surface_.Get()->Commit();  // trigger the initial configure
  return wl::RoundtripWithTimeout(display_.Get(), kRoundtripTimeoutMs);
}

// ── Vulkan / dma-buf present ─────────────────────────────────────────────────

bool App::InitVulkan() {
  auto dev = water::Device::create({.dmabuf_export = true});
  if (!dev) {
    std::fprintf(stderr, "water_window: Device::create failed: %s\n",
                 dev.error().where.data());
    return false;
  }
  device_.emplace(std::move(*dev));
  get_memory_fd_ = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
      vkGetDeviceProcAddr(device_->handle(), "vkGetMemoryFdKHR"));
  get_mod_props_ =
      reinterpret_cast<PFN_vkGetImageDrmFormatModifierPropertiesEXT>(
          vkGetDeviceProcAddr(device_->handle(),
                              "vkGetImageDrmFormatModifierPropertiesEXT"));
  if (!get_memory_fd_ || !get_mod_props_) {
    std::fprintf(stderr,
                 "water_window: required dma-buf entry points missing\n");
    return false;
  }
  const VkCommandPoolCreateInfo pci{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = device_->queue_family(),
  };
  if (vkCreateCommandPool(device_->handle(), &pci, nullptr, &pool_) !=
      VK_SUCCESS)
    return false;
  std::printf("water_window: device '%s'\n",
              device_->caps().device_name.c_str());
  return CreatePresentRing();
}

bool App::CreatePresentRing() {
  VkDevice dev = device_->handle();
  for (int i = 0; i < kNumBuffers; ++i) {
    Slot& s = slots_[i];
    // Allocate by DRM format modifier (LINEAR), exportable as a dma-buf.
    constexpr uint64_t kLinearMod = DRM_FORMAT_MOD_LINEAR;
    VkExternalMemoryImageCreateInfo ext_img{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    const VkImageDrmFormatModifierListCreateInfoEXT mod_list{
        .sType =
            VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
        .pNext = &ext_img,
        .drmFormatModifierCount = 1,
        .pDrmFormatModifiers = &kLinearMod,
    };
    const VkImageCreateInfo ici{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &mod_list,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = kVkFormat,
        .extent = {uint32_t(width_), uint32_t(height_), 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        // COLOR_ATTACHMENT for scanout-capable; TRANSFER_DST for clear / blit.
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    if (vkCreateImage(dev, &ici, nullptr, &s.image) != VK_SUCCESS)
      return false;

    VkMemoryRequirements reqs{};
    vkGetImageMemoryRequirements(dev, s.image, &reqs);
    auto mt = device_->find_memory_type(reqs.memoryTypeBits, 0);
    if (!mt)
      return false;
    // DRM-modifier images require a dedicated allocation.
    const VkMemoryDedicatedAllocateInfo dedicated{
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .image = s.image,
    };
    const VkExportMemoryAllocateInfo export_mem{
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .pNext = &dedicated,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    const VkMemoryAllocateInfo mai{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &export_mem,
        .allocationSize = reqs.size,
        .memoryTypeIndex = *mt,
    };
    if (vkAllocateMemory(dev, &mai, nullptr, &s.memory) != VK_SUCCESS)
      return false;
    if (vkBindImageMemory(dev, s.image, s.memory, 0) != VK_SUCCESS)
      return false;

    // Query the modifier the driver actually chose, and plane 0's layout.
    VkImageDrmFormatModifierPropertiesEXT mod_props{
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT};
    if (get_mod_props_(dev, s.image, &mod_props) != VK_SUCCESS)
      return false;
    s.modifier = mod_props.drmFormatModifier;
    const VkImageSubresource sub{VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT, 0, 0};
    VkSubresourceLayout sl{};
    vkGetImageSubresourceLayout(dev, s.image, &sub, &sl);
    s.stride = uint32_t(sl.rowPitch);
    s.offset = uint32_t(sl.offset);

    const VkMemoryGetFdInfoKHR fd_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = s.memory,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    if (get_memory_fd_(dev, &fd_info, &s.dma_fd) != VK_SUCCESS)
      return false;

    const VkCommandBufferAllocateInfo cbi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool_,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(dev, &cbi, &s.cmd) != VK_SUCCESS)
      return false;
    const VkFenceCreateInfo fci{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                .flags = VK_FENCE_CREATE_SIGNALED_BIT};
    if (vkCreateFence(dev, &fci, nullptr, &s.fence) != VK_SUCCESS)
      return false;

    if (!CreateWlBuffer(s, i))
      return false;
  }
  std::printf("water_window: present ring ready (%d slots, %dx%d, stride=%u)\n",
              kNumBuffers, width_, height_, slots_[0].stride);
  return true;
}

bool App::CreateWlBuffer(Slot& s, int index) {
  using namespace linux_dmabuf_unstable_v1::client;
  using namespace wayland::client;

  wl::WlPtr<LinuxBufferParamsHandler> params;
  if (wl_proxy* raw =
          wl::construct<zwp_linux_buffer_params_v1_traits,
                        zwp_linux_dmabuf_v1_traits::Op::CreateParams>(
              *linux_dmabuf_.Get())) {
    params.Attach(raw);
  } else {
    return false;
  }
  // plane 0, queried offset + modifier (high/low 32 bits).
  params.Get()->Add(s.dma_fd, 0u, s.offset, s.stride,
                    uint32_t(s.modifier >> 32u), uint32_t(s.modifier));
  wl_proxy* raw =
      wl::construct<wl_buffer_traits,
                    zwp_linux_buffer_params_v1_traits::Op::CreateImmed>(
          *params.Get(), int32_t(width_), int32_t(height_), kDrmFormat, 0u);
  if (!raw)
    return false;
  s.wl_buffer.Get()->_SetProxy(raw);
  s.wl_buffer.Get()->app_ = this;
  s.wl_buffer.Get()->slot_ = index;
  return true;
}

int App::Acquire() {
  for (int i = 0; i < kNumBuffers; ++i) {
    Slot& s = slots_[i];
    if (s.held)
      continue;  // compositor still owns it
    if (s.initialized &&
        vkGetFenceStatus(device_->handle(), s.fence) != VK_SUCCESS)
      continue;  // GPU still writing it
    return i;
  }
  return -1;
}

void App::RequestFrameCallback() {
  using wl_s = wayland::client::wl_surface_traits;
  using wl_c = wayland::client::wl_callback_traits;
  if (wl_proxy* raw = wl::construct<wl_c, wl_s::Op::Frame>(*surface_.Get())) {
    frame_callback_.Get()->app_ = this;
    frame_callback_.Get()->_SetProxy(raw);
  }
}

void App::RenderFrame() {
  const int slot = Acquire();
  if (slot < 0) {
    deferred_ = true;  // FrameEconomy: hold; wl_buffer.release will serve it
    return;
  }
  Slot& s = slots_[slot];
  VkDevice dev = device_->handle();
  const uint32_t mine = device_->queue_family();

  // Animated clear colour (cycling hue) to prove frames are flowing; the engine
  // scene blits in here next.
  static float t = 0.0f;
  t += 0.02f;
  const VkClearColorValue clear{{0.5f + 0.5f * std::sin(t),
                                 0.5f + 0.5f * std::sin(t + 2.094f),
                                 0.5f + 0.5f * std::sin(t + 4.188f), 1.0f}};

  vkResetCommandBuffer(s.cmd, 0);
  const VkCommandBufferBeginInfo bi{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  vkBeginCommandBuffer(s.cmd, &bi);

  // Acquire the slot for writing. First use: from UNDEFINED. Reuse: reclaim
  // ownership from the compositor (FOREIGN) — the matching half of the release
  // below, which is what arms implicit dma-buf fencing.
  if (!s.initialized) {
    ImageBarrier(
        s.cmd, s.image, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    s.initialized = true;
  } else {
    ImageBarrier(
        s.cmd, s.image, VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_QUEUE_FAMILY_FOREIGN_EXT, mine);
  }
  const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdClearColorImage(s.cmd, s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       &clear, 1, &range);
  // Release the slot to the compositor (FOREIGN): hands off with implicit sync.
  ImageBarrier(s.cmd, s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
               VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, mine,
               VK_QUEUE_FAMILY_FOREIGN_EXT);
  vkEndCommandBuffer(s.cmd);

  vkResetFences(dev, 1, &s.fence);
  const VkSubmitInfo si{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                        .commandBufferCount = 1,
                        .pCommandBuffers = &s.cmd};
  if (vkQueueSubmit(device_->queue(), 1, &si, s.fence) != VK_SUCCESS) {
    running_ = false;
    return;
  }

  s.held = true;
  RequestFrameCallback();  // before commit, same batch
  surface_.Get()->Attach(s.wl_buffer.Get()->GetProxy(), 0, 0);
  surface_.Get()->Damage(0, 0, width_, height_);
  surface_.Get()->Commit();
}

void App::OnFrameReady(uint32_t) {
  wl_proxy* spent = frame_callback_.Detach();
  if (spent)
    wl_proxy_destroy(spent);
  RenderFrame();
}

void App::OnBufferRelease(int slot) {
  if (slot >= 0 && slot < kNumBuffers)
    slots_[slot].held = false;
  if (deferred_) {
    deferred_ = false;
    RenderFrame();  // serve the held-back frame now that a slot is free
  }
}

App::~App() {
  if (!device_)
    return;
  VkDevice dev = device_->handle();
  vkDeviceWaitIdle(dev);
  for (auto& s : slots_) {
    if (s.fence)
      vkDestroyFence(dev, s.fence, nullptr);
    if (s.image)
      vkDestroyImage(dev, s.image, nullptr);
    if (s.memory)
      vkFreeMemory(dev, s.memory, nullptr);
    if (s.dma_fd >= 0)
      close(s.dma_fd);
  }
  if (pool_)
    vkDestroyCommandPool(dev, pool_, nullptr);
}

int App::Run() {
  if (!display_.Connect()) {
    std::fprintf(stderr, "water_window: wl_display_connect failed\n");
    return EXIT_FAILURE;
  }
  if (!BindGlobals() || !CreateWindow() || !InitVulkan())
    return EXIT_FAILURE;

  std::printf("water_window: presenting (close the window to quit)\n");
  RenderFrame();  // kickstart frame 0; subsequent frames driven by callbacks
  const bool ok = wl::RunEventLoop(
      display_.Get(), [this] { return !running_; }, "water_window");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

void WlCallbackHandler::OnDone(uint32_t time_ms) {
  if (app_)
    app_->OnFrameReady(time_ms);
}

void WlBufferHandler::OnRelease() {
  if (app_)
    app_->OnBufferRelease(slot_);
}

}  // namespace

int main() {
  App app;
  return app.Run();
}
