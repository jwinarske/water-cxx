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
// The water engine renders the scene on the same device; each frame is copied
// into the acquired slot and handed to the compositor.

#include "drm_syncobj_client.hpp"
#include "linux_dmabuf_client.hpp"
#include "wayland_client.hpp"
#include "xdg_shell_client.hpp"

extern "C" {
#include <drm_fourcc.h>  // DRM_FORMAT_ABGR8888, DRM_FORMAT_MOD_LINEAR
#include <linux/input-event-codes.h>  // BTN_LEFT, KEY_ESC
#include <unistd.h>                   // close()
#include <wayland-client-protocol.h>  // wl_*_interface symbols + capabilities
}

#include <wl/client_helpers.hpp>  // SetupHandler, BindHandler, RunEventLoop
#include <wl/display.hpp>         // DisplayHandle, RoundtripWithTimeout
#include <wl/linux_dmabuf.hpp>    // linux-dmabuf interface tables
#include <wl/registry.hpp>        // CRegistry
#include <wl/wl_ptr.hpp>          // WlPtr
#include <wl/xdg_shell.hpp>       // XDG interface tables + CRTP handlers

#include "water/camera.hpp"
#include "water/caustics_pass.hpp"
#include "water/device.hpp"
#include "water/heightfield_sim.hpp"
#include "water/mesh.hpp"
#include "water/scene_pass.hpp"
#include "water/skybox.hpp"
#include "water/texture.hpp"

#include "scene_shaders.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

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
const wl_interface& wl_seat_traits::wl_iface() noexcept {
  return wl_seat_interface;
}
const wl_interface& wl_pointer_traits::wl_iface() noexcept {
  return wl_pointer_interface;
}
const wl_interface& wl_keyboard_traits::wl_iface() noexcept {
  return wl_keyboard_interface;
}
}  // namespace wayland::client

namespace {
constexpr int kRoundtripTimeoutMs = 2000;
constexpr int kNumBuffers = 3;  // N-slot ring; >2 gives release-gating headroom
// DRM_FORMAT_ABGR8888 (LE bytes R,G,B,A) == VK_FORMAT_R8G8B8A8_UNORM, matching
// the engine's scene color format so the eventual blit needs no swizzle.
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

// wp_linux_drm_syncobj_* — explicit-sync (request-only, no events). The acquire
// timeline point (signaled by our GPU submit) is the compositor's acquire
// fence; the release point (signaled by the compositor) gates slot reuse.
class DrmSyncobjManagerHandler
    : public linux_drm_syncobj_v1::client::CWpLinuxDrmSyncobjManagerV1<
          DrmSyncobjManagerHandler> {};
class DrmSyncobjTimelineHandler
    : public linux_drm_syncobj_v1::client::CWpLinuxDrmSyncobjTimelineV1<
          DrmSyncobjTimelineHandler> {};
class DrmSyncobjSurfaceHandler
    : public linux_drm_syncobj_v1::client::CWpLinuxDrmSyncobjSurfaceV1<
          DrmSyncobjSurfaceHandler> {};

// What a left-drag does, decided by what the press ray hit (main.js model).
enum class Drag { None, Orbit, Ball, Drops };

// wl_pointer: orbit / drag-the-ball / paint-drops + wheel zoom. Forwards to App
// (methods defined after App).
class WlPointerHandler : public wayland::client::CWlPointer<WlPointerHandler> {
 public:
  App* app_ = nullptr;
  void OnMotion(uint32_t, wl_fixed_t sx, wl_fixed_t sy) override;
  void OnButton(uint32_t, uint32_t, uint32_t button, uint32_t state) override;
  void OnAxis(uint32_t, uint32_t axis, wl_fixed_t value) override;
  void OnLeave(uint32_t, wl_proxy*) override;
};

// wl_keyboard: ESC quits. Raw evdev keycodes (no xkbcommon needed here).
class WlKeyboardHandler
    : public wayland::client::CWlKeyboard<WlKeyboardHandler> {
 public:
  App* app_ = nullptr;
  void OnKey(uint32_t, uint32_t, uint32_t key, uint32_t state) override;
};

// wl_seat: creates the pointer / keyboard when the capabilities arrive.
class WlSeatHandler : public wayland::client::CWlSeat<WlSeatHandler> {
 public:
  App* app_ = nullptr;
  wl::WlPtr<WlPointerHandler> pointer_;
  wl::WlPtr<WlKeyboardHandler> keyboard_;

  void OnCapabilities(uint32_t caps) override {
    using namespace wayland::client;
    if ((caps & WL_SEAT_CAPABILITY_POINTER) && pointer_.IsNull() &&
        wl::SetupHandler(
            pointer_,
            wl::construct<wl_pointer_traits, wl_seat_traits::Op::GetPointer>(
                *this)))
      pointer_.Get()->app_ = app_;
    if ((caps & WL_SEAT_CAPABILITY_KEYBOARD) && keyboard_.IsNull() &&
        wl::SetupHandler(
            keyboard_,
            wl::construct<wl_keyboard_traits, wl_seat_traits::Op::GetKeyboard>(
                *this)))
      keyboard_.Get()->app_ = app_;
  }
};

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
  VkFence fence = VK_NULL_HANDLE;  // GPU-retired (implicit-sync path)
  uint64_t release_pt = 0;   // explicit-sync: compositor-released at this point
  bool held = false;         // attached, compositor not yet released it
  bool initialized = false;  // has been rendered at least once
};

class App {
 public:
  ~App();
  int Run();
  int RunShot(
      const char* path);  // headless: render a still to a PPM, no window

  void OnXdgSurfaceConfigure(uint32_t /*serial*/) { configured_ = true; }
  void OnToplevelConfigure(int32_t w, int32_t h) {
    if (w <= 0 || h <= 0)
      return;  // compositor leaves the size to us
    if (!live_) {
      width_ = w;  // initial configure, before resources exist
      height_ = h;
    } else if (w != width_ || h != height_) {
      pending_w_ = w;  // a real resize; applied at the next frame
      pending_h_ = h;
      resize_pending_ = true;
    }
  }
  void OnToplevelClose() { running_ = false; }
  void OnFrameReady(uint32_t time_ms);
  void OnBufferRelease(int slot);

  // Input (forwarded by the pointer / keyboard handlers).
  void OnPointerMotion(double x, double y);
  void OnPointerButton(uint32_t button, uint32_t state);
  void OnPointerAxis(double value);
  void OnPointerLeave() { drag_ = Drag::None; }
  void OnKey(uint32_t key, uint32_t state);

 private:
  bool BindGlobals();
  bool CreateWindow();
  bool InitVulkan();
  bool CreatePresentRing();
  bool CreateWlBuffer(Slot& s, int index);
  bool InitExplicitSync();  // wp_linux_drm_syncobj timelines, if advertised
  bool CreateTimeline(VkSemaphore& sem,
                      wl::WlPtr<DrmSyncobjTimelineHandler>& tl);
  bool InitEngine();            // the water renderer (sim, caustics, scene)
  bool CreateSceneResources();  // size-dependent: scene pass, skybox, pipelines
  void DestroySceneResources();
  void DestroyPresentRing();
  bool DoResize();  // apply a pending size change (recreate the size-dependent)
  void RecordEngine(VkCommandBuffer cmd);  // record one engine frame -> scene
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
  wl::WlPtr<WlSeatHandler> seat_;
  wl::WlPtr<DrmSyncobjManagerHandler> syncobj_mgr_;
  wl::WlPtr<DrmSyncobjSurfaceHandler> syncobj_surface_;
  wl::WlPtr<DrmSyncobjTimelineHandler> acquire_timeline_;
  wl::WlPtr<DrmSyncobjTimelineHandler> release_timeline_;

  uint32_t compositor_name_ = 0, compositor_ver_ = 0;
  uint32_t xdg_wm_base_name_ = 0, xdg_wm_base_ver_ = 0;
  uint32_t dmabuf_name_ = 0, dmabuf_ver_ = 0;
  uint32_t seat_name_ = 0, seat_ver_ = 0;
  uint32_t syncobj_name_ = 0, syncobj_ver_ = 0;

  // Explicit sync (wp_linux_drm_syncobj): two Vulkan timeline semaphores
  // exported as DRM syncobjs. We signal acquire_sem_; the compositor signals
  // release_sem_. explicit_sync_ is false when the protocol is absent (then the
  // implicit foreign-transfer path is used).
  bool explicit_sync_ = false;
  VkSemaphore acquire_sem_ = VK_NULL_HANDLE;
  VkSemaphore release_sem_ = VK_NULL_HANDLE;
  uint64_t acquire_value_ = 0, release_value_ = 0;
  int next_slot_ = 0;  // round-robin cursor (explicit-sync path)
  PFN_vkGetSemaphoreFdKHR get_semaphore_fd_ = nullptr;
  bool configured_ = false;
  bool running_ = true;
  bool deferred_ = false;  // a frame was due but no slot was free
  bool live_ = false;      // resources exist; configures are now resizes
  bool resize_pending_ = false;
  int width_ = 1024, height_ = 768;
  int pending_w_ = 0, pending_h_ = 0;

  std::optional<water::Device> device_;
  PFN_vkGetMemoryFdKHR get_memory_fd_ = nullptr;
  PFN_vkGetImageDrmFormatModifierPropertiesEXT get_mod_props_ = nullptr;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  Slot slots_[kNumBuffers];

  // ── Water engine (declared after device_ so they tear down before it) ──────
  VkSampler clamp_ = VK_NULL_HANDLE;
  VkSampler repeat_ = VK_NULL_HANDLE;
  water::Texture tiles_{};
  water::Texture sky_{};
  std::optional<water::HeightfieldSim> sim_;
  std::optional<water::CausticsPass> caustics_;
  std::optional<water::ScenePass> scene_;
  std::optional<water::Skybox> skybox_;
  water::Mesh plane_{}, pool_mesh_{}, sphere_{};
  VkPipeline pool_pipe_ = VK_NULL_HANDLE, sphere_pipe_ = VK_NULL_HANDLE;
  VkPipeline above_pipe_ = VK_NULL_HANDLE, below_pipe_ = VK_NULL_HANDLE;
  // One descriptor set per ping-pong heightfield image (index by current()).
  VkDescriptorSet water_set_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  VkDescriptorSet surf_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  water::OrbitCamera cam_;
  water::ScenePush push_{};
  uint64_t frame_ = 0;

  // Interaction state.
  static constexpr float kSphereR = 0.25f;
  glm::vec3 sphere_pos_{-0.4f, -0.75f, 0.2f};
  glm::vec3 sphere_prev_{-0.4f, -0.75f, 0.2f};  // last frame, for move_sphere
  std::vector<glm::vec2> pending_drops_;  // applied (and cleared) each frame
  Drag drag_ = Drag::None;
  double cur_x_ = 0, cur_y_ = 0, last_x_ = 0, last_y_ = 0;

  // A world-space ray through window pixel (px, py) for picking.
  [[nodiscard]] std::pair<glm::vec3, glm::vec3> PixelRay(double px,
                                                         double py) const;
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

// Global execution + memory dependency between two pipeline stages. Used
// between the engine passes (which the single per-frame submit no longer
// separates).
void MemBarrier(VkCommandBuffer cmd,
                VkPipelineStageFlags src_stage,
                VkPipelineStageFlags dst_stage,
                VkAccessFlags src_access,
                VkAccessFlags dst_access) {
  const VkMemoryBarrier mb{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                           .srcAccessMask = src_access,
                           .dstAccessMask = dst_access};
  vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 1, &mb, 0, nullptr, 0,
                       nullptr);
}

// A conservative serialize-everything barrier. The engine render passes declare
// no subpass->EXTERNAL dependency for an in-submit consumer (they were written
// for isolated submits), so their finalLayout transitions aren't covered by
// stage-scoped barriers; ALL_COMMANDS + MEMORY_WRITE/READ orders them
// correctly. The engine stages are sequential anyway, so this costs no real
// parallelism.
void FullBarrier(VkCommandBuffer cmd) {
  MemBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
             VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
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
    } else if (iface == wl_seat_traits::interface_name) {
      seat_name_ = name;
      seat_ver_ = ver;
    } else if (iface ==
               linux_drm_syncobj_v1::client::
                   wp_linux_drm_syncobj_manager_v1_traits::interface_name) {
      syncobj_name_ = name;
      syncobj_ver_ = ver;
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

  // wl_seat (optional): pointer + keyboard for interaction. A roundtrip lets
  // the capabilities arrive so the pointer / keyboard are created before we
  // present.
  if (seat_name_) {
    if (!wl::BindHandler<wl_seat_traits>(
            registry_, seat_, seat_name_,
            std::min(seat_ver_, wl_seat_traits::version)))
      return false;
    seat_.Get()->app_ = this;
    if (!wl::RoundtripWithTimeout(display_.Get(), kRoundtripTimeoutMs))
      return false;
  }

  // wp_linux_drm_syncobj (optional): explicit sync. Request-only, so attach
  // with no listener. The timelines + surface object are set up once Vulkan
  // exists.
  if (syncobj_name_) {
    using mgr =
        linux_drm_syncobj_v1::client::wp_linux_drm_syncobj_manager_v1_traits;
    if (wl_proxy* raw = registry_.Bind<mgr>(
            syncobj_name_, std::min(syncobj_ver_, mgr::version)))
      syncobj_mgr_.Attach(raw);
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
  get_semaphore_fd_ = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
      vkGetDeviceProcAddr(device_->handle(), "vkGetSemaphoreFdKHR"));
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
  return CreatePresentRing() && InitEngine() && InitExplicitSync();
}

// Set up wp_linux_drm_syncobj explicit sync if the compositor offers it: two
// Vulkan timeline semaphores exported as DRM syncobjs and imported as the
// acquire/release timelines, plus a syncobj surface object. Best-effort — on
// any failure we fall back to the implicit foreign-transfer path.
bool App::CreateTimeline(VkSemaphore& sem,
                         wl::WlPtr<DrmSyncobjTimelineHandler>& tl) {
  using mgr =
      linux_drm_syncobj_v1::client::wp_linux_drm_syncobj_manager_v1_traits;
  using tl_traits =
      linux_drm_syncobj_v1::client::wp_linux_drm_syncobj_timeline_v1_traits;
  VkDevice dev = device_->handle();
  const VkSemaphoreTypeCreateInfo type{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = 0};
  const VkExportSemaphoreCreateInfo exp{
      .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
      .pNext = &type,
      .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT};
  const VkSemaphoreCreateInfo ci{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &exp};
  if (vkCreateSemaphore(dev, &ci, nullptr, &sem) != VK_SUCCESS)
    return false;
  const VkSemaphoreGetFdInfoKHR gi{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
      .semaphore = sem,
      .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT};
  int fd = -1;
  if (get_semaphore_fd_(dev, &gi, &fd) != VK_SUCCESS || fd < 0)
    return false;
  // import_timeline: libwayland dups the fd while marshalling; close ours
  // after.
  wl_proxy* raw = wl::construct<tl_traits, mgr::Op::ImportTimeline>(
      *syncobj_mgr_.Get(), fd);
  close(fd);
  if (!raw)
    return false;
  tl.Attach(raw);
  return true;
}

bool App::InitExplicitSync() {
  if (std::getenv("WATER_NO_EXPLICIT_SYNC") || !get_semaphore_fd_ ||
      syncobj_mgr_.IsNull())
    return true;  // forced off, or protocol / entry point absent — implicit
  using mgr =
      linux_drm_syncobj_v1::client::wp_linux_drm_syncobj_manager_v1_traits;
  using surf =
      linux_drm_syncobj_v1::client::wp_linux_drm_syncobj_surface_v1_traits;
  if (!CreateTimeline(acquire_sem_, acquire_timeline_) ||
      !CreateTimeline(release_sem_, release_timeline_)) {
    std::fprintf(stderr,
                 "water_window: timeline syncobj setup failed; "
                 "using implicit sync\n");
    return true;
  }
  wl_proxy* raw = wl::construct<surf, mgr::Op::GetSurface>(
      *syncobj_mgr_.Get(), surface_.Get()->GetProxy());
  if (!raw)
    return true;
  syncobj_surface_.Attach(raw);
  explicit_sync_ = true;
  std::printf("water_window: explicit sync via wp_linux_drm_syncobj\n");
  return true;
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
  if (!live_)  // quiet on resize (DoResize logs the new size)
    std::printf("water_window: present ring (%d slots, %dx%d, stride=%u)\n",
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

// ── Water engine ─────────────────────────────────────────────────────────────

#define ASSET(name) (WATER_ASSET_DIR "/" name)
// Unwrap a water::Result into `dst`, logging + returning false on error.
#define ENG_TRY(dst, expr)                                                 \
  do {                                                                     \
    auto _r = (expr);                                                      \
    if (!_r) {                                                             \
      std::fprintf(stderr, "water_window: engine init: %.*s\n",            \
                   int(_r.error().where.size()), _r.error().where.data()); \
      return false;                                                        \
    }                                                                      \
    (dst) = std::move(*_r);                                                \
  } while (0)

bool App::InitEngine() {
  using namespace water;
  Device& dev = *device_;

  ENG_TRY(clamp_, make_sampler(dev, VK_FILTER_LINEAR,
                               VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f));
  ENG_TRY(repeat_,
          make_sampler(dev, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT,
                       VK_LOD_CLAMP_NONE));
  ENG_TRY(tiles_, load_texture_2d(dev, ASSET("tiles.jpg"), true));
  ENG_TRY(sky_, load_cubemap(dev, {ASSET("xpos.jpg"), ASSET("xneg.jpg"),
                                   ASSET("ypos.jpg"), ASSET("ypos.jpg"),
                                   ASSET("zpos.jpg"), ASSET("zneg.jpg")}));

  {
    auto r = HeightfieldSim::create(dev, 256);
    if (!r)
      return false;
    sim_.emplace(std::move(*r));
  }
  if (!dev.submit_now([&](VkCommandBuffer cmd) {
        sim_->clear(cmd);
        const glm::vec2 drops[] = {
            {0, 0}, {0.4f, 0.3f}, {-0.4f, -0.2f}, {0.3f, -0.4f}, {-0.3f, 0.4f}};
        for (auto d : drops)
          sim_->add_drop(cmd, d, 0.05f, 0.5f);
        for (int i = 0; i < 30; ++i)
          sim_->step(cmd);
      }))
    return false;

  {
    auto r = CausticsPass::create(dev, 1024, 2);
    if (!r)
      return false;
    caustics_.emplace(std::move(*r));
  }
  ENG_TRY(plane_, upload_mesh(dev, make_plane(200)));
  ENG_TRY(pool_mesh_, upload_mesh(dev, make_pool()));
  ENG_TRY(sphere_, upload_mesh(dev, make_sphere(32)));
  ENG_TRY(water_set_[0],
          caustics_->make_water_set(sim_->image(0).view, clamp_));
  ENG_TRY(water_set_[1],
          caustics_->make_water_set(sim_->image(1).view, clamp_));

  const glm::vec3 light = glm::normalize(glm::vec3(2.0f, 2.0f, -1.0f));
  push_.light = glm::vec4(light, 0.0f);
  if (!CreateSceneResources())
    return false;
  std::printf("water_window: engine ready (scene %dx%d)\n", width_, height_);
  return true;
}

// The window-size-dependent render resources: the scene pass, the skybox built
// against its render pass, the surface descriptor set, and the scene pipelines.
// Created at startup and recreated on resize (sim / caustics / textures, which
// are size-independent, are left alone).
bool App::CreateSceneResources() {
  using namespace water;
  Device& dev = *device_;
  {
    auto r = ScenePass::create(dev, uint32_t(width_), uint32_t(height_));
    if (!r)
      return false;
    scene_.emplace(std::move(*r));
  }
  {
    auto r = Skybox::create(dev, scene_->render_pass(), sky_, clamp_);
    if (!r)
      return false;
    skybox_.emplace(std::move(*r));
  }
  for (uint32_t i = 0; i < 2; ++i)
    ENG_TRY(surf_[i],
            scene_->make_surface_set({{{sim_->image(i).view, clamp_},
                                       {tiles_.view, repeat_},
                                       {caustics_->image(0).view, clamp_},
                                       {sky_.view, clamp_}}}));

  auto P = [&](std::span<const uint32_t> v, std::span<const uint32_t> f,
               VkCullModeFlags c) {
    return scene_->make_pipeline({.vert = v, .frag = f, .cull = c});
  };
  ENG_TRY(pool_pipe_, P(std::span(pool_vert_spv, pool_vert_spv_count),
                        std::span(pool_frag_spv, pool_frag_spv_count),
                        VK_CULL_MODE_BACK_BIT));
  ENG_TRY(sphere_pipe_, P(std::span(sphere_vert_spv, sphere_vert_spv_count),
                          std::span(sphere_frag_spv, sphere_frag_spv_count),
                          VK_CULL_MODE_NONE));
  ENG_TRY(above_pipe_,
          P(std::span(water_vert_spv, water_vert_spv_count),
            std::span(water_above_frag_spv, water_above_frag_spv_count),
            VK_CULL_MODE_FRONT_BIT));
  ENG_TRY(below_pipe_,
          P(std::span(water_vert_spv, water_vert_spv_count),
            std::span(water_below_frag_spv, water_below_frag_spv_count),
            VK_CULL_MODE_BACK_BIT));
  return true;
}

// Advance the sim and render one composited frame into scene_->color().
// The engine's multi-pass stages keep the tested per-stage submit boundaries;
// the present hand-off (below) is the zero-CPU-wait part.
// Record the whole engine frame (sim -> caustics -> scene) into one command
// buffer. The passes used to be three CPU-synced submits; here explicit memory
// barriers replace those submit boundaries, and a frame-start barrier
// serializes the shared targets against the previous (now pipelined) frame.
void App::RecordEngine(VkCommandBuffer cmd) {
  ++frame_;
  const glm::mat4 vp = cam_.view_projection(scene_->aspect());
  push_.vp = vp;
  push_.eye = glm::vec4(cam_.position(), 1.0f);
  push_.sphere = glm::vec4(sphere_pos_, kSphereR);

  // An idle ambient drop keeps the surface alive when no one is interacting.
  const float t = float(frame_) * 0.03f;
  if (frame_ % 60 == 0)
    pending_drops_.emplace_back(0.6f * std::sin(t * 1.3f),
                                0.6f * std::cos(t * 0.7f));

  // Order this frame's writes to the shared engine targets (heightfield,
  // caustics, scene color) after the previous frame's reads of them. Without
  // the old per-stage CPU waits, consecutive frames otherwise overlap on the
  // queue.
  FullBarrier(cmd);

  // sim: the ball displaces the water (move_sphere from last frame's position;
  // static = its volume held, dragging = a wake), then queued drops, then 2x
  // step. The swap count no longer needs to be even — we pick the descriptor
  // set bound to whichever ping-pong image ends up current below.
  sim_->move_sphere(cmd, sphere_prev_, sphere_pos_, kSphereR);
  sphere_prev_ = sphere_pos_;
  for (const glm::vec2& d : pending_drops_)
    sim_->add_drop(cmd, d, 0.04f, 0.5f);
  pending_drops_.clear();
  sim_->step(cmd);
  sim_->step(cmd);
  const uint32_t hf = (sim_->current().image == sim_->image(0).image) ? 0u : 1u;

  FullBarrier(cmd);  // heightfield writes -> caustics + scene sampling
  caustics_->render(cmd, 0, push_, plane_, water_set_[hf]);
  FullBarrier(cmd);  // caustics output -> scene fragment sampling

  scene_->begin(cmd, {0, 0, 0, 1});
  skybox_->record(cmd, glm::inverse(vp), cam_.position());
  scene_->draw(cmd, pool_pipe_, push_, pool_mesh_, surf_[hf]);
  scene_->draw(cmd, above_pipe_, push_, plane_, surf_[hf]);
  scene_->draw(cmd, below_pipe_, push_, plane_, surf_[hf]);
  scene_->draw(cmd, sphere_pipe_, push_, sphere_, surf_[hf]);
  scene_->end(cmd);

  FullBarrier(
      cmd);  // scene color write -> transfer read (present copy/readback)
}

int App::Acquire() {
  if (explicit_sync_) {
    // Round-robin. The per-slot fence gates command-buffer reuse on GPU-done;
    // overwriting the dma-buf waits on the compositor's read via the implicit
    // reservation fence engaged by the FOREIGN re-acquire. mutter signals no
    // release point or wl_buffer.release here, so we don't gate on those.
    const int i = next_slot_;
    next_slot_ = (next_slot_ + 1) % kNumBuffers;
    if (slots_[i].initialized)
      vkWaitForFences(device_->handle(), 1, &slots_[i].fence, VK_TRUE,
                      100'000'000);  // ~always already signalled
    return i;
  }
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
  if (resize_pending_ && !DoResize()) {
    running_ = false;
    return;
  }
  const int slot = Acquire();
  if (slot < 0) {
    deferred_ = true;  // FrameEconomy: hold; wl_buffer.release will serve it
    return;
  }
  Slot& s = slots_[slot];
  VkDevice dev = device_->handle();
  const uint32_t mine = device_->queue_family();

  // One command buffer per frame: the whole engine render, then the dma-buf
  // hand-off — submitted once with no CPU wait. Synchronization to the
  // compositor is the explicit acquire/release timeline when available, else
  // the implicit FOREIGN-transfer dma-buf fence.
  vkResetCommandBuffer(s.cmd, 0);
  const VkCommandBufferBeginInfo bi{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  vkBeginCommandBuffer(s.cmd, &bi);

  RecordEngine(s.cmd);  // sim + caustics + scene -> scene_->color()

  // Acquire the present image for writing. First use: from UNDEFINED. Reuse:
  // reclaim ownership from the compositor (FOREIGN) — the proper Vulkan
  // hand-off for an external consumer. The fence (when does the compositor
  // read) is the explicit acquire timeline below; implicitly, the dma-buf
  // reservation fence.
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
  // scene_->color() is TRANSFER_SRC_OPTIMAL and made available to the transfer
  // by RecordEngine's final barrier, so the same-size copy needs no src
  // barrier.
  const VkImageCopy region{
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .extent = {uint32_t(width_), uint32_t(height_), 1}};
  vkCmdCopyImage(s.cmd, scene_->color().image,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.image,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  // Release the slot to the compositor (FOREIGN) for scanout.
  ImageBarrier(s.cmd, s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
               VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, mine,
               VK_QUEUE_FAMILY_FOREIGN_EXT);
  vkEndCommandBuffer(s.cmd);

  if (explicit_sync_) {
    // Signal the acquire timeline when the GPU work retires; tell the
    // compositor to wait for it (acquire point) and to signal the release point
    // when done.
    const uint64_t acquire_pt = ++acquire_value_;
    const VkTimelineSemaphoreSubmitInfo tssi{
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &acquire_pt};
    const VkSubmitInfo si{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .pNext = &tssi,
                          .commandBufferCount = 1,
                          .pCommandBuffers = &s.cmd,
                          .signalSemaphoreCount = 1,
                          .pSignalSemaphores = &acquire_sem_};
    vkResetFences(dev, 1, &s.fence);  // also a GPU-done gate for cmd reuse
    if (vkQueueSubmit(device_->queue(), 1, &si, s.fence) != VK_SUCCESS) {
      running_ = false;
      return;
    }
    s.release_pt = ++release_value_;
    auto* so = syncobj_surface_.Get();
    so->SetAcquirePoint(acquire_timeline_.Get()->GetProxy(),
                        uint32_t(acquire_pt >> 32), uint32_t(acquire_pt));
    so->SetReleasePoint(release_timeline_.Get()->GetProxy(),
                        uint32_t(s.release_pt >> 32), uint32_t(s.release_pt));
  } else {
    vkResetFences(dev, 1, &s.fence);
    const VkSubmitInfo si{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .commandBufferCount = 1,
                          .pCommandBuffers = &s.cmd};
    if (vkQueueSubmit(device_->queue(), 1, &si, s.fence) != VK_SUCCESS) {
      running_ = false;
      return;
    }
    s.held = true;
  }

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

void App::DestroyPresentRing() {
  VkDevice dev = device_->handle();
  for (auto& s : slots_) {
    if (s.fence)
      vkDestroyFence(dev, s.fence, nullptr);
    if (s.cmd)
      vkFreeCommandBuffers(dev, pool_, 1, &s.cmd);
    if (s.image)
      vkDestroyImage(dev, s.image, nullptr);
    if (s.memory)
      vkFreeMemory(dev, s.memory, nullptr);
    if (s.dma_fd >= 0)
      close(s.dma_fd);
    s = Slot{};  // reset handles + lifecycle flags + the wl_buffer (destroyed)
  }
}

void App::DestroySceneResources() {
  skybox_.reset();  // its pipeline is built against the scene render pass
  scene_.reset();   // owns the scene pipelines and the surface set's pool
  pool_pipe_ = sphere_pipe_ = above_pipe_ = below_pipe_ = VK_NULL_HANDLE;
  surf_[0] = surf_[1] = VK_NULL_HANDLE;
}

// Apply a pending size change: recreate the size-dependent resources at the new
// dimensions. Infrequent, so a device wait-idle here is fine.
bool App::DoResize() {
  resize_pending_ = false;
  if (pending_w_ == width_ && pending_h_ == height_)
    return true;
  vkDeviceWaitIdle(device_->handle());
  DestroyPresentRing();
  DestroySceneResources();
  width_ = pending_w_;
  height_ = pending_h_;
  if (!CreateSceneResources() || !CreatePresentRing())
    return false;
  std::printf("water_window: resized to %dx%d\n", width_, height_);
  return true;
}

App::~App() {
  if (!device_)
    return;
  VkDevice dev = device_->handle();
  vkDeviceWaitIdle(dev);
  DestroyPresentRing();
  // Engine objects without RAII handles (pipelines/sets are owned by the
  // passes; the passes themselves tear down as members before device_).
  water::destroy_mesh(*device_, plane_);
  water::destroy_mesh(*device_, pool_mesh_);
  water::destroy_mesh(*device_, sphere_);
  water::destroy_texture(*device_, tiles_);
  water::destroy_texture(*device_, sky_);
  if (clamp_)
    vkDestroySampler(dev, clamp_, nullptr);
  if (repeat_)
    vkDestroySampler(dev, repeat_, nullptr);
  if (acquire_sem_)
    vkDestroySemaphore(dev, acquire_sem_, nullptr);
  if (release_sem_)
    vkDestroySemaphore(dev, release_sem_, nullptr);
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
  live_ = true;  // further xdg configures are now treated as resizes

  std::printf("water_window: presenting (close the window to quit)\n");
  RenderFrame();  // kickstart frame 0; subsequent frames driven by callbacks
  const bool ok = wl::RunEventLoop(
      display_.Get(), [this] { return !running_; }, "water_window");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Headless still: render the scene off-screen (no compositor) and write it as a
// binary PPM. Used to generate the README image and as a smoke test of the
// engine path without a Wayland session.
int App::RunShot(const char* path) {
  auto dev = water::Device::create({});
  if (!dev) {
    std::fprintf(stderr, "water_window: Device::create failed: %s\n",
                 dev.error().where.data());
    return EXIT_FAILURE;
  }
  device_.emplace(std::move(*dev));
  if (!InitEngine())
    return EXIT_FAILURE;
  // Warm the sim up (with periodic drops) so the surface is lively. Each frame
  // is one recorded-and-waited submit here (no compositor to pace us).
  for (int i = 0; i < 320 && running_; ++i)
    if (!device_->submit_now([&](VkCommandBuffer cmd) { RecordEngine(cmd); })) {
      running_ = false;
      break;
    }
  auto rgba = scene_->readback_color(*device_);
  if (!rgba) {
    std::fprintf(stderr, "water_window: readback failed\n");
    return EXIT_FAILURE;
  }
  FILE* fp = std::fopen(path, "wb");
  if (!fp) {
    std::fprintf(stderr, "water_window: cannot open %s\n", path);
    return EXIT_FAILURE;
  }
  std::fprintf(fp, "P6\n%d %d\n255\n", width_, height_);
  for (std::size_t i = 0; i < std::size_t(width_) * height_; ++i)
    std::fwrite(&(*rgba)[i * 4], 1, 3, fp);  // RGB, drop alpha
  std::fclose(fp);
  std::printf("water_window: wrote %s (%dx%d)\n", path, width_, height_);
  return EXIT_SUCCESS;
}

// ── Picking + interaction ────────────────────────────────────────────────────

// Nearest positive hit distance of ray o+t*d (d normalized) with a sphere, <0
// if it misses.
float HitSphere(glm::vec3 o, glm::vec3 d, glm::vec3 c, float r) {
  const glm::vec3 oc = o - c;
  const float b = glm::dot(oc, d);
  const float disc = b * b - (glm::dot(oc, oc) - r * r);
  if (disc < 0.0f)
    return -1.0f;
  const float s = std::sqrt(disc);
  const float t0 = -b - s;
  return t0 >= 0.0f ? t0 : -b + s;
}

std::pair<glm::vec3, glm::vec3> App::PixelRay(double px, double py) const {
  const float aspect = float(width_) / float(height_);
  const glm::mat4 inv = glm::inverse(cam_.view_projection(aspect));
  const float nx = 2.0f * float(px) / float(width_) - 1.0f;
  // Window y is top-down; the scene's negative-height viewport maps it to GL
  // clip y, so the same flip applies when unprojecting.
  const float ny = 1.0f - 2.0f * float(py) / float(height_);
  glm::vec4 a = inv * glm::vec4(nx, ny, 0.0f, 1.0f);  // near plane
  glm::vec4 b = inv * glm::vec4(nx, ny, 1.0f, 1.0f);  // far plane
  const glm::vec3 o = glm::vec3(a) / a.w;
  return {o, glm::normalize(glm::vec3(b) / b.w - o)};
}

void App::OnPointerMotion(double x, double y) {
  last_x_ = cur_x_;
  last_y_ = cur_y_;
  cur_x_ = x;
  cur_y_ = y;
  if (drag_ == Drag::Orbit) {
    cam_.angle_y_deg -= float(cur_x_ - last_x_) * 0.4f;
    cam_.angle_x_deg += float(cur_y_ - last_y_) * 0.4f;
    cam_.clamp();
    return;
  }
  if (drag_ != Drag::Ball && drag_ != Drag::Drops)
    return;
  const auto [o, d] = PixelRay(x, y);
  if (std::abs(d.y) < 1e-5f)
    return;
  if (drag_ == Drag::Ball) {
    // Slide the ball on its own horizontal plane.
    const float t = (sphere_pos_.y - o.y) / d.y;
    if (t > 0.0f) {
      const glm::vec3 p = o + t * d;
      const float lim = 1.0f - kSphereR;
      sphere_pos_.x = glm::clamp(p.x, -lim, lim);
      sphere_pos_.z = glm::clamp(p.z, -lim, lim);
    }
  } else {  // Drag::Drops — paint a trail of drops on the water plane.
    const float t = -o.y / d.y;
    if (t > 0.0f) {
      const glm::vec3 p = o + t * d;
      if (std::abs(p.x) <= 1.0f && std::abs(p.z) <= 1.0f)
        pending_drops_.emplace_back(p.x, p.z);
    }
  }
}

void App::OnPointerButton(uint32_t button, uint32_t state) {
  if (button != BTN_LEFT)
    return;
  if (state != WL_POINTER_BUTTON_STATE_PRESSED) {
    drag_ = Drag::None;
    return;
  }
  const auto [o, d] = PixelRay(cur_x_, cur_y_);
  const float ts = HitSphere(o, d, sphere_pos_, kSphereR);
  float tw = -1.0f;
  glm::vec2 hit{};
  if (std::abs(d.y) > 1e-5f) {
    const float t = -o.y / d.y;  // water rest plane y=0
    if (t > 0.0f) {
      const glm::vec3 p = o + t * d;
      if (std::abs(p.x) <= 1.0f && std::abs(p.z) <= 1.0f) {
        tw = t;
        hit = {p.x, p.z};
      }
    }
  }
  if (ts >= 0.0f && (tw < 0.0f || ts < tw))
    drag_ = Drag::Ball;  // grabbed the ball
  else if (tw >= 0.0f) {
    drag_ = Drag::Drops;
    pending_drops_.push_back(hit);  // splash where clicked
  } else
    drag_ = Drag::Orbit;
}

void App::OnPointerAxis(double value) {
  cam_.distance += float(value) * 0.05f;  // wheel: dolly the camera
  cam_.clamp();
}

void App::OnKey(uint32_t key, uint32_t state) {
  if (key == KEY_ESC && state == WL_KEYBOARD_KEY_STATE_PRESSED)
    running_ = false;
}

void WlPointerHandler::OnMotion(uint32_t, wl_fixed_t sx, wl_fixed_t sy) {
  if (app_)
    app_->OnPointerMotion(wl_fixed_to_double(sx), wl_fixed_to_double(sy));
}
void WlPointerHandler::OnButton(uint32_t,
                                uint32_t,
                                uint32_t button,
                                uint32_t state) {
  if (app_)
    app_->OnPointerButton(button, state);
}
void WlPointerHandler::OnAxis(uint32_t, uint32_t axis, wl_fixed_t value) {
  if (app_ && axis == WL_POINTER_AXIS_VERTICAL_SCROLL)
    app_->OnPointerAxis(wl_fixed_to_double(value));
}
void WlPointerHandler::OnLeave(uint32_t, wl_proxy*) {
  if (app_)
    app_->OnPointerLeave();
}
void WlKeyboardHandler::OnKey(uint32_t,
                              uint32_t,
                              uint32_t key,
                              uint32_t state) {
  if (app_)
    app_->OnKey(key, state);
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
  if (const char* shot = std::getenv("WATER_SHOT"))
    return app.RunShot(shot);
  return app.Run();
}
