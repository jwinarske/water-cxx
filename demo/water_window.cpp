// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
//
// Standalone Wayland windowed demo (work in progress): opens an XDG toplevel
// via the wayland-cxx-scanner framework. The dma-buf present path and the water
// engine are wired in on top of this scaffold.

#include "wayland_client.hpp"
#include "xdg_shell_client.hpp"

extern "C" {
#include <wayland-client-protocol.h>  // wl_*_interface symbols
}

#include <wl/client_helpers.hpp>  // SetupHandler, BindHandler, RunEventLoop
#include <wl/display.hpp>         // DisplayHandle, RoundtripWithTimeout
#include <wl/registry.hpp>        // CRegistry
#include <wl/wl_ptr.hpp>          // WlPtr
#include <wl/xdg_shell.hpp>       // XDG interface tables + CRTP handlers

#include <cstdint>
#include <cstdio>
#include <string_view>

// Core-Wayland wl_interface tables come from libwayland-client; map the
// generated traits onto them. (xdg-shell tables are provided by
// <wl/xdg_shell.hpp>.)
namespace wayland::client {
const wl_interface& wl_compositor_traits::wl_iface() noexcept {
  return wl_compositor_interface;
}
const wl_interface& wl_surface_traits::wl_iface() noexcept {
  return wl_surface_interface;
}
}  // namespace wayland::client

namespace {
constexpr int kRoundtripTimeoutMs = 2000;

class App;

class WlCompositorHandler
    : public wayland::client::CWlCompositor<WlCompositorHandler> {};
class WlSurfaceHandler : public wayland::client::CWlSurface<WlSurfaceHandler> {
};

class App {
 public:
  int Run();

  // Invoked by the XDG CRTP handlers.
  void OnXdgSurfaceConfigure(uint32_t /*serial*/) { configured_ = true; }
  void OnToplevelConfigure(int32_t w, int32_t h) {
    if (w > 0 && h > 0) {
      width_ = w;
      height_ = h;
    }
  }
  void OnToplevelClose() { running_ = false; }

 private:
  bool BindGlobals();
  bool CreateWindow();

  wl::DisplayHandle display_;
  wl::CRegistry registry_;
  wl::WlPtr<WlCompositorHandler> compositor_;
  wl::WlPtr<WlSurfaceHandler> surface_;
  wl::WlPtr<wl::XdgWmBaseHandler> xdg_wm_base_;
  wl::WlPtr<wl::XdgSurfaceHandler<App>> xdg_surface_;
  wl::WlPtr<wl::XdgToplevelHandler<App>> xdg_toplevel_;

  uint32_t compositor_name_ = 0, compositor_ver_ = 0;
  uint32_t xdg_wm_base_name_ = 0, xdg_wm_base_ver_ = 0;
  bool configured_ = false;
  bool running_ = true;
  int width_ = 1024, height_ = 768;
};

bool App::BindGlobals() {
  using namespace wayland::client;
  using namespace xdg_shell::client;

  registry_.OnGlobal([this](wl::CRegistry& /*r*/, const uint32_t name,
                            const std::string_view iface, const uint32_t ver) {
    if (iface == wl_compositor_traits::interface_name) {
      compositor_name_ = name;
      compositor_ver_ = ver;
    } else if (iface == xdg_wm_base_traits::interface_name) {
      xdg_wm_base_name_ = name;
      xdg_wm_base_ver_ = ver;
    }
  });

  if (!registry_.Create(display_.Get()) ||
      !wl::RoundtripWithTimeout(display_.Get(), kRoundtripTimeoutMs)) {
    std::fprintf(stderr, "water_window: registry scan failed\n");
    return false;
  }
  if (!compositor_name_ || !xdg_wm_base_name_) {
    std::fprintf(stderr, "water_window: missing wl_compositor / xdg_wm_base\n");
    return false;
  }

  if (wl_proxy* raw = registry_.Bind<wl_compositor_traits>(
          compositor_name_,
          std::min(compositor_ver_, wl_compositor_traits::version))) {
    compositor_.Attach(raw);
  } else {
    return false;
  }
  return wl::BindHandler<xdg_wm_base_traits>(
      registry_, xdg_wm_base_, xdg_wm_base_name_, xdg_wm_base_ver_);
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
                            *xdg_wm_base_.Get(), surface_.Get()->GetProxy()))) {
    return false;
  }
  xdg_surface_.Get()->app_ = this;
  if (!wl::SetupHandler(xdg_toplevel_,
                        wl::construct<xdg_toplevel_traits,
                                      xdg_surface_traits::Op::GetToplevel>(
                            *xdg_surface_.Get()))) {
    return false;
  }
  xdg_toplevel_.Get()->app_ = this;
  xdg_toplevel_.Get()->SetTitle("water");
  xdg_toplevel_.Get()->SetAppId("org.water-ffi.water-window");

  surface_.Get()->Commit();  // trigger the initial configure
  return wl::RoundtripWithTimeout(display_.Get(), kRoundtripTimeoutMs);
}

int App::Run() {
  if (!display_.Connect()) {
    std::fprintf(stderr, "water_window: wl_display_connect failed\n");
    return EXIT_FAILURE;
  }
  if (!BindGlobals() || !CreateWindow())
    return EXIT_FAILURE;

  std::printf(
      "water_window: XDG window created (%dx%d). Dma-buf present + the "
      "water engine are wired on top of this next.\n",
      width_, height_);

  const bool ok = wl::RunEventLoop(
      display_.Get(), [this] { return !running_; }, "water_window");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main() {
  App app;
  return app.Run();
}
