// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// Scene infrastructure test: render the pool + sphere through the
// world-space vertex pipeline with depth, the orbit camera/VP, and the Vulkan
// Y-flip. Validates the hardest scene-pass risk (Y-orientation) with visible
// geometry before the ray-traced shaders land in . Asserts non-trivial coverage
// + shading; dumps a PPM for visual confirmation (orientation, occlusion). Exit
// 77 = SKIP, 1 = fail.

#include "water/camera.hpp"
#include "water/device.hpp"
#include "water/mesh.hpp"
#include "water/scene_pass.hpp"

#include "scene_shaders.h"  // pool_vert_spv, sphere_vert_spv, debug_scene_frag_spv

#include <cstdio>
#include <cstdlib>
#include <set>
#include <span>
#include <vector>

using namespace water;

namespace {
constexpr uint32_t kW = 512, kH = 512;

void dump_ppm(const std::vector<uint8_t>& rgba,
              uint32_t w,
              uint32_t h,
              const char* path) {
  if (FILE* fp = std::fopen(path, "wb")) {
    std::fprintf(fp, "P6\n%u %u\n255\n", w, h);
    for (std::size_t i = 0; i < std::size_t(w) * h; ++i)
      std::fwrite(&rgba[i * 4], 1, 3, fp);
    std::fclose(fp);
    std::printf("  dumped scene -> %s\n", path);
  }
}

Result<int> run() {
  DeviceConfig cfg{};
  if (const char* sub = std::getenv("WATER_VK_DEVICE"))
    cfg.device_substr = sub;
  Device dev = WATER_TRY(Device::create(cfg));
  std::printf("device=\"%s\"  %ux%u\n", dev.caps().device_name.c_str(), kW, kH);

  ScenePass scene = WATER_TRY(ScenePass::create(dev, kW, kH));
  Mesh pool = WATER_TRY(upload_mesh(dev, make_pool()));
  Mesh sphere = WATER_TRY(upload_mesh(dev, make_sphere(32)));

  // Real cull modes: pool BACK (so the near walls drop out and the camera sees
  // into the pool), sphere NONE. This also exercises the frontFace=CW
  // compensation for the negative-height viewport.
  VkPipeline pool_pipe = WATER_TRY(scene.make_pipeline({
      .vert = std::span(pool_vert_spv, pool_vert_spv_count),
      .frag = std::span(debug_scene_frag_spv, debug_scene_frag_spv_count),
      .cull = VK_CULL_MODE_BACK_BIT,
  }));
  VkPipeline sphere_pipe = WATER_TRY(scene.make_pipeline({
      .vert = std::span(sphere_vert_spv, sphere_vert_spv_count),
      .frag = std::span(debug_scene_frag_spv, debug_scene_frag_spv_count),
      .cull = VK_CULL_MODE_NONE,
  }));

  OrbitCamera cam;  // main.js defaults: target (0,-0.5,0), dist 4, -25/-200.5
                    // deg, fov 45
  ScenePush push{};
  push.vp = cam.view_projection(scene.aspect());
  push.eye = glm::vec4(cam.position(), 1.0f);
  push.light = glm::vec4(glm::normalize(glm::vec3(2.0f, 2.0f, -1.0f)), 0.0f);
  push.sphere = glm::vec4(-0.4f, -0.75f, 0.2f, 0.25f);  // center, radius

  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
    scene.begin(cmd, {0, 0, 0, 1});
    scene.draw(cmd, pool_pipe, push, pool);
    scene.draw(cmd, sphere_pipe, push, sphere);
    scene.end(cmd);
  }));

  std::vector<uint8_t> img = WATER_TRY(scene.readback_color(dev));

  // Coverage + shading-variety checks (chaos-free; pixel golden vs web is ).
  std::size_t lit = 0;
  std::set<int> lums;
  for (std::size_t i = 0; i < std::size_t(kW) * kH; ++i) {
    const int r = img[i * 4], g = img[i * 4 + 1], b = img[i * 4 + 2];
    if (r + g + b > 12)
      ++lit;  // above the black background
    lums.insert((r * 30 + g * 59 + b * 11) / 100);
  }
  const double coverage = double(lit) / (double(kW) * kH);

  int fails = 0;
  auto check = [&](bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", msg);
    fails += !ok;
  };
  std::printf("  coverage=%.1f%%  distinct-luma=%zu\n", coverage * 100.0,
              lums.size());
  check(coverage > 0.05 && coverage < 0.95,
        "geometry present (5%% < coverage < 95%%)");
  check(lums.size() > 16,
        "non-trivial shading gradient (depth/normal variation)");

  const char* path = std::getenv("WATER_DUMP");
  dump_ppm(img, kW, kH, path ? path : "/tmp/water_scene.ppm");

  destroy_mesh(dev, pool);
  destroy_mesh(dev, sphere);
  std::printf(
      "\n  VERDICT: %s\n",
      fails == 0 ? "PASS — scene pass renders pool+sphere with depth" : "FAIL");
  return fails == 0 ? 0 : 1;
}

}  // namespace

int main() {
  Result<int> r = run();
  if (!r) {
    std::fprintf(stderr, "SKIP/ERROR: %.*s (%s)\n", int(r.error().where.size()),
                 r.error().where.data(), to_string(r.error().code));
    return r.error().code == VK_ERROR_INITIALIZATION_FAILED ? 77 : 1;
  }
  return *r;
}
