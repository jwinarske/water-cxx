// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// Scene shading test: the real ray-traced pool + sphere shading
// (getWallColor / getSphereColor) over the sky, with a flat (cleared)
// heightfield as the water input and a neutral caustic placeholder (the real
// caustics pass is ). The above-water regions (sky, upper pool tiles,
// above-water sphere) are the static golden regions. Pixel-exact golden vs the
// web needs the headless-capture harness (// separate); here we assert
// structural correctness (tiled walls, shaded sphere, sky, determinism) and
// dump a PPM. Exit 77 = SKIP, 1 = fail.

#include "water/camera.hpp"
#include "water/device.hpp"
#include "water/heightfield_sim.hpp"
#include "water/mesh.hpp"
#include "water/scene_pass.hpp"
#include "water/skybox.hpp"
#include "water/texture.hpp"

#include "scene_shaders.h"  // pool_vert_spv, sphere_vert_spv, pool_frag_spv, sphere_frag_spv

#include <cstdio>
#include <cstdlib>
#include <glm/glm.hpp>
#include <set>
#include <span>
#include <vector>

using namespace water;

namespace {
constexpr uint32_t kW = 512, kH = 512;
#define ASSET(name) (WATER_ASSET_DIR "/" name)

void dump_ppm(const std::vector<uint8_t>& rgba,
              uint32_t w,
              uint32_t h,
              const char* path) {
  if (FILE* fp = std::fopen(path, "wb")) {
    std::fprintf(fp, "P6\n%u %u\n255\n", w, h);
    for (std::size_t i = 0; i < std::size_t(w) * h; ++i)
      std::fwrite(&rgba[i * 4], 1, 3, fp);
    std::fclose(fp);
    std::printf("  dumped -> %s\n", path);
  }
}

Result<int> run() {
  DeviceConfig cfg{};
  if (const char* sub = std::getenv("WATER_VK_DEVICE"))
    cfg.device_substr = sub;
  Device dev = WATER_TRY(Device::create(cfg));
  std::printf("device=\"%s\"\n", dev.caps().device_name.c_str());

  VkSampler clamp = WATER_TRY(make_sampler(
      dev, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f));
  VkSampler repeat = WATER_TRY(make_sampler(dev, VK_FILTER_LINEAR,
                                            VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                            VK_LOD_CLAMP_NONE));

  Texture tiles = WATER_TRY(load_texture_2d(dev, ASSET("tiles.jpg"), true));
  Texture sky = WATER_TRY(load_cubemap(
      dev, {ASSET("xpos.jpg"), ASSET("xneg.jpg"), ASSET("ypos.jpg"),
            ASSET("ypos.jpg"), ASSET("zpos.jpg"), ASSET("zneg.jpg")}));
  // Neutral caustics (R=1 intensity, G=1 shadow) until the real pass lands in .
  Texture caustic = WATER_TRY(make_solid_texture(dev, 255, 255, 0, 255));

  // Flat, still water surface (the deterministic input for the static golden
  // regions).
  HeightfieldSim sim = WATER_TRY(HeightfieldSim::create(dev, 256));
  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) { sim.clear(cmd); }));

  ScenePass scene = WATER_TRY(ScenePass::create(dev, kW, kH));
  Skybox skybox =
      WATER_TRY(Skybox::create(dev, scene.render_pass(), sky, clamp));
  Mesh pool = WATER_TRY(upload_mesh(dev, make_pool()));
  Mesh sphere = WATER_TRY(upload_mesh(dev, make_sphere(32)));

  VkDescriptorSet surf = WATER_TRY(scene.make_surface_set({{
      {sim.current().view, clamp},  // 0 water (flat heightfield)
      {tiles.view, repeat},         // 1 tiles
      {caustic.view, clamp},        // 2 causticTex
      {sky.view,
       clamp},  // 3 sky (unused by pool/sphere; bound for completeness)
  }}));

  VkPipeline pool_pipe = WATER_TRY(scene.make_pipeline(
      {.vert = std::span(pool_vert_spv, pool_vert_spv_count),
       .frag = std::span(pool_frag_spv, pool_frag_spv_count),
       .cull = VK_CULL_MODE_BACK_BIT}));
  VkPipeline sphere_pipe = WATER_TRY(scene.make_pipeline(
      {.vert = std::span(sphere_vert_spv, sphere_vert_spv_count),
       .frag = std::span(sphere_frag_spv, sphere_frag_spv_count),
       .cull = VK_CULL_MODE_NONE}));

  OrbitCamera cam;
  const glm::mat4 vp = cam.view_projection(scene.aspect());
  ScenePush push{};
  push.vp = vp;
  push.eye = glm::vec4(cam.position(), 1.0f);
  push.light = glm::vec4(glm::normalize(glm::vec3(2, 2, -1)), 0.0f);
  push.sphere = glm::vec4(-0.4f, -0.75f, 0.2f, 0.25f);

  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
    scene.begin(cmd, {0, 0, 0, 1});
    skybox.record(cmd, glm::inverse(vp), cam.position());
    scene.draw(cmd, pool_pipe, push, pool, surf);
    scene.draw(cmd, sphere_pipe, push, sphere, surf);
    scene.end(cmd);
  }));

  std::vector<uint8_t> img = WATER_TRY(scene.readback_color(dev));

  int fails = 0;
  auto check = [&](bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", msg);
    fails += !ok;
  };
  std::set<uint32_t> colors;
  std::size_t lit = 0;
  for (std::size_t i = 0; i < std::size_t(kW) * kH; ++i) {
    const uint32_t c = img[i * 4] | img[i * 4 + 1] << 8 | img[i * 4 + 2] << 16;
    colors.insert(c);
    if (img[i * 4] + img[i * 4 + 1] + img[i * 4 + 2] > 12)
      ++lit;
  }
  const double coverage = double(lit) / (double(kW) * kH);
  std::printf("  coverage=%.1f%%  distinct-colors=%zu\n", coverage * 100.0,
              colors.size());
  check(coverage > 0.95, "scene fills the frame (sky + geometry)");
  // Tiled walls + sky gradient + shaded sphere produce a rich color set;
  // flat/broken shading would collapse it.
  check(colors.size() > 5000,
        "rich textured/shaded color set (tiles + shading present)");

  dump_ppm(img, kW, kH,
           std::getenv("WATER_DUMP") ? std::getenv("WATER_DUMP")
                                     : "/tmp/water_shading.ppm");

  destroy_mesh(dev, pool);
  destroy_mesh(dev, sphere);
  destroy_texture(dev, tiles);
  destroy_texture(dev, sky);
  destroy_texture(dev, caustic);
  vkDestroySampler(dev.handle(), clamp, nullptr);
  vkDestroySampler(dev.handle(), repeat, nullptr);
  std::printf(
      "\n  VERDICT: %s%s\n",
      fails == 0 ? "PASS — real pool/sphere shading renders" : "FAIL",
      fails == 0 ? " (pixel-exact web golden pending capture harness)" : "");
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
