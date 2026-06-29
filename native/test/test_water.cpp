// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// Water surface test: above (reflect/refract Fresnel mix off the sky cubemap +
// scene) and below — over the live rippled heightfield, composited with the
// pool
// + sphere. Caustics are still the neutral placeholder (the real caustics pass
// arrives separately). Asserts the full frame renders with rich reflection/
// refraction structure; dumps a PPM for visual confirmation of sky reflection +
// refracted floor. Exit 77 = SKIP, 1 = fail.

#include "water/camera.hpp"
#include "water/device.hpp"
#include "water/heightfield_sim.hpp"
#include "water/mesh.hpp"
#include "water/scene_pass.hpp"
#include "water/skybox.hpp"
#include "water/texture.hpp"

#include "scene_shaders.h"

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
  Texture caustic = WATER_TRY(make_solid_texture(dev, 255, 255, 0, 255));

  // Deterministic ripples: a fixed pattern of drops, then settle a few dozen
  // frames.
  HeightfieldSim sim = WATER_TRY(HeightfieldSim::create(dev, 256));
  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
    sim.clear(cmd);
    const glm::vec2 drops[] = {{0, 0},        {0.4f, 0.3f},  {-0.4f, -0.2f},
                               {0.3f, -0.4f}, {-0.3f, 0.4f}, {0.1f, -0.1f},
                               {-0.5f, 0.1f}, {0.5f, -0.3f}};
    for (auto d : drops)
      sim.add_drop(cmd, d, 0.05f, 0.5f);
    for (int i = 0; i < 40; ++i)
      sim.step(cmd);
  }));

  ScenePass scene = WATER_TRY(ScenePass::create(dev, kW, kH));
  Skybox skybox =
      WATER_TRY(Skybox::create(dev, scene.render_pass(), sky, clamp));
  Mesh pool = WATER_TRY(upload_mesh(dev, make_pool()));
  Mesh sphere = WATER_TRY(upload_mesh(dev, make_sphere(32)));
  Mesh plane = WATER_TRY(upload_mesh(dev, make_plane(100)));

  VkDescriptorSet surf =
      WATER_TRY(scene.make_surface_set({{{sim.current().view, clamp},
                                         {tiles.view, repeat},
                                         {caustic.view, clamp},
                                         {sky.view, clamp}}}));

  auto pipe = [&](std::span<const uint32_t> v, std::span<const uint32_t> f,
                  VkCullModeFlags cull) {
    return scene.make_pipeline({.vert = v, .frag = f, .cull = cull});
  };
  VkPipeline pool_pipe = WATER_TRY(pipe(
      std::span(pool_vert_spv, pool_vert_spv_count),
      std::span(pool_frag_spv, pool_frag_spv_count), VK_CULL_MODE_BACK_BIT));
  VkPipeline sphere_pipe = WATER_TRY(pipe(
      std::span(sphere_vert_spv, sphere_vert_spv_count),
      std::span(sphere_frag_spv, sphere_frag_spv_count), VK_CULL_MODE_NONE));
  VkPipeline above_pipe = WATER_TRY(
      pipe(std::span(water_vert_spv, water_vert_spv_count),
           std::span(water_above_frag_spv, water_above_frag_spv_count),
           VK_CULL_MODE_FRONT_BIT));
  VkPipeline below_pipe = WATER_TRY(
      pipe(std::span(water_vert_spv, water_vert_spv_count),
           std::span(water_below_frag_spv, water_below_frag_spv_count),
           VK_CULL_MODE_BACK_BIT));

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
    scene.draw(cmd, pool_pipe, push, pool,
               surf);  // order: pool, water above/below, sphere
    scene.draw(cmd, above_pipe, push, plane, surf);
    scene.draw(cmd, below_pipe, push, plane, surf);
    scene.draw(cmd, sphere_pipe, push, sphere, surf);
    scene.end(cmd);
  }));

  std::vector<uint8_t> img = WATER_TRY(scene.readback_color(dev));
  std::set<uint32_t> colors;
  std::size_t lit = 0;
  for (std::size_t i = 0; i < std::size_t(kW) * kH; ++i) {
    colors.insert(img[i * 4] | img[i * 4 + 1] << 8 | img[i * 4 + 2] << 16);
    if (img[i * 4] + img[i * 4 + 1] + img[i * 4 + 2] > 12)
      ++lit;
  }
  const double coverage = double(lit) / (double(kW) * kH);
  std::printf("  coverage=%.1f%%  distinct-colors=%zu\n", coverage * 100.0,
              colors.size());

  int fails = 0;
  auto check = [&](bool ok, const char* m) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", m);
    fails += !ok;
  };
  check(coverage > 0.95, "full frame rendered");
  check(colors.size() > 10000, "rich reflect/refract structure");

  dump_ppm(img, kW, kH,
           std::getenv("WATER_DUMP") ? std::getenv("WATER_DUMP")
                                     : "/tmp/water_surface.ppm");

  destroy_mesh(dev, pool);
  destroy_mesh(dev, sphere);
  destroy_mesh(dev, plane);
  destroy_texture(dev, tiles);
  destroy_texture(dev, sky);
  destroy_texture(dev, caustic);
  vkDestroySampler(dev.handle(), clamp, nullptr);
  vkDestroySampler(dev.handle(), repeat, nullptr);
  std::printf(
      "\n  VERDICT: %s\n",
      fails == 0
          ? "PASS — water surface (reflect/refract) renders over the scene"
          : "FAIL");
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
