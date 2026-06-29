// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// Asset + skybox test: load the pool tiles (2D, mipmapped) + sky cubemap (6
// faces), verify the cubemap face order by reading layers back and comparing to
// the CPU-decoded source (— wrong face order silently corrupts 's reflections),
// then render the skybox background under the pool+sphere. Dumps a PPM. Exit 77
// = SKIP, 1 = fail.

#include "water/camera.hpp"
#include "water/device.hpp"
#include "water/mesh.hpp"
#include "water/scene_pass.hpp"
#include "water/skybox.hpp"
#include "water/texture.hpp"

#include "scene_shaders.h"  // pool_vert_spv, sphere_vert_spv, debug_scene_frag_spv
#include "stb/stb_image.h"  // declarations only; stbi_load is defined in water_core

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <glm/gtc/matrix_inverse.hpp>
#include <span>
#include <vector>

using namespace water;

namespace {
constexpr uint32_t kW = 512, kH = 512;
#define ASSET(name) (WATER_ASSET_DIR "/" name)

glm::vec3 mean_rgb(const std::vector<uint8_t>& rgba) {
  glm::dvec3 s{};
  const std::size_t n = rgba.size() / 4;
  for (std::size_t i = 0; i < n; ++i)
    s += glm::dvec3(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]);
  return glm::vec3(s / double(n));
}

glm::vec3 cpu_face_mean(const char* path) {
  int w = 0, h = 0, c = 0;
  stbi_uc* p = stbi_load(path, &w, &h, &c, 4);
  if (!p)
    return glm::vec3(-1);
  std::vector<uint8_t> v(p, p + std::size_t(w) * h * 4);
  stbi_image_free(p);
  return mean_rgb(v);
}

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
  std::printf("  tiles: %ux%u mips=%u\n", tiles.width, tiles.height,
              tiles.mip_levels);
  // Vulkan cube layer order [+X,-X,+Y,-Y,+Z,-Z]; +Y reused for -Y (upstream).
  Texture sky = WATER_TRY(load_cubemap(
      dev, {ASSET("xpos.jpg"), ASSET("xneg.jpg"), ASSET("ypos.jpg"),
            ASSET("ypos.jpg"), ASSET("zpos.jpg"), ASSET("zneg.jpg")}));
  std::printf("  sky cubemap: %ux%u\n", sky.width, sky.height);

  int fails = 0;
  auto check = [&](bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", msg);
    fails += !ok;
  };
  check(tiles.mip_levels > 1, "tiles mipmapped");

  // Face-order verification: layer N's GPU content matches the source face N
  // decoded on CPU.
  const std::array<const char*, 6> face_path{
      ASSET("xpos.jpg"), ASSET("xneg.jpg"), ASSET("ypos.jpg"),
      ASSET("ypos.jpg"), ASSET("zpos.jpg"), ASSET("zneg.jpg")};
  const std::array<const char*, 6> face_name{"+X", "-X", "+Y",
                                             "-Y", "+Z", "-Z"};
  for (uint32_t i = 0; i < 6; ++i) {
    const glm::vec3 gpu = mean_rgb(WATER_TRY(readback_layer(dev, sky, i)));
    const glm::vec3 cpu = cpu_face_mean(face_path[i]);
    const float d = glm::length(gpu - cpu);
    std::printf(
        "  layer %u (%s): gpu(%.0f,%.0f,%.0f) cpu(%.0f,%.0f,%.0f) dist=%.1f\n",
        i, face_name[i], gpu.r, gpu.g, gpu.b, cpu.r, cpu.g, cpu.b, d);
    check(d < 2.0f, "cube layer matches source face");
  }

  // Render skybox background + pool + sphere.
  ScenePass scene = WATER_TRY(ScenePass::create(dev, kW, kH));
  Skybox skybox =
      WATER_TRY(Skybox::create(dev, scene.render_pass(), sky, clamp));
  Mesh pool = WATER_TRY(upload_mesh(dev, make_pool()));
  Mesh sphere = WATER_TRY(upload_mesh(dev, make_sphere(32)));
  VkPipeline pool_pipe = WATER_TRY(scene.make_pipeline(
      {.vert = std::span(pool_vert_spv, pool_vert_spv_count),
       .frag = std::span(debug_scene_frag_spv, debug_scene_frag_spv_count),
       .cull = VK_CULL_MODE_BACK_BIT}));
  VkPipeline sphere_pipe = WATER_TRY(scene.make_pipeline(
      {.vert = std::span(sphere_vert_spv, sphere_vert_spv_count),
       .frag = std::span(debug_scene_frag_spv, debug_scene_frag_spv_count),
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
    skybox.record(cmd, glm::inverse(vp), cam.position());  // background first
    scene.draw(cmd, pool_pipe, push, pool);
    scene.draw(cmd, sphere_pipe, push, sphere);
    scene.end(cmd);
  }));

  std::vector<uint8_t> img = WATER_TRY(scene.readback_color(dev));
  std::size_t lit = 0;
  for (std::size_t i = 0; i < std::size_t(kW) * kH; ++i)
    if (img[i * 4] + img[i * 4 + 1] + img[i * 4 + 2] > 12)
      ++lit;
  const double coverage = double(lit) / (double(kW) * kH);
  std::printf("  coverage=%.1f%% (skybox fills background)\n",
              coverage * 100.0);
  check(coverage > 0.95, "skybox + geometry fill the frame");

  dump_ppm(img, kW, kH,
           std::getenv("WATER_DUMP") ? std::getenv("WATER_DUMP")
                                     : "/tmp/water_sky.ppm");

  destroy_mesh(dev, pool);
  destroy_mesh(dev, sphere);
  destroy_texture(dev, tiles);
  destroy_texture(dev, sky);
  vkDestroySampler(dev.handle(), clamp, nullptr);
  vkDestroySampler(dev.handle(), repeat, nullptr);
  std::printf("\n  VERDICT: %s\n",
              fails == 0
                  ? "PASS — assets load, cube order verified, sky renders"
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
