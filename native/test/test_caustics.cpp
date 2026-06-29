// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// Caustics test: the caustics pass (differential-area, light-POV
// projected water mesh) feeding the full scene, plus the VALIDATION GATE — the
// sphere blob-shadow must TRACK the ball, not mirror it. We render the caustic
// for the ball at opposite x (and z) offsets, read back the shadow channel (G),
// and assert the shadow centroid moves in the SAME direction as the ball. A
// mirror (wrong Y reconciliation) would move it opposite. Then renders the full
// composited scene. Exit 77 = SKIP, 1 = fail.

#include "water/camera.hpp"
#include "water/caustics_pass.hpp"
#include "water/device.hpp"
#include "water/heightfield_sim.hpp"
#include "water/mesh.hpp"
#include "water/scene_pass.hpp"
#include "water/skybox.hpp"
#include "water/texture.hpp"

#include "scene_shaders.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <glm/glm.hpp>
#include <set>
#include <span>
#include <vector>

using namespace water;

namespace {
constexpr uint32_t kW = 512, kH = 512;
#define ASSET(name) (WATER_ASSET_DIR "/" name)

// Copy an RGBA8 image (in SHADER_READ_ONLY) to host; leaves it
// SHADER_READ_ONLY.
Result<std::vector<uint8_t>> read_rgba8(const Device& dev, const Image& img) {
  const VkDeviceSize bytes = VkDeviceSize(img.width) * img.height * 4;
  Buffer host = WATER_TRY(
      dev.create_host_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT));
  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
    auto bar = [&](VkImageLayout f, VkImageLayout t, VkAccessFlags sa,
                   VkAccessFlags da, VkPipelineStageFlags ss,
                   VkPipelineStageFlags ds) {
      const VkImageMemoryBarrier b{
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .srcAccessMask = sa,
          .dstAccessMask = da,
          .oldLayout = f,
          .newLayout = t,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = img.image,
          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      vkCmdPipelineBarrier(cmd, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
    };
    bar(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkBufferImageCopy region{
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {img.width, img.height, 1}};
    vkCmdCopyImageToBuffer(cmd, img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           host.buffer, 1, &region);
    bar(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
        VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
  }));
  std::vector<uint8_t> out(bytes);
  std::memcpy(out.data(), host.mapped, bytes);
  dev.destroy_buffer(host);
  return out;
}

// Where did the sphere blob-shadow MOVE between two caustic frames?
// Differencing the G channel cancels the static lit/cleared background and the
// static caustic intensity, so only the moving shadow survives. Returns
// {centroid where B got darker (new shadow), centroid where A was darker (old
// shadow)}.
std::pair<glm::vec2, glm::vec2> shadow_shift(const std::vector<uint8_t>& A,
                                             const std::vector<uint8_t>& B,
                                             uint32_t n) {
  double bx = 0, by = 0, bw = 0, ax = 0, ay = 0, aw = 0;
  for (uint32_t y = 0; y < n; ++y)
    for (uint32_t x = 0; x < n; ++x) {
      const std::size_t i = (std::size_t(y) * n + x) * 4 + 1;  // G
      const int d = int(B[i]) - int(A[i]);
      if (d < 0) {
        bx += -d * double(x);
        by += -d * double(y);
        bw += -d;
      }  // B darker = new
      else if (d > 0) {
        ax += d * double(x);
        ay += d * double(y);
        aw += d;
      }  // A darker = old
    }
  return {bw > 0 ? glm::vec2(bx / bw, by / bw) : glm::vec2(-1),
          aw > 0 ? glm::vec2(ax / aw, ay / aw) : glm::vec2(-1)};
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
  Texture sky = WATER_TRY(load_cubemap(
      dev, {ASSET("xpos.jpg"), ASSET("xneg.jpg"), ASSET("ypos.jpg"),
            ASSET("ypos.jpg"), ASSET("zpos.jpg"), ASSET("zneg.jpg")}));

  HeightfieldSim sim = WATER_TRY(HeightfieldSim::create(dev, 256));
  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
    sim.clear(cmd);
    const glm::vec2 drops[] = {
        {0, 0}, {0.4f, 0.3f}, {-0.4f, -0.2f}, {0.3f, -0.4f}, {-0.3f, 0.4f}};
    for (auto d : drops)
      sim.add_drop(cmd, d, 0.05f, 0.5f);
    for (int i = 0; i < 30; ++i)
      sim.step(cmd);
  }));

  CausticsPass caustics = WATER_TRY(CausticsPass::create(dev, 1024, 2));
  Mesh plane = WATER_TRY(upload_mesh(dev, make_plane(200)));
  VkDescriptorSet water_set =
      WATER_TRY(caustics.make_water_set(sim.current().view, clamp));

  const float light[3] = {2, 2, -1};
  auto base_push = [&](glm::vec4 sphere) {
    ScenePush p{};
    p.light = glm::vec4(glm::normalize(glm::vec3(light[0], light[1], light[2])),
                        0.0f);
    p.sphere = sphere;
    return p;
  };
  auto render_caustic = [&](glm::vec4 sphere) -> Result<std::vector<uint8_t>> {
    WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
      caustics.render(cmd, 0, base_push(sphere), plane, water_set);
    }));
    return read_rgba8(dev, caustics.image(0));
  };

  int fails = 0;
  auto check = [&](bool ok, const char* m) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", m);
    fails += !ok;
  };

  // ---- tracking gate: the moving shadow must follow the ball (not mirror it)
  // ----
  const float r = 0.25f, y = -0.75f;
  const uint32_t cn = caustics.size();
  auto [newx, oldx] =
      shadow_shift(WATER_TRY(render_caustic({-0.5f, y, 0.0f, r})),
                   WATER_TRY(render_caustic({0.5f, y, 0.0f, r})), cn);
  auto [newz, oldz] =
      shadow_shift(WATER_TRY(render_caustic({0.0f, y, -0.5f, r})),
                   WATER_TRY(render_caustic({0.0f, y, 0.5f, r})), cn);
  std::printf(
      "  ball.x -0.5->+0.5: old-shadow u=%.0f  new-shadow u=%.0f  (du=%.0f)\n",
      oldx.x, newx.x, newx.x - oldx.x);
  std::printf(
      "  ball.z -0.5->+0.5: old-shadow v=%.0f  new-shadow v=%.0f  (dv=%.0f)\n",
      oldz.y, newz.y, newz.y - oldz.y);
  // Ball moves +x -> the new shadow must be at higher u than the old (and
  // likewise z->v).
  check(newx.x - oldx.x > 80.0f, "shadow tracks ball in X (not mirrored)");
  check(newz.y - oldz.y > 80.0f, "shadow tracks ball in Z (not mirrored)");

  // ---- full scene with real caustics ----
  const glm::vec4 ball{-0.4f, -0.75f, 0.2f, 0.25f};
  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
    caustics.render(cmd, 0, base_push(ball), plane, water_set);
  }));

  ScenePass scene = WATER_TRY(ScenePass::create(dev, kW, kH));
  Skybox skybox =
      WATER_TRY(Skybox::create(dev, scene.render_pass(), sky, clamp));
  Mesh pool = WATER_TRY(upload_mesh(dev, make_pool()));
  Mesh sphere = WATER_TRY(upload_mesh(dev, make_sphere(32)));
  VkDescriptorSet surf =
      WATER_TRY(scene.make_surface_set({{{sim.current().view, clamp},
                                         {tiles.view, repeat},
                                         {caustics.image(0).view, clamp},
                                         {sky.view, clamp}}}));

  auto P = [&](std::span<const uint32_t> v, std::span<const uint32_t> f,
               VkCullModeFlags c) {
    return scene.make_pipeline({.vert = v, .frag = f, .cull = c});
  };
  VkPipeline pool_pipe = WATER_TRY(
      P(std::span(pool_vert_spv, pool_vert_spv_count),
        std::span(pool_frag_spv, pool_frag_spv_count), VK_CULL_MODE_BACK_BIT));
  VkPipeline sphere_pipe = WATER_TRY(
      P(std::span(sphere_vert_spv, sphere_vert_spv_count),
        std::span(sphere_frag_spv, sphere_frag_spv_count), VK_CULL_MODE_NONE));
  VkPipeline above_pipe =
      WATER_TRY(P(std::span(water_vert_spv, water_vert_spv_count),
                  std::span(water_above_frag_spv, water_above_frag_spv_count),
                  VK_CULL_MODE_FRONT_BIT));
  VkPipeline below_pipe =
      WATER_TRY(P(std::span(water_vert_spv, water_vert_spv_count),
                  std::span(water_below_frag_spv, water_below_frag_spv_count),
                  VK_CULL_MODE_BACK_BIT));

  OrbitCamera cam;
  const glm::mat4 vp = cam.view_projection(scene.aspect());
  ScenePush push = base_push(ball);
  push.vp = vp;
  push.eye = glm::vec4(cam.position(), 1.0f);

  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
    scene.begin(cmd, {0, 0, 0, 1});
    skybox.record(cmd, glm::inverse(vp), cam.position());
    scene.draw(cmd, pool_pipe, push, pool, surf);
    scene.draw(cmd, above_pipe, push, plane, surf);
    scene.draw(cmd, below_pipe, push, plane, surf);
    scene.draw(cmd, sphere_pipe, push, sphere, surf);
    scene.end(cmd);
  }));
  std::vector<uint8_t> img = WATER_TRY(scene.readback_color(dev));
  std::set<uint32_t> colors;
  for (std::size_t i = 0; i < std::size_t(kW) * kH; ++i)
    colors.insert(img[i * 4] | img[i * 4 + 1] << 8 | img[i * 4 + 2] << 16);
  std::printf("  full scene distinct-colors=%zu\n", colors.size());
  check(colors.size() > 10000, "full caustic-lit scene renders");
  dump_ppm(img, kW, kH,
           std::getenv("WATER_DUMP") ? std::getenv("WATER_DUMP")
                                     : "/tmp/water_caustics.ppm");

  destroy_mesh(dev, pool);
  destroy_mesh(dev, sphere);
  destroy_mesh(dev, plane);
  destroy_texture(dev, tiles);
  destroy_texture(dev, sky);
  vkDestroySampler(dev.handle(), clamp, nullptr);
  vkDestroySampler(dev.handle(), repeat, nullptr);
  std::printf("\n  VERDICT: %s\n",
              fails == 0
                  ? "PASS — caustics render; sphere shadow TRACKS the ball"
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
