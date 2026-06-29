// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// Sim invariants test: the heightfield sim is chaotic, so we assert
// structural invariants rather than pixels:
//   * a drop raises the mean height (S1 injects),
//   * energy decays over time (the *0.995 damping in S3),
//   * stored normals are unit-bounded: B^2 + A^2 <= 1 (so normal.y is real),
//   * no non-finite texels over a long soak (the NaN trap holds).
// Deterministic enough for CI on lavapipe. Exit 77 = SKIP (no device), 1 =
// fail.

#include "water/device.hpp"
#include "water/heightfield_sim.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace water;

namespace {
constexpr uint32_t kSize = 256;

struct Stats {
  double mean_height = 0;
  double energy = 0;         // sum(height^2 + velocity^2)
  double max_normal_sq = 0;  // max(B^2 + A^2)
  bool all_finite = true;
};

Stats analyze(const std::vector<glm::vec4>& f) {
  Stats s;
  for (const auto& t : f) {
    if (!std::isfinite(t.x) || !std::isfinite(t.y) || !std::isfinite(t.z) ||
        !std::isfinite(t.w))
      s.all_finite = false;
    s.mean_height += t.x;
    s.energy += double(t.x) * t.x + double(t.y) * t.y;
    s.max_normal_sq =
        std::max(s.max_normal_sq, double(t.z) * t.z + double(t.w) * t.w);
  }
  s.mean_height /= double(f.size());
  return s;
}

void dump_pgm(const std::vector<glm::vec4>& f, uint32_t n, const char* path) {
  float lo = 1e30f, hi = -1e30f;
  for (const auto& t : f) {
    lo = std::min(lo, t.x);
    hi = std::max(hi, t.x);
  }
  const float range = std::max(hi - lo, 1e-6f);
  std::vector<uint8_t> px(std::size_t(n) * n);
  for (std::size_t i = 0; i < px.size(); ++i)
    px[i] = uint8_t(std::clamp((f[i].x - lo) / range, 0.0f, 1.0f) * 255.0f);
  if (FILE* fp = std::fopen(path, "wb")) {
    std::fprintf(fp, "P5\n%u %u\n255\n", n, n);
    std::fwrite(px.data(), 1, px.size(), fp);
    std::fclose(fp);
    std::printf("  dumped heightmap -> %s\n", path);
  }
}

Result<int> run() {
  DeviceConfig cfg{};
  if (const char* sub = std::getenv("WATER_VK_DEVICE"))
    cfg.device_substr = sub;
  cfg.force_half =
      std::getenv("WATER_FORCE_HALF") != nullptr;  // exercise the 16F path
  Device dev = WATER_TRY(Device::create(cfg));
  HeightfieldSim sim = WATER_TRY(HeightfieldSim::create(dev, kSize));
  std::printf(
      "  device=\"%s\"  grid=%u  heightfield=%s\n",
      dev.caps().device_name.c_str(), kSize,
      sim.format() == VK_FORMAT_R32G32B32A32_SFLOAT ? "RGBA32F" : "RGBA16F");

  int fails = 0;
  auto check = [&](bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", msg);
    fails += !ok;
  };

  // --- flat after clear ---
  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) { sim.clear(cmd); }));
  Stats flat = analyze(WATER_TRY(sim.readback(dev)));
  check(flat.all_finite && std::abs(flat.mean_height) < 1e-7,
        "clear -> flat, finite, mean~0");

  // --- a drop raises mean height ---
  WATER_TRYV(dev.submit_now(
      [&](VkCommandBuffer cmd) { sim.add_drop(cmd, {0, 0}, 0.05f, 0.5f); }));
  Stats dropped = analyze(WATER_TRY(sim.readback(dev)));
  check(dropped.all_finite && dropped.mean_height > 1e-5,
        "drop raises mean height");

  // --- energy decays under damping ---
  auto run_frames = [&](int frames) -> Result<void> {
    constexpr int kPerSubmit = 500;  // 500 frames * 3 passes per command buffer
    for (int done = 0; done < frames;) {
      const int n = std::min(kPerSubmit, frames - done);
      WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
        for (int i = 0; i < n; ++i)
          sim.step(cmd);
      }));
      done += n;
    }
    return {};
  };
  WATER_TRYV(run_frames(50));
  Stats e_early = analyze(WATER_TRY(sim.readback(dev)));
  WATER_TRYV(run_frames(350));
  Stats e_late = analyze(WATER_TRY(sim.readback(dev)));
  std::printf("  energy: frame50=%.4g  frame400=%.4g\n", e_early.energy,
              e_late.energy);
  check(e_late.energy < e_early.energy, "energy decays (frame400 < frame50)");
  check(e_late.max_normal_sq <= 1.0 + 1e-3,
        "stored normals unit-bounded (B^2+A^2 <= 1)");
  check(e_late.all_finite, "field finite after 400 frames");

  // --- sphere pass smoke (S2 stays finite) ---
  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
    sim.move_sphere(cmd, {0, 0, 0}, {0.2f, 0, 0.1f}, 0.25f);
    sim.step(cmd);
  }));
  check(analyze(WATER_TRY(sim.readback(dev))).all_finite,
        "sphere displacement stays finite");

  // --- NaN soak (~1e5 steps) ---
  const char* soak_env = std::getenv("WATER_SOAK_STEPS");
  const int soak = soak_env ? int(std::strtol(soak_env, nullptr, 10)) : 100000;
  for (int done = 0; done < soak;) {
    const int n = std::min(5000, soak - done);
    WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
      for (int i = 0; i < n; ++i)
        sim.update(cmd);
    }));
    done += n;
  }
  Stats soaked = analyze(WATER_TRY(sim.readback(dev)));
  check(soaked.all_finite, "no NaN/Inf after soak");
  std::printf("  soak=%d steps  mean=%.3g  energy=%.3g\n", soak,
              soaked.mean_height, soaked.energy);

  if (const char* p = std::getenv("WATER_DUMP"))
    dump_pgm(WATER_TRY(sim.readback(dev)), kSize, p);

  std::printf("\n  VERDICT: %s\n",
              fails == 0 ? "PASS — sim invariants hold" : "FAIL");
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
