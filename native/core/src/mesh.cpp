// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#include "water/mesh.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <numbers>

namespace water {

MeshData make_plane(int detail) {
  MeshData m;
  for (int y = 0; y <= detail; ++y) {
    const float t = float(y) / float(detail);
    for (int x = 0; x <= detail; ++x) {
      const float s = float(x) / float(detail);
      m.positions.emplace_back(2 * s - 1, 2 * t - 1, 0);
      if (x < detail && y < detail) {
        const auto i = uint32_t(x + y * (detail + 1));
        m.indices.insert(m.indices.end(), {i, i + 1, i + uint32_t(detail) + 1});
        m.indices.insert(m.indices.end(), {i + uint32_t(detail) + 1, i + 1,
                                           i + uint32_t(detail) + 2});
      }
    }
  }
  return m;
}

MeshData make_sphere(int segments) {
  MeshData m;
  const float pi = std::numbers::pi_v<float>;
  for (int lat = 0; lat <= segments; ++lat) {
    const float theta = float(lat) * pi / float(segments);
    const float sin_t = std::sin(theta), cos_t = std::cos(theta);
    for (int lon = 0; lon <= segments; ++lon) {
      const float phi = float(lon) * 2.0f * pi / float(segments);
      m.positions.emplace_back(std::cos(phi) * sin_t, cos_t,
                               std::sin(phi) * sin_t);
    }
  }
  for (int lat = 0; lat < segments; ++lat) {
    for (int lon = 0; lon < segments; ++lon) {
      const auto a = uint32_t(lat * (segments + 1) + lon);
      const uint32_t b = a + uint32_t(segments) + 1;
      m.indices.insert(m.indices.end(), {a, b, a + 1});
      m.indices.insert(m.indices.end(), {a + 1, b, b + 1});
    }
  }
  return m;
}

MeshData make_pool() {
  // -x, +x, +y (-> floor after Y remap), -z, +z; 4 corners each (meshes.js
  // CUBE_FACES).
  static const std::array<std::array<glm::vec3, 4>, 5> faces{{
      {{{-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1}}},  // -x
      {{{1, -1, -1}, {1, 1, -1}, {1, -1, 1}, {1, 1, 1}}},      // +x
      {{{-1, 1, -1}, {-1, 1, 1}, {1, 1, -1}, {1, 1, 1}}},      // +y
      {{{-1, -1, -1}, {-1, 1, -1}, {1, -1, -1}, {1, 1, -1}}},  // -z
      {{{-1, -1, 1}, {1, -1, 1}, {-1, 1, 1}, {1, 1, 1}}},      // +z
  }};
  MeshData m;
  for (uint32_t f = 0; f < faces.size(); ++f) {
    const uint32_t v = f * 4;
    for (const auto& corner : faces[f])
      m.positions.push_back(corner);
    m.indices.insert(m.indices.end(), {v, v + 1, v + 2});
    m.indices.insert(m.indices.end(), {v + 2, v + 1, v + 3});
  }
  return m;
}

Result<Mesh> upload_mesh(const Device& dev, const MeshData& data) {
  Mesh mesh;
  mesh.index_count = uint32_t(data.indices.size());

  const VkDeviceSize vbytes = data.positions.size() * sizeof(glm::vec3);
  mesh.vbuf = WATER_TRY(
      dev.create_host_buffer(vbytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));
  std::memcpy(mesh.vbuf.mapped, data.positions.data(), vbytes);

  const VkDeviceSize ibytes = data.indices.size() * sizeof(uint32_t);
  mesh.ibuf = WATER_TRY(
      dev.create_host_buffer(ibytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT));
  std::memcpy(mesh.ibuf.mapped, data.indices.data(), ibytes);
  return mesh;
}

void destroy_mesh(const Device& dev, Mesh& mesh) noexcept {
  dev.destroy_buffer(mesh.vbuf);
  dev.destroy_buffer(mesh.ibuf);
  mesh.index_count = 0;
}

}  // namespace water
