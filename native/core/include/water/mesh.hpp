// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// water/mesh.hpp — procedural geometry (meshes.js) and its GPU buffers.
// Pool (open-top [-1,1] cube, 5 faces), unit sphere, and the water/caustics
// plane grid. All authored in the local space the vertex shaders expect;
// positions are vec3, the only vertex attribute (SEMANTIC_POSITION / location
// 0).
#pragma once

#include "water/device.hpp"
#include "water/result.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace water {

struct MeshData {
  std::vector<glm::vec3> positions;
  std::vector<uint32_t> indices;
};

// The five interior faces of a [-1,1] cube (no +Y; the pool vertex shader
// remaps Y so that face becomes the open top). Winding matches GL.Mesh.cube
// minus the -y face.
[[nodiscard]] MeshData make_pool();
// Unit sphere (radius 1) at the origin; the sphere VS scales/offsets by
// radius/center.
[[nodiscard]] MeshData make_sphere(int segments = 32);
// Flat grid in XY spanning [-1,1], z=0, with `detail` subdivisions/axis.
[[nodiscard]] MeshData make_plane(int detail);

struct Mesh {
  Buffer vbuf;
  Buffer ibuf;
  uint32_t index_count = 0;
};

// Upload to host-visible vertex/index buffers (small static meshes — no
// staging).
[[nodiscard]] Result<Mesh> upload_mesh(const Device& dev, const MeshData& data);
void destroy_mesh(const Device& dev, Mesh& mesh) noexcept;

}  // namespace water
