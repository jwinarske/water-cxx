// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// water/device.hpp — Vulkan engine-slice device:
// instance/physical/device/queue, capability query, and the heightfield-format
// gate. Owns the one queue the single-threaded renderer uses (single-threaded).
// Resource helpers (image/host-buffer/ one-time submit) live here because they
// need the device's memory properties.
#pragma once

#include "water/result.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace water {

// Coarse device class, vendor-derived. The full DeviceProfile
// (auto-detect + mandatory config override) lands in ;  only records the guess.
enum class DeviceClass { Desktop, Mobile };

struct Caps {
  std::string device_name;
  DeviceClass device_class = DeviceClass::Desktop;
  // RGBA32F renderable AND filterable. Gates the heightfield format choice.
  bool float32_filterable = false;
  // Selected heightfield format: R32G32B32A32_SFLOAT when filterable and not
  // forced to half, else R16G16B16A16_SFLOAT (the safe floor on
  // Mali/V3DV/Adreno).
  VkFormat heightfield_format = VK_FORMAT_R16G16B16A16_SFLOAT;
  float timestamp_period_ns = 0.0f;
  bool timestamps_valid = false;
};

struct DeviceConfig {
  bool force_half =
      false;  // exercise the 16F path even where 32F is filterable
  bool enable_validation = false;  // VK_LAYER_KHRONOS_validation
  std::string_view
      device_substr{};  // pick the physical device whose name contains this
};

struct ImageDesc {
  uint32_t width = 0;
  uint32_t height = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkImageUsageFlags usage = 0;
  VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
  VkImageAspectFlags aspect =
      VK_IMAGE_ASPECT_COLOR_BIT;  // DEPTH_BIT for depth targets
};

struct Image {
  VkImage image = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  uint32_t width = 0;
  uint32_t height = 0;
};

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  void* mapped = nullptr;  // persistent map for host-visible buffers
};

class Device {
 public:
  [[nodiscard]] static Result<Device> create(const DeviceConfig& cfg);
  ~Device();
  Device(Device&&) noexcept;
  Device& operator=(Device&&) noexcept;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  [[nodiscard]] VkDevice handle() const noexcept { return device_; }
  [[nodiscard]] VkPhysicalDevice physical() const noexcept { return phys_; }
  [[nodiscard]] VkQueue queue() const noexcept { return queue_; }
  [[nodiscard]] uint32_t queue_family() const noexcept { return qfam_; }
  [[nodiscard]] const Caps& caps() const noexcept { return caps_; }

  [[nodiscard]] Result<Image> create_image(const ImageDesc&) const;
  void destroy_image(Image&) const noexcept;

  // Host-visible, coherent, persistently mapped (for readback / staging).
  [[nodiscard]] Result<Buffer> create_host_buffer(
      VkDeviceSize size,
      VkBufferUsageFlags usage) const;
  void destroy_buffer(Buffer&) const noexcept;

  // Allocate a primary command buffer, run `record`, submit, and wait idle.
  //  uses this; the per-frame renderer will not CPU-wait (no-CPU-wait).
  [[nodiscard]] Result<void> submit_now(
      const std::function<void(VkCommandBuffer)>& record) const;

  [[nodiscard]] Result<uint32_t> find_memory_type(
      uint32_t type_bits,
      VkMemoryPropertyFlags want) const;

 private:
  Device() = default;
  void destroy() noexcept;

  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice phys_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t qfam_ = 0;
  VkPhysicalDeviceMemoryProperties mem_props_{};
  Caps caps_{};
};

}  // namespace water
