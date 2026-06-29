// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// vkboot.hpp — minimal explicit Vulkan bootstrap shared by the throwaway
// spikes. Header-only, C API, no exceptions (matches the std::expected error
// discipline the real engine-slice will use). Not production code — just enough
// to stand up a device, allocate images/buffers, run command buffers, and read
// GPU timestamps.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace vkboot {

#define VK_CHECK(expr)                                                    \
  do {                                                                    \
    VkResult vk_check_res_ = (expr);                                      \
    if (vk_check_res_ != VK_SUCCESS) {                                    \
      std::fprintf(stderr, "VK_CHECK failed: %s == %d at %s:%d\n", #expr, \
                   static_cast<int>(vk_check_res_), __FILE__, __LINE__);  \
      std::abort();                                                       \
    }                                                                     \
  } while (0)

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
};

struct Image {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  uint32_t width = 0, height = 0;
};

class Boot {
 public:
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  uint32_t qfam = 0;
  VkQueue queue = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties mem_props{};
  VkPhysicalDeviceProperties props{};
  float timestamp_period_ns = 0.0f;
  bool timestamps_valid = false;
  std::string device_name;

  // device_exts: extra device extensions (e.g. dma-buf export) the caller
  // needs.
  void init(const std::vector<const char*>& device_exts = {}) {
    create_instance();
    pick_physical();
    create_device(device_exts);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = qfam;
    VK_CHECK(vkCreateCommandPool(device, &pci, nullptr, &pool));
  }

  void destroy() {
    if (pool)
      vkDestroyCommandPool(device, pool, nullptr);
    if (device)
      vkDestroyDevice(device, nullptr);
    if (instance)
      vkDestroyInstance(instance, nullptr);
  }

  uint32_t find_memory_type(uint32_t type_bits,
                            VkMemoryPropertyFlags want) const {
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
      if ((type_bits & (1u << i)) &&
          (mem_props.memoryTypes[i].propertyFlags & want) == want) {
        return i;
      }
    }
    std::fprintf(stderr, "no memory type for bits=%#x want=%#x\n", type_bits,
                 want);
    std::abort();
  }

  Buffer create_buffer(VkDeviceSize size,
                       VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags mem) {
    Buffer b;
    b.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &bi, nullptr, &b.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, b.buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, mem);
    VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &b.memory));
    VK_CHECK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
    return b;
  }

  Image create_image(uint32_t w,
                     uint32_t h,
                     VkFormat format,
                     VkImageUsageFlags usage,
                     VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL) {
    Image img;
    img.format = format;
    img.width = w;
    img.height = h;
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = format;
    ii.extent = {w, h, 1};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = tiling;
    ii.usage = usage;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device, &ii, nullptr, &img.image));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, img.image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &img.memory));
    VK_CHECK(vkBindImageMemory(device, img.image, img.memory, 0));

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device, &vi, nullptr, &img.view));
    return img;
  }

  void destroy_image(Image& img) {
    if (img.view)
      vkDestroyImageView(device, img.view, nullptr);
    if (img.image)
      vkDestroyImage(device, img.image, nullptr);
    if (img.memory)
      vkFreeMemory(device, img.memory, nullptr);
    img = {};
  }
  void destroy_buffer(Buffer& b) {
    if (b.buffer)
      vkDestroyBuffer(device, b.buffer, nullptr);
    if (b.memory)
      vkFreeMemory(device, b.memory, nullptr);
    b = {};
  }

  VkShaderModule load_shader(const uint32_t* code, uint32_t word_count) {
    VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    si.codeSize = static_cast<size_t>(word_count) * 4u;
    si.pCode = code;
    VkShaderModule m;
    VK_CHECK(vkCreateShaderModule(device, &si, nullptr, &m));
    return m;
  }

  VkCommandBuffer begin_cmd() {
    VkCommandBufferAllocateInfo ai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(device, &ai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    return cb;
  }

  void submit_wait(VkCommandBuffer cb) {
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(device, pool, 1, &cb);
  }

  // Simple layout transition helper for the spikes (full pipeline barrier).
  static void image_barrier(VkCommandBuffer cb,
                            VkImage image,
                            VkImageLayout from,
                            VkImageLayout to,
                            VkAccessFlags src_access,
                            VkAccessFlags dst_access,
                            VkPipelineStageFlags src_stage,
                            VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1,
                         &b);
  }

 private:
  void create_instance() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "water-ffi-spike";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;

    std::vector<const char*> layers;
    if (std::getenv("WATER_VK_VALIDATE")) {
      layers.push_back("VK_LAYER_KHRONOS_validation");
      ci.enabledLayerCount = 1;
      ci.ppEnabledLayerNames = layers.data();
    }
    // Needed to query dma-buf-capable physical device features later.
    const char* inst_exts[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME};
    ci.enabledExtensionCount = 2;
    ci.ppEnabledExtensionNames = inst_exts;
    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance));
  }

  void pick_physical() {
    uint32_t n = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &n, nullptr));
    if (n == 0) {
      std::fprintf(stderr, "no Vulkan physical devices\n");
      std::abort();
    }
    std::vector<VkPhysicalDevice> devs(n);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &n, devs.data()));

    const char* want =
        std::getenv("WATER_VK_DEVICE");  // substring of deviceName
    std::optional<VkPhysicalDevice> chosen;
    VkPhysicalDeviceProperties chosen_props{};
    for (auto d : devs) {
      VkPhysicalDeviceProperties p;
      vkGetPhysicalDeviceProperties(d, &p);
      if (want) {
        if (std::strstr(p.deviceName, want)) {
          chosen = d;
          chosen_props = p;
          break;
        }
      } else if (!chosen ||
                 (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
                  chosen_props.deviceType !=
                      VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) {
        chosen = d;
        chosen_props = p;
      }
    }
    if (!chosen) {
      std::fprintf(stderr, "no device matched WATER_VK_DEVICE=%s\n",
                   want ? want : "(any)");
      std::abort();
    }
    phys = *chosen;
    props = chosen_props;
    device_name = chosen_props.deviceName;
    timestamp_period_ns = chosen_props.limits.timestampPeriod;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
  }

  void create_device(const std::vector<const char*>& device_exts) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, fams.data());
    bool found = false;
    for (uint32_t i = 0; i < n; ++i) {
      if ((fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
          (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
        qfam = i;
        timestamps_valid = fams[i].timestampValidBits > 0;
        found = true;
        break;
      }
    }
    if (!found) {
      std::fprintf(stderr, "no graphics+compute queue family\n");
      std::abort();
    }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qfam;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    di.enabledExtensionCount = static_cast<uint32_t>(device_exts.size());
    di.ppEnabledExtensionNames =
        device_exts.empty() ? nullptr : device_exts.data();
    VK_CHECK(vkCreateDevice(phys, &di, nullptr, &device));
    vkGetDeviceQueue(device, qfam, 0, &queue);
  }
};

}  // namespace vkboot
