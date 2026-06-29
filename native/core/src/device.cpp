// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#include "water/device.hpp"

#include <array>
#include <cstring>
#include <optional>
#include <vector>

namespace water {

const char* to_string(VkResult r) noexcept {
  switch (r) {
    case VK_SUCCESS:
      return "VK_SUCCESS";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
      return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
      return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:
      return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
      return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    default:
      return "VK_ERROR_<other>";
  }
}

namespace {

DeviceClass classify(uint32_t vendor_id) {
  // ARM / Qualcomm / Broadcom / Imagination → Mobile.
  switch (vendor_id) {
    case 0x13B5:  // ARM (Mali)
    case 0x5143:  // Qualcomm (Adreno)
    case 0x14E4:  // Broadcom (VideoCore / V3D)
    case 0x1010:  // Imagination (PowerVR)
      return DeviceClass::Mobile;
    default:
      return DeviceClass::Desktop;
  }
}

// gate: RGBA32F usable as a filterable sampled color attachment?
bool float32_filterable(VkPhysicalDevice phys) {
  VkFormatProperties fp{};
  vkGetPhysicalDeviceFormatProperties(phys, VK_FORMAT_R32G32B32A32_SFLOAT, &fp);
  const VkFormatFeatureFlags need =
      VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
  return (fp.optimalTilingFeatures & need) == need;
}

// Physical-device preference (higher wins). Real GPUs beat a CPU (llvmpipe)
// fallback, so multi-device boards (e.g. Pi 5: V3D + llvmpipe) never silently
// land on software unless it is the only device offered.
int type_rank(VkPhysicalDeviceType t) {
  switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return 4;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return 3;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return 0;  // last resort
    default:
      return 1;  // OTHER
  }
}

}  // namespace

Result<Device> Device::create(const DeviceConfig& cfg) {
  Device d;

  // ---- instance ----
  const VkApplicationInfo app{
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "water-ffi",
      .applicationVersion = 0,
      .pEngineName = "water-engine-slice",
      .engineVersion = 0,
      .apiVersion = VK_API_VERSION_1_2,
  };
  const char* layer = "VK_LAYER_KHRONOS_validation";
  const VkInstanceCreateInfo ici{
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
      .enabledLayerCount = cfg.enable_validation ? 1u : 0u,
      .ppEnabledLayerNames = cfg.enable_validation ? &layer : nullptr,
  };
  WATER_CHECK(vkCreateInstance(&ici, nullptr, &d.instance_),
              "vkCreateInstance");

  // ---- physical device ----
  uint32_t n = 0;
  WATER_CHECK(vkEnumeratePhysicalDevices(d.instance_, &n, nullptr),
              "vkEnumeratePhysicalDevices");
  if (n == 0) {
    d.destroy();
    return fail(VK_ERROR_INITIALIZATION_FAILED, "no Vulkan physical devices");
  }
  std::vector<VkPhysicalDevice> devs(n);
  WATER_CHECK(vkEnumeratePhysicalDevices(d.instance_, &n, devs.data()),
              "vkEnumeratePhysicalDevices(2)");

  std::optional<VkPhysicalDevice> chosen;
  VkPhysicalDeviceProperties chosen_props{};
  for (auto pd : devs) {
    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(pd, &p);
    if (!cfg.device_substr.empty()) {
      if (std::string_view(p.deviceName).find(cfg.device_substr) !=
          std::string_view::npos) {
        chosen = pd;
        chosen_props = p;
        break;
      }
    } else if (!chosen ||
               type_rank(p.deviceType) > type_rank(chosen_props.deviceType)) {
      // Highest-ranked device wins; ties keep enumeration order. A CPU device
      // is taken only when nothing better is present.
      chosen = pd;
      chosen_props = p;
    }
  }
  if (!chosen) {
    d.destroy();
    return fail(VK_ERROR_INITIALIZATION_FAILED,
                "no physical device matched device_substr");
  }
  d.phys_ = *chosen;
  vkGetPhysicalDeviceMemoryProperties(d.phys_, &d.mem_props_);

  // ---- queue family: one graphics+compute queue (single-threaded renderer)
  // ----
  uint32_t qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(d.phys_, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> fams(qn);
  vkGetPhysicalDeviceQueueFamilyProperties(d.phys_, &qn, fams.data());
  std::optional<uint32_t> qf;
  for (uint32_t i = 0; i < qn; ++i) {
    const auto want = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    if ((fams[i].queueFlags & want) == want) {
      qf = i;
      d.caps_.timestamps_valid = fams[i].timestampValidBits > 0;
      break;
    }
  }
  if (!qf) {
    d.destroy();
    return fail(VK_ERROR_INITIALIZATION_FAILED,
                "no graphics+compute queue family");
  }
  d.qfam_ = *qf;

  const float prio = 1.0f;
  const VkDeviceQueueCreateInfo qci{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = d.qfam_,
      .queueCount = 1,
      .pQueuePriorities = &prio,
  };
  // Optional dma-buf export (for presenting the rendered image to a Wayland
  // compositor or KMS plane): exportable device memory as a dma-buf fd.
  std::vector<const char*> dev_exts;
  if (cfg.dmabuf_export) {
    dev_exts.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    dev_exts.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
    // Allocate exportable images by DRM format modifier (the compatible,
    // validation-clean way to make dma-buf/scanout images — not plain LINEAR).
    dev_exts.push_back(VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME);
    // Ownership transfer to VK_QUEUE_FAMILY_FOREIGN_EXT engages implicit
    // dma-buf fencing, so an external consumer (compositor / KMS) waits on our
    // GPU writes without a CPU stall — the producer-side acquire fence.
    dev_exts.push_back(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
    // Export semaphores as sync_file / DRM-syncobj fds for explicit-sync
    // present (wp_linux_drm_syncobj timeline points).
    dev_exts.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  }
  // Timeline semaphores back the explicit-sync acquire/release timelines.
  const VkPhysicalDeviceTimelineSemaphoreFeatures timeline_feat{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
      .timelineSemaphore = VK_TRUE,
  };
  const VkDeviceCreateInfo dci{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = cfg.dmabuf_export ? &timeline_feat : nullptr,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
      .enabledExtensionCount = uint32_t(dev_exts.size()),
      .ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data(),
  };
  WATER_CHECK(vkCreateDevice(d.phys_, &dci, nullptr, &d.device_),
              "vkCreateDevice");
  vkGetDeviceQueue(d.device_, d.qfam_, 0, &d.queue_);

  const VkCommandPoolCreateInfo pci{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = d.qfam_,
  };
  WATER_CHECK(vkCreateCommandPool(d.device_, &pci, nullptr, &d.pool_),
              "vkCreateCommandPool");

  // ---- caps + format selection ----
  d.caps_.device_name = chosen_props.deviceName;
  d.caps_.device_class = classify(chosen_props.vendorID);
  d.caps_.timestamp_period_ns = chosen_props.limits.timestampPeriod;
  d.caps_.float32_filterable = float32_filterable(d.phys_);
  d.caps_.heightfield_format = (d.caps_.float32_filterable && !cfg.force_half)
                                   ? VK_FORMAT_R32G32B32A32_SFLOAT
                                   : VK_FORMAT_R16G16B16A16_SFLOAT;
  return d;
}

Device::~Device() {
  destroy();
}

void Device::destroy() noexcept {
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);
    if (pool_ != VK_NULL_HANDLE)
      vkDestroyCommandPool(device_, pool_, nullptr);
    vkDestroyDevice(device_, nullptr);
  }
  if (instance_ != VK_NULL_HANDLE)
    vkDestroyInstance(instance_, nullptr);
  instance_ = VK_NULL_HANDLE;
  phys_ = VK_NULL_HANDLE;
  device_ = VK_NULL_HANDLE;
  pool_ = VK_NULL_HANDLE;
  queue_ = VK_NULL_HANDLE;
}

Device::Device(Device&& o) noexcept {
  *this = std::move(o);
}

Device& Device::operator=(Device&& o) noexcept {
  if (this != &o) {
    destroy();
    instance_ = o.instance_;
    phys_ = o.phys_;
    device_ = o.device_;
    pool_ = o.pool_;
    queue_ = o.queue_;
    qfam_ = o.qfam_;
    mem_props_ = o.mem_props_;
    caps_ = std::move(o.caps_);
    o.instance_ = VK_NULL_HANDLE;
    o.phys_ = VK_NULL_HANDLE;
    o.device_ = VK_NULL_HANDLE;
    o.pool_ = VK_NULL_HANDLE;
    o.queue_ = VK_NULL_HANDLE;
  }
  return *this;
}

Result<uint32_t> Device::find_memory_type(uint32_t type_bits,
                                          VkMemoryPropertyFlags want) const {
  for (uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i) {
    if ((type_bits & (1u << i)) &&
        (mem_props_.memoryTypes[i].propertyFlags & want) == want) {
      return i;
    }
  }
  return fail(VK_ERROR_OUT_OF_DEVICE_MEMORY, "no compatible memory type");
}

Result<Image> Device::create_image(const ImageDesc& desc) const {
  Image img{.format = desc.format, .width = desc.width, .height = desc.height};
  const VkImageCreateInfo ici{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = desc.format,
      .extent = {desc.width, desc.height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = desc.tiling,
      .usage = desc.usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };
  WATER_CHECK(vkCreateImage(device_, &ici, nullptr, &img.image),
              "vkCreateImage");

  VkMemoryRequirements req{};
  vkGetImageMemoryRequirements(device_, img.image, &req);
  const uint32_t mt = WATER_TRY(find_memory_type(
      req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
  const VkMemoryAllocateInfo ai{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = mt,
  };
  WATER_CHECK(vkAllocateMemory(device_, &ai, nullptr, &img.memory),
              "vkAllocateMemory(image)");
  WATER_CHECK(vkBindImageMemory(device_, img.image, img.memory, 0),
              "vkBindImageMemory");

  const VkImageViewCreateInfo vci{
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img.image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = desc.format,
      .subresourceRange = {desc.aspect, 0, 1, 0, 1},
  };
  WATER_CHECK(vkCreateImageView(device_, &vci, nullptr, &img.view),
              "vkCreateImageView");
  return img;
}

void Device::destroy_image(Image& img) const noexcept {
  if (img.view)
    vkDestroyImageView(device_, img.view, nullptr);
  if (img.image)
    vkDestroyImage(device_, img.image, nullptr);
  if (img.memory)
    vkFreeMemory(device_, img.memory, nullptr);
  img = {};
}

Result<Buffer> Device::create_host_buffer(VkDeviceSize size,
                                          VkBufferUsageFlags usage) const {
  Buffer b{.size = size};
  const VkBufferCreateInfo bci{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  WATER_CHECK(vkCreateBuffer(device_, &bci, nullptr, &b.buffer),
              "vkCreateBuffer");
  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(device_, b.buffer, &req);
  const uint32_t mt = WATER_TRY(find_memory_type(
      req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
  const VkMemoryAllocateInfo ai{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = mt,
  };
  WATER_CHECK(vkAllocateMemory(device_, &ai, nullptr, &b.memory),
              "vkAllocateMemory(buffer)");
  WATER_CHECK(vkBindBufferMemory(device_, b.buffer, b.memory, 0),
              "vkBindBufferMemory");
  WATER_CHECK(vkMapMemory(device_, b.memory, 0, size, 0, &b.mapped),
              "vkMapMemory");
  return b;
}

void Device::destroy_buffer(Buffer& b) const noexcept {
  if (b.buffer)
    vkDestroyBuffer(device_, b.buffer, nullptr);
  if (b.memory)
    vkFreeMemory(device_, b.memory, nullptr);
  b = {};
}

Result<void> Device::submit_now(
    const std::function<void(VkCommandBuffer)>& record) const {
  const VkCommandBufferAllocateInfo ai{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool_,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  VkCommandBuffer cb = VK_NULL_HANDLE;
  WATER_CHECK(vkAllocateCommandBuffers(device_, &ai, &cb),
              "vkAllocateCommandBuffers");

  const VkCommandBufferBeginInfo bi{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  WATER_CHECK(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
  record(cb);
  WATER_CHECK(vkEndCommandBuffer(cb), "vkEndCommandBuffer");

  const VkSubmitInfo si{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cb,
  };
  WATER_CHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
  WATER_CHECK(vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
  vkFreeCommandBuffers(device_, pool_, 1, &cb);
  return {};
}

}  // namespace water
