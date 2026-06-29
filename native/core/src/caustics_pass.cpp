// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#include "water/caustics_pass.hpp"

#include "caustics_shaders.h"  // caustics_vert_spv, caustics_frag_spv

#include <array>

namespace water {

namespace {
constexpr VkFormat kFmt = VK_FORMAT_R8G8B8A8_UNORM;

Result<VkShaderModule> mk(VkDevice dev, std::span<const uint32_t> spirv) {
  const VkShaderModuleCreateInfo ci{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirv.size() * 4,
      .pCode = spirv.data()};
  VkShaderModule m = VK_NULL_HANDLE;
  WATER_CHECK(vkCreateShaderModule(dev, &ci, nullptr, &m),
              "vkCreateShaderModule(caustics)");
  return m;
}
}  // namespace

Result<CausticsPass> CausticsPass::create(const Device& dev,
                                          uint32_t size,
                                          uint32_t buffers) {
  CausticsPass c;
  c.dev_ = &dev;
  c.size_ = size;

  // Render pass: one RGBA8 color attachment, CLEAR to 0, no depth; ends
  // SHADER_READ.
  const VkAttachmentDescription att{
      .format = kFmt,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
  };
  const VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  const VkSubpassDescription sub{
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &ref};
  // Order a prior frame's sampling of this buffer before we overwrite it
  // (double-buffer hazard): FRAGMENT_SHADER read -> COLOR_ATTACHMENT_OUTPUT
  // write.
  const VkSubpassDependency dep{
      .srcSubpass = VK_SUBPASS_EXTERNAL,
      .dstSubpass = 0,
      .srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
  };
  const VkRenderPassCreateInfo rpci{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &att,
      .subpassCount = 1,
      .pSubpasses = &sub,
      .dependencyCount = 1,
      .pDependencies = &dep};
  WATER_CHECK(vkCreateRenderPass(dev.handle(), &rpci, nullptr, &c.render_pass_),
              "vkCreateRenderPass(caustics)");

  for (uint32_t i = 0; i < buffers; ++i) {
    Image img = WATER_TRY(
        dev.create_image({.width = size,
                          .height = size,
                          .format = kFmt,
                          .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                   VK_IMAGE_USAGE_SAMPLED_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT}));
    const VkFramebufferCreateInfo fci{
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = c.render_pass_,
        .attachmentCount = 1,
        .pAttachments = &img.view,
        .width = size,
        .height = size,
        .layers = 1};
    VkFramebuffer fb = VK_NULL_HANDLE;
    WATER_CHECK(vkCreateFramebuffer(dev.handle(), &fci, nullptr, &fb),
                "vkCreateFramebuffer(caustics)");
    c.images_.push_back(img);
    c.fb_.push_back(fb);
  }

  // Descriptor: binding 0 = water heightfield (sampled in the caustics VS).
  const VkDescriptorSetLayoutBinding b{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT};
  const VkDescriptorSetLayoutCreateInfo dlci{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &b};
  WATER_CHECK(
      vkCreateDescriptorSetLayout(dev.handle(), &dlci, nullptr, &c.dsl_),
      "vkCreateDescriptorSetLayout(caustics)");
  const VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4};
  const VkDescriptorPoolCreateInfo dpci{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 4,
      .poolSizeCount = 1,
      .pPoolSizes = &ps};
  WATER_CHECK(vkCreateDescriptorPool(dev.handle(), &dpci, nullptr, &c.pool_),
              "vkCreateDescriptorPool(caustics)");

  const VkPushConstantRange pcr{
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
      sizeof(ScenePush)};
  const VkPipelineLayoutCreateInfo plci{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &c.dsl_,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcr};
  WATER_CHECK(vkCreatePipelineLayout(dev.handle(), &plci, nullptr, &c.layout_),
              "vkCreatePipelineLayout(caustics)");

  VkShaderModule vs = WATER_TRY(
      mk(dev.handle(), std::span(caustics_vert_spv, caustics_vert_spv_count)));
  VkShaderModule fs = WATER_TRY(
      mk(dev.handle(), std::span(caustics_frag_spv, caustics_frag_spv_count)));
  const std::array<VkPipelineShaderStageCreateInfo, 2> stages{
      {{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vs,
        .pName = "main"},
       {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fs,
        .pName = "main"}}};
  const VkVertexInputBindingDescription bind{0, sizeof(glm::vec3),
                                             VK_VERTEX_INPUT_RATE_VERTEX};
  const VkVertexInputAttributeDescription attr{0, 0, VK_FORMAT_R32G32B32_SFLOAT,
                                               0};
  const VkPipelineVertexInputStateCreateInfo vi{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &bind,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &attr};
  const VkPipelineInputAssemblyStateCreateInfo ia{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
  const VkPipelineViewportStateCreateInfo vp{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1};
  // No cull, no depth, no blend (opaque last-write).
  const VkPipelineRasterizationStateCreateInfo rs{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f};
  const VkPipelineMultisampleStateCreateInfo ms{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
  const VkPipelineColorBlendAttachmentState cba{.colorWriteMask = 0xf};
  const VkPipelineColorBlendStateCreateInfo cb{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &cba};
  const std::array<VkDynamicState, 2> dyn{VK_DYNAMIC_STATE_VIEWPORT,
                                          VK_DYNAMIC_STATE_SCISSOR};
  const VkPipelineDynamicStateCreateInfo ds{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2,
      .pDynamicStates = dyn.data()};
  const VkGraphicsPipelineCreateInfo gpci{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages.data(),
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pColorBlendState = &cb,
      .pDynamicState = &ds,
      .layout = c.layout_,
      .renderPass = c.render_pass_};
  const VkResult pr = vkCreateGraphicsPipelines(dev.handle(), VK_NULL_HANDLE, 1,
                                                &gpci, nullptr, &c.pipeline_);
  vkDestroyShaderModule(dev.handle(), vs, nullptr);
  vkDestroyShaderModule(dev.handle(), fs, nullptr);
  if (pr != VK_SUCCESS)
    return fail(pr, "vkCreateGraphicsPipelines(caustics)");
  return c;
}

Result<VkDescriptorSet> CausticsPass::make_water_set(VkImageView water,
                                                     VkSampler sampler) {
  const VkDescriptorSetAllocateInfo ai{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool_,
      .descriptorSetCount = 1,
      .pSetLayouts = &dsl_};
  VkDescriptorSet set = VK_NULL_HANDLE;
  WATER_CHECK(vkAllocateDescriptorSets(dev_->handle(), &ai, &set),
              "vkAllocateDescriptorSets(caustics)");
  const VkDescriptorImageInfo dii{sampler, water,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  const VkWriteDescriptorSet w{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &dii};
  vkUpdateDescriptorSets(dev_->handle(), 1, &w, 0, nullptr);
  return set;
}

void CausticsPass::render(VkCommandBuffer cmd,
                          uint32_t index,
                          const ScenePush& push,
                          const Mesh& plane,
                          VkDescriptorSet water_set) {
  VkClearValue clear{};
  clear.color = {{0, 0, 0, 0}};
  const VkRenderPassBeginInfo rpbi{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass_,
      .framebuffer = fb_[index],
      .renderArea = {{0, 0}, {size_, size_}},
      .clearValueCount = 1,
      .pClearValues = &clear};
  vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
  // Positive-height viewport: generation NDC matches the scene's caustic
  // sampling UV.
  const VkViewport viewport{0, 0, float(size_), float(size_), 0, 1};
  const VkRect2D scissor{{0, 0}, {size_, size_}};
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1,
                          &water_set, 0, nullptr);
  vkCmdPushConstants(cmd, layout_,
                     VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(ScenePush), &push);
  const VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(cmd, 0, 1, &plane.vbuf.buffer, &offset);
  vkCmdBindIndexBuffer(cmd, plane.ibuf.buffer, 0, VK_INDEX_TYPE_UINT32);
  vkCmdDrawIndexed(cmd, plane.index_count, 1, 0, 0, 0);
  vkCmdEndRenderPass(cmd);
}

CausticsPass::~CausticsPass() {
  destroy();
}

void CausticsPass::destroy() noexcept {
  if (dev_) {
    VkDevice d = dev_->handle();
    if (pipeline_)
      vkDestroyPipeline(d, pipeline_, nullptr);
    if (layout_)
      vkDestroyPipelineLayout(d, layout_, nullptr);
    if (pool_)
      vkDestroyDescriptorPool(d, pool_, nullptr);
    if (dsl_)
      vkDestroyDescriptorSetLayout(d, dsl_, nullptr);
    for (VkFramebuffer fb : fb_)
      vkDestroyFramebuffer(d, fb, nullptr);
    if (render_pass_)
      vkDestroyRenderPass(d, render_pass_, nullptr);
    for (Image& im : images_)
      dev_->destroy_image(im);
  }
  images_.clear();
  fb_.clear();
  render_pass_ = VK_NULL_HANDLE;
  dsl_ = VK_NULL_HANDLE;
  pool_ = VK_NULL_HANDLE;
  layout_ = VK_NULL_HANDLE;
  pipeline_ = VK_NULL_HANDLE;
  dev_ = nullptr;
}

CausticsPass::CausticsPass(CausticsPass&& o) noexcept {
  *this = std::move(o);
}

CausticsPass& CausticsPass::operator=(CausticsPass&& o) noexcept {
  if (this != &o) {
    destroy();
    dev_ = o.dev_;
    size_ = o.size_;
    images_ = std::move(o.images_);
    fb_ = std::move(o.fb_);
    render_pass_ = o.render_pass_;
    dsl_ = o.dsl_;
    pool_ = o.pool_;
    layout_ = o.layout_;
    pipeline_ = o.pipeline_;
    o.dev_ = nullptr;
    o.render_pass_ = VK_NULL_HANDLE;
    o.dsl_ = VK_NULL_HANDLE;
    o.pool_ = VK_NULL_HANDLE;
    o.layout_ = VK_NULL_HANDLE;
    o.pipeline_ = VK_NULL_HANDLE;
  }
  return *this;
}

}  // namespace water
