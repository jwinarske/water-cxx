// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#include "water/scene_pass.hpp"

#include <array>
#include <cstring>

namespace water {

namespace {
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

Result<VkShaderModule> make_module(VkDevice dev,
                                   std::span<const uint32_t> spirv) {
  const VkShaderModuleCreateInfo ci{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirv.size() * sizeof(uint32_t),
      .pCode = spirv.data(),
  };
  VkShaderModule m = VK_NULL_HANDLE;
  WATER_CHECK(vkCreateShaderModule(dev, &ci, nullptr, &m),
              "vkCreateShaderModule");
  return m;
}
}  // namespace

Result<ScenePass> ScenePass::create(const Device& dev,
                                    uint32_t width,
                                    uint32_t height,
                                    VkFormat color) {
  ScenePass p;
  p.dev_ = &dev;
  p.width_ = width;
  p.height_ = height;

  p.color_ = WATER_TRY(dev.create_image(
      {.width = width,
       .height = height,
       .format = color,
       .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT}));
  p.depth_ = WATER_TRY(
      dev.create_image({.width = width,
                        .height = height,
                        .format = kDepthFormat,
                        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                        .aspect = VK_IMAGE_ASPECT_DEPTH_BIT}));

  const std::array<VkAttachmentDescription, 2> att{{
      {.format = color,
       .samples = VK_SAMPLE_COUNT_1_BIT,
       .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
       .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
       .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
       .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
       .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
       .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
      {.format = kDepthFormat,
       .samples = VK_SAMPLE_COUNT_1_BIT,
       .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
       .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
       .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
       .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
       .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
       .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
  }};
  const VkAttachmentReference color_ref{
      0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  const VkAttachmentReference depth_ref{
      1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
  const VkSubpassDescription sub{
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_ref,
      .pDepthStencilAttachment = &depth_ref,
  };
  // Order the clear/writes after any prior use of these images.
  const VkSubpassDependency dep{
      .srcSubpass = VK_SUBPASS_EXTERNAL,
      .dstSubpass = 0,
      .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
  };
  const VkRenderPassCreateInfo rpci{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = att.size(),
      .pAttachments = att.data(),
      .subpassCount = 1,
      .pSubpasses = &sub,
      .dependencyCount = 1,
      .pDependencies = &dep,
  };
  WATER_CHECK(vkCreateRenderPass(dev.handle(), &rpci, nullptr, &p.render_pass_),
              "vkCreateRenderPass");

  const std::array<VkImageView, 2> views{p.color_.view, p.depth_.view};
  const VkFramebufferCreateInfo fci{
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = p.render_pass_,
      .attachmentCount = views.size(),
      .pAttachments = views.data(),
      .width = width,
      .height = height,
      .layers = 1,
  };
  WATER_CHECK(vkCreateFramebuffer(dev.handle(), &fci, nullptr, &p.fb_),
              "vkCreateFramebuffer");

  // Surface descriptor set: 4 combined image samplers. water is fetched in the
  // vertex stage too ( water VS), so cover VERTEX|FRAGMENT.
  std::array<VkDescriptorSetLayoutBinding, 4> binds{};
  for (uint32_t i = 0; i < 4; ++i)
    binds[i] = {.binding = i,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags =
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT};
  const VkDescriptorSetLayoutCreateInfo dlci{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = binds.size(),
      .pBindings = binds.data(),
  };
  WATER_CHECK(vkCreateDescriptorSetLayout(dev.handle(), &dlci, nullptr,
                                          &p.surface_dsl_),
              "vkCreateDescriptorSetLayout(surface)");
  const VkDescriptorPoolSize psize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                   4 * 8};
  const VkDescriptorPoolCreateInfo dpci{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 8,
      .poolSizeCount = 1,
      .pPoolSizes = &psize};
  WATER_CHECK(
      vkCreateDescriptorPool(dev.handle(), &dpci, nullptr, &p.desc_pool_),
      "vkCreateDescriptorPool(surface)");

  const VkPushConstantRange pcr{
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      .offset = 0,
      .size = sizeof(ScenePush),
  };
  const VkPipelineLayoutCreateInfo plci{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &p.surface_dsl_,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcr,
  };
  WATER_CHECK(vkCreatePipelineLayout(dev.handle(), &plci, nullptr, &p.layout_),
              "vkCreatePipelineLayout");
  return p;
}

Result<VkPipeline> ScenePass::make_pipeline(const PipelineDesc& desc) {
  VkShaderModule vs = WATER_TRY(make_module(dev_->handle(), desc.vert));
  VkShaderModule fs = WATER_TRY(make_module(dev_->handle(), desc.frag));
  const std::array<VkPipelineShaderStageCreateInfo, 2> stages{{
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = vs,
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = fs,
       .pName = "main"},
  }};
  const VkVertexInputBindingDescription bind{0, sizeof(glm::vec3),
                                             VK_VERTEX_INPUT_RATE_VERTEX};
  const VkVertexInputAttributeDescription attr{0, 0, VK_FORMAT_R32G32B32_SFLOAT,
                                               0};
  const VkPipelineVertexInputStateCreateInfo vi{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &bind,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &attr,
  };
  const VkPipelineInputAssemblyStateCreateInfo ia{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
  };
  const VkPipelineViewportStateCreateInfo vp{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
  };
  const VkPipelineRasterizationStateCreateInfo rs{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = desc.cull,
      // GL-authored winding (CCW front) with cull BACK shows the pool interior.
      // The negative-height viewport does not change which triangles are
      // front/back for culling here, so we keep CCW to match the reference cull
      // results.
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
  };
  const VkPipelineMultisampleStateCreateInfo ms{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
  };
  const VkPipelineDepthStencilStateCreateInfo depth{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = desc.depth_test,
      .depthWriteEnable = desc.depth_write,
      .depthCompareOp = VK_COMPARE_OP_LESS,
  };
  const VkPipelineColorBlendAttachmentState cba{.colorWriteMask = 0xf};
  const VkPipelineColorBlendStateCreateInfo cb{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &cba,
  };
  const std::array<VkDynamicState, 2> dyn{VK_DYNAMIC_STATE_VIEWPORT,
                                          VK_DYNAMIC_STATE_SCISSOR};
  const VkPipelineDynamicStateCreateInfo ds{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = dyn.size(),
      .pDynamicStates = dyn.data(),
  };
  const VkGraphicsPipelineCreateInfo gpci{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = stages.size(),
      .pStages = stages.data(),
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pDepthStencilState = &depth,
      .pColorBlendState = &cb,
      .pDynamicState = &ds,
      .layout = layout_,
      .renderPass = render_pass_,
  };
  VkPipeline pipe = VK_NULL_HANDLE;
  const VkResult pr = vkCreateGraphicsPipelines(dev_->handle(), VK_NULL_HANDLE,
                                                1, &gpci, nullptr, &pipe);
  vkDestroyShaderModule(dev_->handle(), vs, nullptr);
  vkDestroyShaderModule(dev_->handle(), fs, nullptr);
  if (pr != VK_SUCCESS)
    return fail(pr, "vkCreateGraphicsPipelines(scene)");
  pipelines_.push_back(pipe);
  return pipe;
}

void ScenePass::begin(VkCommandBuffer cmd, glm::vec4 clear_color) {
  std::array<VkClearValue, 2> clears{};
  clears[0].color = {
      {clear_color.r, clear_color.g, clear_color.b, clear_color.a}};
  clears[1].depthStencil = {1.0f, 0};
  const VkRenderPassBeginInfo rpbi{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass_,
      .framebuffer = fb_,
      .renderArea = {{0, 0}, {width_, height_}},
      .clearValueCount = clears.size(),
      .pClearValues = clears.data(),
  };
  vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

  // negative-height viewport flips Y so the GL-authored projection renders
  // upright.
  const VkViewport viewport{
      0, float(height_), float(width_), -float(height_), 0, 1};
  const VkRect2D scissor{{0, 0}, {width_, height_}};
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
}

Result<VkDescriptorSet> ScenePass::make_surface_set(
    const std::array<Binding, 4>& bindings) {
  const VkDescriptorSetAllocateInfo ai{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = desc_pool_,
      .descriptorSetCount = 1,
      .pSetLayouts = &surface_dsl_};
  VkDescriptorSet set = VK_NULL_HANDLE;
  WATER_CHECK(vkAllocateDescriptorSets(dev_->handle(), &ai, &set),
              "vkAllocateDescriptorSets(surface)");
  std::array<VkDescriptorImageInfo, 4> infos{};
  std::array<VkWriteDescriptorSet, 4> writes{};
  for (uint32_t i = 0; i < 4; ++i) {
    infos[i] = {bindings[i].sampler, bindings[i].view,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    writes[i] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 .dstSet = set,
                 .dstBinding = i,
                 .descriptorCount = 1,
                 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                 .pImageInfo = &infos[i]};
  }
  vkUpdateDescriptorSets(dev_->handle(), writes.size(), writes.data(), 0,
                         nullptr);
  return set;
}

void ScenePass::draw(VkCommandBuffer cmd,
                     VkPipeline pipeline,
                     const ScenePush& push,
                     const Mesh& mesh,
                     VkDescriptorSet set) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  if (set != VK_NULL_HANDLE)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1,
                            &set, 0, nullptr);
  vkCmdPushConstants(cmd, layout_,
                     VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(ScenePush), &push);
  const VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vbuf.buffer, &offset);
  vkCmdBindIndexBuffer(cmd, mesh.ibuf.buffer, 0, VK_INDEX_TYPE_UINT32);
  vkCmdDrawIndexed(cmd, mesh.index_count, 1, 0, 0, 0);
}

void ScenePass::end(VkCommandBuffer cmd) {
  vkCmdEndRenderPass(cmd);
}

Result<std::vector<uint8_t>> ScenePass::readback_color(
    const Device& dev) const {
  const VkDeviceSize bytes = VkDeviceSize(width_) * height_ * 4;
  Buffer host = WATER_TRY(
      dev.create_host_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT));
  WATER_TRYV(dev.submit_now([&](VkCommandBuffer cmd) {
    // color_ is already TRANSFER_SRC_OPTIMAL from the render pass finalLayout.
    const VkBufferImageCopy region{
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {width_, height_, 1}};
    vkCmdCopyImageToBuffer(cmd, color_.image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, host.buffer, 1,
                           &region);
  }));
  std::vector<uint8_t> out(bytes);
  std::memcpy(out.data(), host.mapped, bytes);
  const_cast<Device&>(dev).destroy_buffer(host);
  return out;
}

ScenePass::~ScenePass() {
  destroy();
}

void ScenePass::destroy() noexcept {
  if (dev_) {
    VkDevice d = dev_->handle();
    for (VkPipeline p : pipelines_)
      vkDestroyPipeline(d, p, nullptr);
    if (layout_)
      vkDestroyPipelineLayout(d, layout_, nullptr);
    if (desc_pool_)
      vkDestroyDescriptorPool(d, desc_pool_, nullptr);
    if (surface_dsl_)
      vkDestroyDescriptorSetLayout(d, surface_dsl_, nullptr);
    if (fb_)
      vkDestroyFramebuffer(d, fb_, nullptr);
    if (render_pass_)
      vkDestroyRenderPass(d, render_pass_, nullptr);
    dev_->destroy_image(color_);
    dev_->destroy_image(depth_);
  }
  pipelines_.clear();
  layout_ = VK_NULL_HANDLE;
  fb_ = VK_NULL_HANDLE;
  render_pass_ = VK_NULL_HANDLE;
  dev_ = nullptr;
}

ScenePass::ScenePass(ScenePass&& o) noexcept {
  *this = std::move(o);
}

ScenePass& ScenePass::operator=(ScenePass&& o) noexcept {
  if (this != &o) {
    destroy();
    dev_ = o.dev_;
    width_ = o.width_;
    height_ = o.height_;
    color_ = o.color_;
    depth_ = o.depth_;
    render_pass_ = o.render_pass_;
    fb_ = o.fb_;
    surface_dsl_ = o.surface_dsl_;
    desc_pool_ = o.desc_pool_;
    layout_ = o.layout_;
    pipelines_ = std::move(o.pipelines_);
    o.dev_ = nullptr;
    o.color_ = {};
    o.depth_ = {};
    o.render_pass_ = VK_NULL_HANDLE;
    o.fb_ = VK_NULL_HANDLE;
    o.surface_dsl_ = VK_NULL_HANDLE;
    o.desc_pool_ = VK_NULL_HANDLE;
    o.layout_ = VK_NULL_HANDLE;
  }
  return *this;
}

}  // namespace water
