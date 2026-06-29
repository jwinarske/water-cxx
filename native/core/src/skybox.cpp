// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#include "water/skybox.hpp"

#include "skybox_shaders.h"  // skybox_vert_spv, skybox_frag_spv

#include <array>
#include <cstring>
#include <span>

namespace water {

namespace {
struct SkyPush {
  glm::mat4 inv_vp;
  glm::vec4 eye;
};

Result<VkShaderModule> mk(VkDevice dev, std::span<const uint32_t> spirv) {
  const VkShaderModuleCreateInfo ci{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirv.size() * 4,
      .pCode = spirv.data()};
  VkShaderModule m = VK_NULL_HANDLE;
  WATER_CHECK(vkCreateShaderModule(dev, &ci, nullptr, &m),
              "vkCreateShaderModule(skybox)");
  return m;
}
}  // namespace

Result<Skybox> Skybox::create(const Device& dev,
                              VkRenderPass render_pass,
                              const Texture& sky,
                              VkSampler sampler) {
  Skybox s;
  s.dev_ = &dev;

  const VkDescriptorSetLayoutBinding b{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
  };
  const VkDescriptorSetLayoutCreateInfo dlci{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &b};
  WATER_CHECK(
      vkCreateDescriptorSetLayout(dev.handle(), &dlci, nullptr, &s.dsl_),
      "vkCreateDescriptorSetLayout(skybox)");

  const VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
  const VkDescriptorPoolCreateInfo dpci{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &ps};
  WATER_CHECK(vkCreateDescriptorPool(dev.handle(), &dpci, nullptr, &s.pool_),
              "vkCreateDescriptorPool(skybox)");
  const VkDescriptorSetAllocateInfo ai{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = s.pool_,
      .descriptorSetCount = 1,
      .pSetLayouts = &s.dsl_};
  WATER_CHECK(vkAllocateDescriptorSets(dev.handle(), &ai, &s.set_),
              "vkAllocateDescriptorSets(skybox)");
  const VkDescriptorImageInfo dii{sampler, sky.view,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  const VkWriteDescriptorSet w{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = s.set_,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &dii};
  vkUpdateDescriptorSets(dev.handle(), 1, &w, 0, nullptr);

  const VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(SkyPush)};
  const VkPipelineLayoutCreateInfo plci{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &s.dsl_,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcr};
  WATER_CHECK(vkCreatePipelineLayout(dev.handle(), &plci, nullptr, &s.layout_),
              "vkCreatePipelineLayout(skybox)");

  VkShaderModule vs = WATER_TRY(
      mk(dev.handle(), std::span(skybox_vert_spv, skybox_vert_spv_count)));
  VkShaderModule fs = WATER_TRY(
      mk(dev.handle(), std::span(skybox_frag_spv, skybox_frag_spv_count)));
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
  const VkPipelineVertexInputStateCreateInfo vi{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  const VkPipelineInputAssemblyStateCreateInfo ia{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
  const VkPipelineViewportStateCreateInfo vp{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1};
  const VkPipelineRasterizationStateCreateInfo rs{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f};
  const VkPipelineMultisampleStateCreateInfo ms{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
  const VkPipelineDepthStencilStateCreateInfo depth{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_FALSE,
      .depthWriteEnable = VK_FALSE};
  const VkPipelineColorBlendAttachmentState cba{.colorWriteMask = 0xf};
  const VkPipelineColorBlendStateCreateInfo cb{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &cba};
  const std::array<VkDynamicState, 2> dyn{VK_DYNAMIC_STATE_VIEWPORT,
                                          VK_DYNAMIC_STATE_SCISSOR};
  const VkPipelineDynamicStateCreateInfo dsci{
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
      .pDepthStencilState = &depth,
      .pColorBlendState = &cb,
      .pDynamicState = &dsci,
      .layout = s.layout_,
      .renderPass = render_pass};
  const VkResult pr = vkCreateGraphicsPipelines(dev.handle(), VK_NULL_HANDLE, 1,
                                                &gpci, nullptr, &s.pipeline_);
  vkDestroyShaderModule(dev.handle(), vs, nullptr);
  vkDestroyShaderModule(dev.handle(), fs, nullptr);
  if (pr != VK_SUCCESS)
    return fail(pr, "vkCreateGraphicsPipelines(skybox)");
  return s;
}

void Skybox::record(VkCommandBuffer cmd,
                    const glm::mat4& inv_vp,
                    const glm::vec3& eye) const {
  const SkyPush push{inv_vp, glm::vec4(eye, 1.0f)};
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1,
                          &set_, 0, nullptr);
  vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                     sizeof(SkyPush), &push);
  vkCmdDraw(cmd, 3, 1, 0, 0);
}

Skybox::~Skybox() {
  destroy();
}

void Skybox::destroy() noexcept {
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
  }
  dev_ = nullptr;
  pipeline_ = VK_NULL_HANDLE;
  layout_ = VK_NULL_HANDLE;
  pool_ = VK_NULL_HANDLE;
  dsl_ = VK_NULL_HANDLE;
}

Skybox::Skybox(Skybox&& o) noexcept {
  *this = std::move(o);
}

Skybox& Skybox::operator=(Skybox&& o) noexcept {
  if (this != &o) {
    destroy();
    dev_ = o.dev_;
    dsl_ = o.dsl_;
    pool_ = o.pool_;
    set_ = o.set_;
    layout_ = o.layout_;
    pipeline_ = o.pipeline_;
    o.dev_ = nullptr;
    o.dsl_ = VK_NULL_HANDLE;
    o.pool_ = VK_NULL_HANDLE;
    o.layout_ = VK_NULL_HANDLE;
    o.pipeline_ = VK_NULL_HANDLE;
  }
  return *this;
}

}  // namespace water
