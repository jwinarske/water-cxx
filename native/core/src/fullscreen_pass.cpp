// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#include "water/fullscreen_pass.hpp"

namespace water {

namespace {

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

Result<FullscreenPass> FullscreenPass::create(const Device& dev,
                                              const FullscreenPassDesc& desc) {
  FullscreenPass p;
  p.device_ = dev.handle();
  p.push_stages_ = desc.push_stages;
  p.push_size_ = desc.push_constant_size;

  // ---- render pass: single color attachment ----
  const VkAttachmentDescription att{
      .format = desc.color_format,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = desc.load_op,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      // LOAD_OP_LOAD needs a defined source layout; CLEAR/DONT_CARE start
      // UNDEFINED.
      .initialLayout = desc.load_op == VK_ATTACHMENT_LOAD_OP_LOAD
                           ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                           : VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = desc.final_layout,
  };
  const VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  const VkSubpassDescription sub{
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &ref,
  };
  const VkRenderPassCreateInfo rpci{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &att,
      .subpassCount = 1,
      .pSubpasses = &sub,
  };
  WATER_CHECK(vkCreateRenderPass(p.device_, &rpci, nullptr, &p.render_pass_),
              "vkCreateRenderPass");

  // ---- pipeline layout ----
  const VkPushConstantRange pcr{
      .stageFlags = desc.push_stages,
      .offset = 0,
      .size = desc.push_constant_size,
  };
  const VkPipelineLayoutCreateInfo plci{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = desc.set_layout != VK_NULL_HANDLE ? 1u : 0u,
      .pSetLayouts =
          desc.set_layout != VK_NULL_HANDLE ? &desc.set_layout : nullptr,
      .pushConstantRangeCount = desc.push_constant_size > 0 ? 1u : 0u,
      .pPushConstantRanges = desc.push_constant_size > 0 ? &pcr : nullptr,
  };
  WATER_CHECK(vkCreatePipelineLayout(p.device_, &plci, nullptr, &p.layout_),
              "vkCreatePipelineLayout");

  // ---- graphics pipeline (no vertex input; dynamic viewport/scissor) ----
  VkShaderModule vs = WATER_TRY(make_module(p.device_, desc.vert_spirv));
  VkShaderModule fs = WATER_TRY(make_module(p.device_, desc.frag_spirv));
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
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
  };
  const VkPipelineMultisampleStateCreateInfo ms{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
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
      .pColorBlendState = &cb,
      .pDynamicState = &ds,
      .layout = p.layout_,
      .renderPass = p.render_pass_,
  };
  const VkResult pr = vkCreateGraphicsPipelines(p.device_, VK_NULL_HANDLE, 1,
                                                &gpci, nullptr, &p.pipeline_);
  vkDestroyShaderModule(p.device_, vs, nullptr);
  vkDestroyShaderModule(p.device_, fs, nullptr);
  if (pr != VK_SUCCESS)
    return fail(pr, "vkCreateGraphicsPipelines");
  return p;
}

FullscreenPass::~FullscreenPass() {
  destroy();
}

void FullscreenPass::destroy() noexcept {
  if (pipeline_)
    vkDestroyPipeline(device_, pipeline_, nullptr);
  if (layout_)
    vkDestroyPipelineLayout(device_, layout_, nullptr);
  if (render_pass_)
    vkDestroyRenderPass(device_, render_pass_, nullptr);
  device_ = VK_NULL_HANDLE;
  pipeline_ = VK_NULL_HANDLE;
  layout_ = VK_NULL_HANDLE;
  render_pass_ = VK_NULL_HANDLE;
}

FullscreenPass::FullscreenPass(FullscreenPass&& o) noexcept {
  *this = std::move(o);
}

FullscreenPass& FullscreenPass::operator=(FullscreenPass&& o) noexcept {
  if (this != &o) {
    destroy();
    device_ = o.device_;
    render_pass_ = o.render_pass_;
    layout_ = o.layout_;
    pipeline_ = o.pipeline_;
    push_stages_ = o.push_stages_;
    push_size_ = o.push_size_;
    o.device_ = VK_NULL_HANDLE;
    o.render_pass_ = VK_NULL_HANDLE;
    o.layout_ = VK_NULL_HANDLE;
    o.pipeline_ = VK_NULL_HANDLE;
  }
  return *this;
}

Result<VkFramebuffer> FullscreenPass::make_framebuffer(VkImageView view,
                                                       uint32_t w,
                                                       uint32_t h) const {
  const VkFramebufferCreateInfo fci{
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass_,
      .attachmentCount = 1,
      .pAttachments = &view,
      .width = w,
      .height = h,
      .layers = 1,
  };
  VkFramebuffer fb = VK_NULL_HANDLE;
  WATER_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &fb),
              "vkCreateFramebuffer");
  return fb;
}

void FullscreenPass::destroy_framebuffer(VkFramebuffer fb) const noexcept {
  if (fb)
    vkDestroyFramebuffer(device_, fb, nullptr);
}

void FullscreenPass::record(VkCommandBuffer cmd,
                            VkFramebuffer fb,
                            VkExtent2D extent,
                            std::array<float, 4> clear,
                            std::span<const std::byte> push,
                            VkDescriptorSet set) const {
  VkClearValue cv{};
  cv.color = {{clear[0], clear[1], clear[2], clear[3]}};
  const VkRenderPassBeginInfo rpbi{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass_,
      .framebuffer = fb,
      .renderArea = {{0, 0}, extent},
      .clearValueCount = 1,
      .pClearValues = &cv,
  };
  vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

  const VkViewport viewport{0, 0, float(extent.width), float(extent.height),
                            0, 1};
  const VkRect2D scissor{{0, 0}, extent};
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);

  if (set != VK_NULL_HANDLE) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1,
                            &set, 0, nullptr);
  }
  if (!push.empty() && push_size_ > 0) {
    vkCmdPushConstants(cmd, layout_, push_stages_, 0, push_size_, push.data());
  }
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRenderPass(cmd);
}

}  // namespace water
