// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// SP-SIM — does compute beat fragment ping-pong for the heightfield
// update pass? Stands up both paths on a 256x256 RGBA16F field, runs N update
// steps each, brackets them with GPU timestamps, and prints a ranked comparison
// + go/kill verdict.
//
// Go/kill: if tiled compute is not materially faster (>=20%) than fragment on
// the bound (worst) target, drop the compute path and ship fragment-only
// (deletes half the // tier matrix). This binary is meant to run on
// Pi5/RK3588/SA8155P; locally it runs on RADV or (WATER_VK_DEVICE=llvmpipe)
// lavapipe to prove the harness.

#include "sp_sim_shaders.h"  // generated: fullscreen_vert_spv, sim_update_frag_spv, ...
#include "vkboot.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <vector>

using namespace vkboot;

namespace {

constexpr uint32_t kSize = 256;
constexpr uint32_t kIters = 1000;  // update steps per measurement
constexpr int kRepeats = 9;        // measurements; report the min (least noise)
constexpr VkFormat kFmt = VK_FORMAT_R16G16B16A16_SFLOAT;

struct Timer {
  VkQueryPool pool = VK_NULL_HANDLE;
  void create(Boot& b) {
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 2;
    VK_CHECK(vkCreateQueryPool(b.device, &qi, nullptr, &pool));
  }
  double read_ms(Boot& b) {
    std::array<uint64_t, 2> ts{};
    VK_CHECK(vkGetQueryPoolResults(
        b.device, pool, 0, 2, sizeof(ts), ts.data(), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    return double(ts[1] - ts[0]) * double(b.timestamp_period_ns) / 1.0e6;
  }
};

// Run a recorder (records kIters steps into a fresh command buffer, bracketed
// by timestamps) kRepeats times and return the minimum elapsed ms.
template <class Record>
double measure(Boot& b, Timer& t, Record&& record) {
  double best = 1e30;
  for (int rep = 0; rep < kRepeats + 1; ++rep) {  // +1 warmup, discarded
    VkCommandBuffer cb = b.begin_cmd();
    vkCmdResetQueryPool(cb, t.pool, 0, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, t.pool, 0);
    record(cb);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, t.pool, 1);
    b.submit_wait(cb);
    if (rep > 0)
      best = std::min(best, t.read_ms(b));
  }
  return best;
}

VkDescriptorSetLayout make_dsl(
    Boot& b,
    std::vector<VkDescriptorSetLayoutBinding> binds) {
  VkDescriptorSetLayoutCreateInfo ci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  ci.bindingCount = uint32_t(binds.size());
  ci.pBindings = binds.data();
  VkDescriptorSetLayout dsl;
  VK_CHECK(vkCreateDescriptorSetLayout(b.device, &ci, nullptr, &dsl));
  return dsl;
}

void clear_to_zero(Boot& b,
                   Image& img,
                   VkImageLayout final_layout,
                   VkAccessFlags final_access,
                   VkPipelineStageFlags final_stage) {
  VkCommandBuffer cb = b.begin_cmd();
  Boot::image_barrier(
      cb, img.image, VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
  VkClearColorValue zero{};
  VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdClearColorImage(cb, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       &zero, 1, &range);
  Boot::image_barrier(cb, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      final_layout, VK_ACCESS_TRANSFER_WRITE_BIT, final_access,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, final_stage);
  b.submit_wait(cb);
}

}  // namespace

int main() {
  Boot b;
  b.init();
  std::printf(
      "SP-SIM  device=\"%s\"  field=%ux%u  fmt=RGBA16F  iters=%u  repeats=%d\n",
      b.device_name.c_str(), kSize, kSize, kIters, kRepeats);
  if (!b.timestamps_valid || b.timestamp_period_ns == 0.0f) {
    std::printf(
        "  WARNING: queue reports no valid timestamp bits; timings "
        "unreliable\n");
  }

  Timer timer;
  timer.create(b);

  VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4},
  };
  VkDescriptorPoolCreateInfo dpci{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 4;
  dpci.poolSizeCount = 2;
  dpci.pPoolSizes = pool_sizes;
  VkDescriptorPool dpool;
  VK_CHECK(vkCreateDescriptorPool(b.device, &dpci, nullptr, &dpool));

  // ---- Fragment ping-pong path
  // ------------------------------------------------------
  Image frag[2] = {
      b.create_image(kSize, kSize, kFmt,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_SAMPLED_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT),
      b.create_image(kSize, kSize, kFmt,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_SAMPLED_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT),
  };
  for (auto& im : frag)
    clear_to_zero(b, im, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_ACCESS_SHADER_READ_BIT,
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

  VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
  sci.addressModeU = sci.addressModeV = sci.addressModeW =
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VkSampler sampler;
  VK_CHECK(vkCreateSampler(b.device, &sci, nullptr, &sampler));

  VkDescriptorSetLayout frag_dsl =
      make_dsl(b, {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                    VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}});

  VkDescriptorSet frag_set[2];
  for (int i = 0; i < 2; ++i) {
    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &frag_dsl;
    VK_CHECK(vkAllocateDescriptorSets(b.device, &ai, &frag_set[i]));
    VkDescriptorImageInfo dii{sampler, frag[i].view,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = frag_set[i];
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    vkUpdateDescriptorSets(b.device, 1, &w, 0, nullptr);
  }

  // Render pass: DONT_CARE load (we overwrite every pixel), SHADER_READ in/out
  // so the ping-pong source/dest layout flips for free; external deps order the
  // WAR/RAW hazard.
  VkAttachmentDescription att{};
  att.format = kFmt;
  att.samples = VK_SAMPLE_COUNT_1_BIT;
  att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  att.initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  att.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sub{};
  sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  sub.colorAttachmentCount = 1;
  sub.pColorAttachments = &ref;
  VkSubpassDependency deps[2]{};
  deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  deps[0].dstSubpass = 0;
  deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  deps[1].srcSubpass = 0;
  deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  rpci.attachmentCount = 1;
  rpci.pAttachments = &att;
  rpci.subpassCount = 1;
  rpci.pSubpasses = &sub;
  rpci.dependencyCount = 2;
  rpci.pDependencies = deps;
  VkRenderPass rp;
  VK_CHECK(vkCreateRenderPass(b.device, &rpci, nullptr, &rp));

  VkFramebuffer fb[2];
  for (int i = 0; i < 2; ++i) {
    VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fci.renderPass = rp;
    fci.attachmentCount = 1;
    fci.pAttachments = &frag[i].view;
    fci.width = kSize;
    fci.height = kSize;
    fci.layers = 1;
    VK_CHECK(vkCreateFramebuffer(b.device, &fci, nullptr, &fb[i]));
  }

  VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float) * 2};
  VkPipelineLayoutCreateInfo flci{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  flci.setLayoutCount = 1;
  flci.pSetLayouts = &frag_dsl;
  flci.pushConstantRangeCount = 1;
  flci.pPushConstantRanges = &pcr;
  VkPipelineLayout frag_pl;
  VK_CHECK(vkCreatePipelineLayout(b.device, &flci, nullptr, &frag_pl));

  VkShaderModule vs =
      b.load_shader(fullscreen_vert_spv, fullscreen_vert_spv_count);
  VkShaderModule fs =
      b.load_shader(sim_update_frag_spv, sim_update_frag_spv_count);
  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vs;
  stages[0].pName = "main";
  stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fs;
  stages[1].pName = "main";

  VkPipelineVertexInputStateCreateInfo vi{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo ia{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkViewport vp{0, 0, float(kSize), float(kSize), 0, 1};
  VkRect2D sc{{0, 0}, {kSize, kSize}};
  VkPipelineViewportStateCreateInfo vps{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vps.viewportCount = 1;
  vps.pViewports = &vp;
  vps.scissorCount = 1;
  vps.pScissors = &sc;
  VkPipelineRasterizationStateCreateInfo rs{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_NONE;
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineColorBlendAttachmentState cba{};
  cba.colorWriteMask = 0xf;
  VkPipelineColorBlendStateCreateInfo cb{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = 1;
  cb.pAttachments = &cba;
  VkGraphicsPipelineCreateInfo gpci{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gpci.stageCount = 2;
  gpci.pStages = stages;
  gpci.pVertexInputState = &vi;
  gpci.pInputAssemblyState = &ia;
  gpci.pViewportState = &vps;
  gpci.pRasterizationState = &rs;
  gpci.pMultisampleState = &ms;
  gpci.pColorBlendState = &cb;
  gpci.layout = frag_pl;
  gpci.renderPass = rp;
  VkPipeline frag_pipe;
  VK_CHECK(vkCreateGraphicsPipelines(b.device, VK_NULL_HANDLE, 1, &gpci,
                                     nullptr, &frag_pipe));

  const float delta[2] = {1.0f / kSize, 1.0f / kSize};
  auto record_frag = [&](VkCommandBuffer cmd) {
    for (uint32_t i = 0; i < kIters; ++i) {
      int src = i & 1;    // source image index
      int dst = 1 - src;  // framebuffer/dest index
      VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      rpbi.renderPass = rp;
      rpbi.framebuffer = fb[dst];
      rpbi.renderArea = sc;
      vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, frag_pipe);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, frag_pl, 0,
                              1, &frag_set[src], 0, nullptr);
      vkCmdPushConstants(cmd, frag_pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(delta), delta);
      vkCmdDraw(cmd, 3, 1, 0, 0);
      vkCmdEndRenderPass(cmd);
    }
  };

  // ---- Compute paths (naive + tiled)
  // ------------------------------------------------
  Image comp[2] = {
      b.create_image(
          kSize, kSize, kFmt,
          VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT),
      b.create_image(
          kSize, kSize, kFmt,
          VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT),
  };
  for (auto& im : comp)
    clear_to_zero(b, im, VK_IMAGE_LAYOUT_GENERAL,
                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  VkDescriptorSetLayout comp_dsl =
      make_dsl(b, {{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                    VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                   {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                    VK_SHADER_STAGE_COMPUTE_BIT, nullptr}});
  VkDescriptorSet
      comp_set[2];  // [0]=src comp0->dst comp1, [1]=src comp1->dst comp0
  for (int i = 0; i < 2; ++i) {
    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &comp_dsl;
    VK_CHECK(vkAllocateDescriptorSets(b.device, &ai, &comp_set[i]));
    VkDescriptorImageInfo si{VK_NULL_HANDLE, comp[i].view,
                             VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo di{VK_NULL_HANDLE, comp[1 - i].view,
                             VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2]{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = comp_set[i];
    w[0].dstBinding = 0;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[0].pImageInfo = &si;
    w[1] = w[0];
    w[1].dstBinding = 1;
    w[1].pImageInfo = &di;
    vkUpdateDescriptorSets(b.device, 2, w, 0, nullptr);
  }
  VkPipelineLayoutCreateInfo clci{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  clci.setLayoutCount = 1;
  clci.pSetLayouts = &comp_dsl;
  VkPipelineLayout comp_pl;
  VK_CHECK(vkCreatePipelineLayout(b.device, &clci, nullptr, &comp_pl));

  auto make_compute = [&](const uint32_t* code, uint32_t n) {
    VkShaderModule m = b.load_shader(code, n);
    VkComputePipelineCreateInfo ci{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = m;
    ci.stage.pName = "main";
    ci.layout = comp_pl;
    VkPipeline p;
    VK_CHECK(vkCreateComputePipelines(b.device, VK_NULL_HANDLE, 1, &ci, nullptr,
                                      &p));
    vkDestroyShaderModule(b.device, m, nullptr);
    return p;
  };
  VkPipeline comp_naive =
      make_compute(sim_update_naive_comp_spv, sim_update_naive_comp_spv_count);
  VkPipeline comp_tiled =
      make_compute(sim_update_tiled_comp_spv, sim_update_tiled_comp_spv_count);

  auto record_compute = [&](VkPipeline pipe) {
    return [&, pipe](VkCommandBuffer cmd) {
      const uint32_t groups = (kSize + 15) / 16;
      for (uint32_t i = 0; i < kIters; ++i) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, comp_pl, 0,
                                1, &comp_set[i & 1], 0, nullptr);
        vkCmdDispatch(cmd, groups, groups, 1);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
      }
    };
  };

  // ---- Measure
  // ----------------------------------------------------------------------
  double t_frag = measure(b, timer, record_frag);
  double t_naive = measure(b, timer, record_compute(comp_naive));
  double t_tiled = measure(b, timer, record_compute(comp_tiled));

  auto per = [](double total) { return total / kIters * 1000.0; };  // us/step
  std::printf(
      "\n  path                 total(ms)   per-step(us)   vs fragment\n");
  std::printf("  fragment ping-pong   %8.3f   %10.3f      1.00x (baseline)\n",
              t_frag, per(t_frag));
  std::printf("  compute naive        %8.3f   %10.3f   %7.2fx\n", t_naive,
              per(t_naive), t_frag / t_naive);
  std::printf("  compute tiled (LDS)  %8.3f   %10.3f   %7.2fx\n", t_tiled,
              per(t_tiled), t_frag / t_tiled);

  double speedup = t_frag / t_tiled;
  bool go = speedup >= 1.20;
  std::printf("\n  VERDICT: tiled compute is %.2fx fragment -> %s\n", speedup,
              go ? "GO (keep compute sim path in the tier matrix)"
                 : "KILL on this device (ship fragment-only here)");
  std::printf(
      "  (run on Pi5/RK3588/SA8155P for the binding decision; this is %s)\n",
      b.device_name.c_str());

  vkDeviceWaitIdle(b.device);
  return go ? 0 : 2;
}
