// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// water/result.hpp — error spine for the engine-slice (Tier A).
//
// std::expected<T, Error> propagation, no exceptions across the eventual FFI
// boundary (you don't throw across extern "C"). Vulkan calls return VkResult;
// engine-slice functions return Result<T> and propagate with
// WATER_TRY/WATER_CHECK.
#pragma once

#include <vulkan/vulkan.h>

#include <expected>
#include <string_view>
#include <utility>

namespace water {

// A failure: the VkResult (or VK_ERROR_UNKNOWN for logical errors) plus a short
// static string naming the call site. Cheap to copy; carries no allocation.
struct Error {
  VkResult code = VK_ERROR_UNKNOWN;
  std::string_view where = {};
};

template <class T>
using Result = std::expected<T, Error>;

[[nodiscard]] inline std::unexpected<Error> fail(VkResult code,
                                                 std::string_view where) {
  return std::unexpected(Error{code, where});
}

const char* to_string(VkResult r) noexcept;

}  // namespace water

// Propagate a Result<T>, yielding its value (GCC/Clang statement expression —
// both toolchains in scope per ). Use for value-returning steps: auto x =
// WATER_TRY(f());
#define WATER_TRY(expr)                         \
  ({                                            \
    auto&& _water_r = (expr);                   \
    if (!_water_r)                              \
      return std::unexpected(_water_r.error()); \
    std::move(*_water_r);                       \
  })

// Propagate a Result<void> with no value.
#define WATER_TRYV(expr)                        \
  do {                                          \
    auto&& _water_r = (expr);                   \
    if (!_water_r)                              \
      return std::unexpected(_water_r.error()); \
  } while (0)

// Turn a raw VkResult call into a propagated water::Error on non-success.
#define WATER_CHECK(call, where)               \
  do {                                         \
    VkResult _water_c = (call);                \
    if (_water_c != VK_SUCCESS)                \
      return ::water::fail(_water_c, (where)); \
  } while (0)
