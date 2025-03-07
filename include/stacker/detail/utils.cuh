#pragma once

#include <cstdint>

#define RETURN_ERROR(e)                                                                            \
  do                                                                                               \
  {                                                                                                \
    cudaError_t Error = (e);                                                                       \
    if (Error != cudaSuccess)                                                                      \
      return Error;                                                                                \
  } while (0)
#define RETURN_ERROR_ALLOCATE(e)                                                                   \
  if (!(e))                                                                                        \
  return cudaErrorMemoryAllocation

struct source_pair
{
  const std::uint8_t* src;
  std::size_t count;
};

constexpr std::uint32_t src_triple_alignment = 32;
struct alignas(src_triple_alignment) source_triple
{
  const std::uint8_t* src;
  std::size_t count;
  std::size_t write_offset;
  std::size_t null;
};

// The scan op for memmove with decoupled look-back
struct memmove_scan_op
{
  __device__ __forceinline__ bool operator()(bool l, bool r) const
  {
    return l && r;
  }
};