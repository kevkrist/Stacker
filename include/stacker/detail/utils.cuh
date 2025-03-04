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

struct source_triple
{
  const std::uint8_t* src;
  std::size_t count;
  std::size_t write_offset;
};