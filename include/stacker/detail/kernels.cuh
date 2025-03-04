#pragma once

#include "binary_search.cuh"
#include "block_memmove.cuh"
#include "utils.cuh"
#include <cstdint>
#include <cub/cub.cuh>

template <std::uint32_t BlockThreads, std::uint32_t ItemsPerThread>
__global__ void memmove_simple(std::uint8_t* dest,
                               const std::uint8_t* src,
                               std::size_t num,
                               cub::ScanTileState<bool> scan_tile_state)
{
  static constexpr std::uint32_t tile_items = BlockThreads * ItemsPerThread;

  // Determine offset (global, since only a single src)
  const auto offset = static_cast<std::size_t>(blockIdx.x) * tile_items;

  // Invoke the MemMove block primitive
  block_memmove<BlockThreads, ItemsPerThread>(dest, src, num, offset, scan_tile_state);
}

template <std::uint32_t BlockThreads, std::uint32_t ItemsPerThread>
__global__ void memmove_complex(std::uint8_t* dest,
                                const std::uint32_t* block_starts,
                                const source_triple* src_triples,
                                std::int32_t num_sources,
                                cub::ScanTileState<bool> scan_tile_state)
{
  static constexpr std::uint32_t tile_items = BlockThreads * ItemsPerThread;

  __shared__ std::uint32_t block_start_smem;
  __shared__ source_triple src_triple_smem;

  // Do binary search to find src index (Separate kernel for binary search?)
  if (threadIdx.x == 0)
  {
    const auto idx   = binary_search(block_starts, blockIdx.x, 0, num_sources);
    block_start_smem = block_starts[idx];
    src_triple_smem  = src_triples[idx];
  }
  __syncthreads();

  // Determine the read offset for this src
  const auto read_offset = static_cast<std::size_t>(blockIdx.x - block_start_smem) * tile_items;

  // Invoke the MemMove block primitive
  block_memmove<BlockThreads, ItemsPerThread>(dest + src_triple_smem.write_offset,
                                              src_triple_smem.src,
                                              src_triple_smem.count,
                                              read_offset,
                                              scan_tile_state);
}

// Initializes state for decoupled look-back
template <typename ScanTileStateT>
__global__ void scan_tile_state_init(ScanTileStateT scan_tile_state, std::size_t num_thread_blocks)
{
  scan_tile_state.InitializeStatus(num_thread_blocks);
}