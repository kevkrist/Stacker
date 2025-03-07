#pragma once

#include "block_memmove.cuh"
#include "utils.cuh"
#include <cooperative_groups.h>
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
  namespace cg                              = cooperative_groups;
  static constexpr std::uint32_t tile_items = BlockThreads * ItemsPerThread;

  // SMEM
  __shared__ source_triple src_triple_smem;
  __shared__ std::size_t read_offset_smem;

  // Find the source triple for this block
  if (threadIdx.x < num_sources)
  {
    auto grp = cg::coalesced_threads();

    // Coalesced load of all block starts
    std::uint32_t block_start = block_starts[threadIdx.x];

    // Shuffle block starts down to get bounding boxes for source triples
    std::uint32_t next_block_start = grp.shfl_down(block_start, 1);
    if (grp.thread_rank() == grp.num_threads() - 1)
    {
      next_block_start = gridDim.x;
    }

    // For matching bounding box, load the source triple
    if (block_start <= blockIdx.x && blockIdx.x < next_block_start)
    {
      auto local_src_triple = static_cast<source_triple>(*reinterpret_cast<const ulonglong4*>(
        __builtin_assume_aligned(src_triples + threadIdx.x, src_triple_alignment)));
      src_triple_smem       = local_src_triple;
      read_offset_smem      = static_cast<std::size_t>(blockIdx.x - block_start) * tile_items;
    }
  }
  __syncthreads();

  // Invoke the MemMove block primitive
  block_memmove<BlockThreads, ItemsPerThread>(dest + src_triple_smem.write_offset,
                                              src_triple_smem.src,
                                              src_triple_smem.count,
                                              read_offset_smem,
                                              scan_tile_state);
}

// Initializes state for decoupled look-back
template <typename ScanTileStateT>
__global__ void scan_tile_state_init(ScanTileStateT scan_tile_state, std::size_t num_thread_blocks)
{
  scan_tile_state.InitializeStatus(num_thread_blocks);
}