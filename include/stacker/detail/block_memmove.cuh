#pragma once

#include "scan.cuh"
#include <cstdint>
#include <cub/cub.cuh>

template <std::uint32_t BlockThreads, std::uint32_t ItemsPerThread>
__device__ __forceinline__ void block_memmove(
  std::uint8_t* dest,
  const std::uint8_t* src,
  std::size_t num,
  std::size_t offset, // The offset for a particular src (not necessarily a global offset)
  cub::ScanTileState<bool> scan_tile_state)
{
  static constexpr std::uint32_t tile_items = BlockThreads * ItemsPerThread;
  using block_load_t =
    cub::BlockLoad<std::uint32_t, BlockThreads, ItemsPerThread, cub::BLOCK_LOAD_STRIPED>;
  using block_load_storage_t = typename block_load_t::TempStorage;
  using block_store_t =
    cub::BlockStore<std::uint32_t, BlockThreads, ItemsPerThread, cub::BLOCK_STORE_STRIPED>;
  using block_store_storage_t = typename block_store_t::TempStorage;
  using scan_op_t             = memmove_scan_op;
  using tile_prefix_op_t    = cub::TilePrefixCallbackOp<bool, scan_op_t, cub::ScanTileState<bool>>;
  using prefix_op_storage_t = typename tile_prefix_op_t::TempStorage;

  // Shared memory
  __shared__ block_load_storage_t load_storage;
  __shared__ prefix_op_storage_t prefix_storage;
  __shared__ block_store_storage_t store_storage;

  // Thread memory
  std::uint32_t thread_chunks[ItemsPerThread];
  auto dest_chunks             = reinterpret_cast<std::uint32_t*>(dest);
  const auto src_chunks        = reinterpret_cast<const std::uint32_t*>(src);
  const std::size_t num_chunks = num / sizeof(std::uint32_t);
  const auto num_tile_chunks = static_cast<std::int32_t>(CUB_MIN(num_chunks - offset, tile_items));
  const bool is_last_tile    = (offset + tile_items) >= num_chunks;
  const auto num_dangling_bytes = static_cast<std::int32_t>(num % sizeof(std::uint32_t));
  std::uint8_t dangling_byte    = 0;

  // Load the data
  if (is_last_tile)
  {
    block_load_t{load_storage}.Load(src_chunks + offset, thread_chunks, num_tile_chunks);

    // Handle dangling bytes
    if (threadIdx.x < num_dangling_bytes)
    {
      dangling_byte = src[num - num_dangling_bytes + threadIdx.x];
    }
  }
  else
  {
    block_load_t{load_storage}.Load(src_chunks + offset, thread_chunks);
  }
  __syncthreads(); // Ensure all threads have completed the load

  // Do decoupled look-back
  if (blockIdx.x == 0)
  {
    if (threadIdx.x == 0)
    {
      scan_tile_state.SetInclusive(blockIdx.x, true);
    }
  }
  else
  {
    // The first warp does the look-back
    if (threadIdx.x < CUB_PTX_WARP_THREADS)
    {
      // Initialize the prefix op
      tile_prefix_op_t prefix_op(scan_tile_state, prefix_storage, scan_op_t{});

      // Do the decoupled look-back
      prefix_op(true);
    }
  }
  __syncthreads();

  // Now we can store
  if (is_last_tile)
  {
    block_store_t{store_storage}.Store(dest_chunks + offset, thread_chunks, num_tile_chunks);

    // Handle dangling bytes
    if (threadIdx.x < num_dangling_bytes)
    {
      dest[num - num_dangling_bytes + threadIdx.x] = dangling_byte;
    }
  }
  else
  {
    block_store_t{store_storage}.Store(dest_chunks + offset, thread_chunks);
  }
}