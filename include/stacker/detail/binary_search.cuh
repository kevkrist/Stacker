#pragma once

#include <cub/cub.cuh>

template <typename T, typename IndexT>
__device__ __forceinline__ IndexT
binary_search(const T* search_data, const T& search_key, IndexT start, IndexT end)
{
  IndexT idx;
  T current_key;

  while (start < end)
  {
    idx         = cub::MidPoint(start, end);
    current_key = search_data[idx];
    if (search_key < current_key)
    {
      end = idx;
    }
    else
    {
      start = idx + 1;
    }
  }
  return start;
}

  //// Do binary search to find src index
// if (threadIdx.x < num_sources)
//{
//   const auto idx   = binary_search(block_starts, blockIdx.x, 0, num_sources);
//   block_start_smem = block_starts[idx];
//   src_triple_smem  = src_triples[idx];
// }
//__syncthreads();