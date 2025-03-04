#pragma once

// The scan op for MemMove with decoupled look-back
struct memmove_scan_op
{
  __device__ __forceinline__ bool operator()(bool l, bool r) const
  {
    return l && r;
  }
};