#pragma once

#include "detail/Kernels.cuh"
#include "detail/Scan.cuh"
#include "detail/Utils.cuh"
#include <cstdint>
#include <cub/cub.cuh>
#include <memory>
#include <vector>

template <std::size_t Alignment, typename ByteAllocatorT>
class MemMove
{
  using scan_tile_state_t = cub::ScanTileState<bool>;
  using byte_allocator_t  = ByteAllocatorT;
  using chunk_t           = std::uint32_t;

  static constexpr std::int32_t alignment = Alignment;

  // Constructor
  explicit __host__ MemMove(std::unique_ptr<byte_allocator_t> byteAllocator)
      : ByteAllocator{std::move(byteAllocator)}
  {}

  // Destructor
  ~MemMove()
  {
    if (CopyStream)
    {
      Error = CubDebug(cudaStreamDestroy(CopyStream));
      if (Error != cudaSuccess)
      {
        CopyStream = nullptr;
      }
    }
    CleanUp();
  }

  template <std::uint32_t BLOCK_THREADS, std::uint32_t ITEMS_PER_THREAD>
  __host__ cudaError_t Move(std::uint8_t* dest,
                            const std::uint8_t* src,
                            std::size_t count,
                            cudaStream_t stream = cudaStreamDefault)
  {
    // If there is no memory overlap, default to cudaMemcpy
    if (dest + count < src)
    {
      return CubDebug(cudaMemcpyAsync(dest, src, count, cudaMemcpyDeviceToDevice, stream));
    }

    // We need the memmove kernel. Prepare decoupled look-back.
    NumTotalThreadBlocks = GetNumThreadBlocks<BLOCK_THREADS>(count);
    RETURN_ERROR(InitScanTileState<BLOCK_THREADS>());

    // Synchronize stream
    RETURN_ERROR(CubDebug(cudaStreamSynchronize(InitStream)));

    // Launch kernel
    MemMoveSimple<BLOCK_THREADS, ITEMS_PER_THREAD>
      <<<NumTotalThreadBlocks, BLOCK_THREADS, 0, stream>>>(dest, src, count, ScanTileState);

    return CubDebug(cudaGetLastError());
  }

  template <std::uint32_t BLOCK_THREADS, std::uint32_t ITEMS_PER_THREAD>
  __host__ cudaError_t Move(std::uint8_t* dest,
                            std::vector<SourcePair>& srcPairs,
                            cudaStream_t stream = cudaStreamDefault)
  {
    ScanSrcPairs<BLOCK_THREADS>(srcPairs, dest);
    RETURN_ERROR(CopyToDevice<BLOCK_THREADS>());
    RETURN_ERROR(InitScanTileState<BLOCK_THREADS>());

    // Synchronize streams
    RETURN_ERROR(CubDebug(cudaStreamSynchronize(CopyStream)));
    RETURN_ERROR(CubDebug(cudaStreamSynchronize(InitStream)));

    // Launch kernel
    MemMoveComplex<BLOCK_THREADS, ITEMS_PER_THREAD>
      <<<NumTotalThreadBlocks, BLOCK_THREADS, 0, stream>>>(dest,
                                                           BlockStartsDevice,
                                                           SrcTriplesDevice,
                                                           NumSources,
                                                           ScanTileState);

    return CubDebug(cudaGetLastError());
  }

  __host__ std::unique_ptr<byte_allocator_t> ReleaseAllocator()
  {
    if (!ByteAllocator)
    {
      std::cerr << "Error: No allocator. Release failed.";
      return nullptr;
    }

    // Clean up before returning the allocator
    CleanUp();
    return std::move(ByteAllocator);
  }

  __host__ void CleanUp()
  {
    // Erase allocated memory, if necessary
    if (ByteAllocator)
    {
      ByteAllocator->deallocate(BytesAllocated);
      BytesAllocated = 0;
    }

    // Reset device pointers
    BlockStartsDevice    = nullptr;
    SrcTriplesDevice     = nullptr;
    ScanTileStateStorage = nullptr;
  }

private:
  std::vector<std::size_t> BlockStartsHost{CUB_PTX_WARP_THREADS};
  std::vector<SourceTriple> SrcTriplesHost{CUB_PTX_WARP_THREADS};
  std::uint32_t* BlockStartsDevice   = nullptr;
  SourceTriple* SrcTriplesDevice     = nullptr;
  std::uint32_t NumTotalThreadBlocks = 0;
  std::size_t NumSources             = 0;
  std::unique_ptr<byte_allocator_t> ByteAllocator;
  std::size_t BytesAllocated = 0;
  cudaStream_t CopyStream{};
  scan_tile_state_t ScanTileState{};
  std::uint8_t* ScanTileStateStorage = nullptr;
  cudaStream_t InitStream{};
  cudaError_t Error{};

  template <std::uint32_t BLOCK_THREADS>
  static __host__ __forceinline__ std::uint32_t GetNumThreadBlocks(std::size_t count)
  {
    return static_cast<std::uint32_t>(cuda::ceil_div((count / sizeof(chunk_t)), BLOCK_THREADS));
  }

  static __host__ __forceinline__ std::size_t GetAlignedCount(std::size_t count)
  {
    return cuda::ceil_div(count, Alignment) * Alignment;
  }

  template <typename T>
  __host__ __forceinline__ T* Allocate(std::size_t num)
  {
    if (!ByteAllocator)
    {
      std::cerr << "Error: No allocator available. Allocation failed.\n";
      return nullptr;
    }

    std::uint8_t* placeholder = ByteAllocator->allocate(sizeof(T) * num);
    if (!placeholder)
      return nullptr;

    BytesAllocated += sizeof(T) * num;
    return reinterpret_cast<T*>(placeholder);
  }

  template <std::uint32_t BLOCK_THREADS>
  __host__ __forceinline__ void ScanSrcPairs(std::vector<SourcePair>& srcPairs, std::uint8_t* dest)
  {
    // Get the number of sources
    // NOTE: there is nothing inherent to the algorithm that imposes this limit on the number of
    // source pairs, but it makes things simpler by imposing a very reasonable upper bound.
    assert(srcPairs.size() <= CUB_PTX_WARP_THREADS && "Only 32 sources are currently supported.");
    NumSources = static_cast<std::int32_t>(srcPairs.size());

    // Scan the source pairs to determine the write offsets, starting block indices, sources, and
    // update the srcPairs with the new src locations
    BlockStartsHost[0] = 0;
    SrcTriplesHost[0]  = {srcPairs[0].Src, srcPairs[0].Count, 0};
    srcPairs[0].Src    = dest;
    for (auto i = 1; i < NumSources; ++i)
    {
      const auto writeOffset =
        GetAlignedCount(srcPairs[i - 1].Count) + SrcTriplesHost[i - 1].WriteOffset;
      BlockStartsHost[i] =
        GetNumThreadBlocks<BLOCK_THREADS>(srcPairs[i - 1].Count) + BlockStartsHost[i - 1];
      SrcTriplesHost[i] = {srcPairs[i].Src, srcPairs[i].Count, writeOffset};
      srcPairs[i].Src   = dest + writeOffset;
    }
    NumTotalThreadBlocks = BlockStartsHost[NumSources - 1] +
                           GetNumThreadBlocks<BLOCK_THREADS>(srcPairs[NumSources - 1].Count);
  }

  template <std::uint32_t BLOCK_THREADS>
  __host__ __forceinline__ cudaError_t CopyToDevice()
  {
    // Initialize the copy stream
    RETURN_ERROR(CubDebug(cudaStreamCreate(&CopyStream)));

    // Block starts
    RETURN_ERROR_ALLOCATE(BlockStartsDevice = Allocate<std::uint32_t>(NumSources));
    RETURN_ERROR(CubDebug(cudaMemcpyAsync(BlockStartsDevice,
                                          BlockStartsHost.data(),
                                          sizeof(std::uint32_t) * NumSources,
                                          cudaMemcpyHostToDevice,
                                          CopyStream)));

    // Source triples
    RETURN_ERROR_ALLOCATE(SrcTriplesDevice = Allocate<SourceTriple>(NumSources));
    return CubDebug(cudaMemcpyAsync(SrcTriplesDevice,
                                    SrcTriplesHost.data(),
                                    sizeof(SourceTriple) * NumSources,
                                    cudaMemcpyHostToDevice,
                                    CopyStream));
  }

  template <std::uint32_t BLOCK_THREADS>
  __host__ __forceinline__ cudaError_t InitScanTileState()
  {
    // Determine temporary storage requirements and allocate
    std::size_t scanTileStateStorageBytes = 0;
    scan_tile_state_t::AllocationSize(NumTotalThreadBlocks, scanTileStateStorageBytes);
    RETURN_ERROR_ALLOCATE(ScanTileStateStorage = Allocate(scanTileStateStorageBytes));

    // Initialize the temporary storage
    RETURN_ERROR(CubDebug(
      ScanTileState.Init(NumTotalThreadBlocks, ScanTileStateStorage, scanTileStateStorageBytes)));

    // Initialize the init stream
    RETURN_ERROR(CubDebug(cudaStreamCreate(&InitStream)));

    // Invoke the initialization kernel
    ScanTileStateInit<<<cuda::ceil_div(NumTotalThreadBlocks, BLOCK_THREADS),
                        BLOCK_THREADS,
                        0,
                        InitStream>>>(ScanTileState, NumTotalThreadBlocks);

    return CubDebug(cudaGetLastError());
  }
};