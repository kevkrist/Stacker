#pragma once

#include "detail/kernels.cuh"
#include "detail/utils.cuh"
#include <cstdint>
#include <cub/cub.cuh>
#include <memory>
#include <vector>

template <std::uint32_t Alignment, typename AllocatorT>
class memmove
{
  using scan_tile_state_t = cub::ScanTileState<bool>;
  using allocator_t       = AllocatorT;
  using chunk_t           = std::uint32_t;

  static constexpr std::uint32_t alignment = Alignment;

public:
  // Constructor
  explicit __host__ memmove(std::unique_ptr<allocator_t>&& allocator)
      : allocator_{std::move(allocator)}
  {}

  // Copying and moving disabled
  __host__ memmove(const memmove&)            = delete;
  __host__ memmove& operator=(const memmove&) = delete;
  __host__ memmove(memmove&&)                 = delete;
  __host__ memmove& operator=(memmove&&)      = delete;

  // Destructor
  ~memmove()
  {
    if (copy_stream_)
    {
      CubDebug(cudaStreamDestroy(copy_stream_));
    }
    if (init_stream_)
    {
      CubDebug(cudaStreamDestroy(init_stream_));
    }
    clean_up();
  }

  /**
   * Implements the C memmove API: https://en.cppreference.com/w/cpp/string/byte/memmove
   *
   * @tparam BlockThreads The number of threads in the thread block
   * @tparam ItemsPerThread The coarsening factor
   * @param dest The destination to which to move data
   * @param src The source from which to move data
   * @param count The number of bytes of source data to move
   * @param stream The stream in which to execute the kernel
   * @return cudaError_t
   */
  template <std::uint32_t BlockThreads, std::uint32_t ItemsPerThread>
  __host__ cudaError_t move(std::uint8_t* dest,
                            const std::uint8_t* src,
                            std::size_t count,
                            cudaStream_t stream = cudaStreamDefault)
  {
    // If there is no memory overlap, default to cudaMemcpy.
    if (dest + count < src)
    {
      return CubDebug(cudaMemcpyAsync(dest, src, count, cudaMemcpyDeviceToDevice, stream));
    }

    // We need the memmove kernel. Prepare decoupled look-back.
    num_total_thread_blocks_ = get_num_thread_blocks<BlockThreads>(count);
    RETURN_ERROR(init_scan_tile_state<BlockThreads>());

    // Synchronize stream
    RETURN_ERROR(CubDebug(cudaStreamSynchronize(init_stream_)));

    // Launch kernel
    memmove_simple<BlockThreads, ItemsPerThread>
      <<<num_total_thread_blocks_, BlockThreads, 0, stream>>>(dest, src, count, scan_tile_state_);
    return CubDebug(cudaGetLastError());
  }

  /**
   * 
   * @tparam BlockThreads 
   * @tparam ItemsPerThread 
   * @param dest 
   * @param src_pairs 
   * @param stream 
   * @return 
   */
  template <std::uint32_t BlockThreads, std::uint32_t ItemsPerThread>
  __host__ cudaError_t move(std::uint8_t* dest,
                            std::vector<source_pair>& src_pairs,
                            cudaStream_t stream = cudaStreamDefault)
  {
    scan_src_pairs<BlockThreads>(src_pairs, dest);
    RETURN_ERROR(copy_to_device());
    RETURN_ERROR(init_scan_tile_state<BlockThreads>());

    // Synchronize streams
    RETURN_ERROR(CubDebug(cudaStreamSynchronize(copy_stream_)));
    RETURN_ERROR(CubDebug(cudaStreamSynchronize(init_stream_)));

    // Launch kernel
    memmove_complex<BlockThreads, ItemsPerThread>
      <<<num_total_thread_blocks_, BlockThreads, 0, stream>>>(dest,
                                                              block_starts_device_,
                                                              src_triples_device_,
                                                              num_sources_,
                                                              scan_tile_state_);
    return CubDebug(cudaGetLastError());
  }

  __host__ std::unique_ptr<allocator_t> release_allocator()
  {
    if (!allocator_)
    {
      std::cerr << "error_: No allocator. Release failed.";
      return nullptr;
    }

    // Clean up before returning the allocator
    clean_up();
    return std::move(allocator_);
  }

  __host__ void clean_up()
  {
    // Erase allocated memory
    if (allocator_)
    {
      allocator_->deallocate(bytes_allocated_);
      bytes_allocated_ = 0;
    }

    // Reset device pointers
    block_starts_device_     = nullptr;
    src_triples_device_      = nullptr;
    scan_tile_state_storage_ = nullptr;
  }

private:
  std::vector<std::size_t> block_starts_host_{CUB_PTX_WARP_THREADS};
  std::vector<source_triple> src_triples_host_{CUB_PTX_WARP_THREADS};
  std::uint32_t* block_starts_device_     = nullptr;
  source_triple* src_triples_device_      = nullptr;
  std::uint32_t num_total_thread_blocks_  = 0;
  std::uint32_t num_sources_              = 0;
  std::unique_ptr<allocator_t> allocator_ = nullptr;
  std::size_t bytes_allocated_            = 0;
  cudaStream_t copy_stream_{};
  scan_tile_state_t scan_tile_state_{};
  std::uint8_t* scan_tile_state_storage_ = nullptr;
  cudaStream_t init_stream_{};

  template <std::uint32_t BlockThreads>
  static __host__ __forceinline__ std::uint32_t get_num_thread_blocks(std::size_t count)
  {
    return static_cast<std::uint32_t>(
      cuda::ceil_div(count / sizeof(chunk_t), static_cast<std::size_t>(BlockThreads)));
  }

  static __host__ __forceinline__ std::size_t get_aligned_count(std::size_t count)
  {
    return cuda::ceil_div(count, static_cast<std::size_t>(alignment)) * alignment;
  }

  template <typename T>
  __host__ __forceinline__ T* allocate(std::size_t num)
  {
    if (!allocator_)
    {
      std::cerr << "Error: No allocator available. Allocation failed.\n";
      return nullptr;
    }

    T* allocation = allocator_->template allocate<T>(num);
    if (!allocation)
    {
      // Error should be reported by allocator
      return nullptr;
    }

    bytes_allocated_ += sizeof(T) * num;
    return allocation;
  }

  template <std::uint32_t BlockThreads>
  __host__ __forceinline__ void scan_src_pairs(std::vector<source_pair>& src_pairs,
                                               std::uint8_t* dest)
  {
    // The current limit is arbitrary
    assert(src_pairs.size() <= CUB_PTX_WARP_THREADS && "Only 32 sources are currently supported.");

    // Get the number of sources
    num_sources_ = static_cast<std::uint32_t>(src_pairs.size());

    // Scan the src_pairs
    block_starts_host_[0] = 0;
    src_triples_host_[0]  = {src_pairs[0].src, src_pairs[0].count, 0};
    src_pairs[0].src      = dest;
    for (auto i = 1; i < num_sources_; ++i)
    {
      // Determine the write offset from dest for the current src
      const auto write_offset =
        get_aligned_count(src_pairs[i - 1].count) + src_triples_host_[i - 1].write_offset;

      // Determine the starting block idx for the current src
      block_starts_host_[i] =
        get_num_thread_blocks<BlockThreads>(src_pairs[i - 1].count) + block_starts_host_[i - 1];
      src_triples_host_[i] = {src_pairs[i].src, src_pairs[i].count, write_offset};

      // Update with new src
      src_pairs[i].src = dest + write_offset;
    }

    // Complete the scan to determine the total number of required thread blocks
    num_total_thread_blocks_ =
      block_starts_host_[num_sources_ - 1] +
      get_num_thread_blocks<BlockThreads>(src_pairs[num_sources_ - 1].count);
  }

  __host__ __forceinline__ cudaError_t copy_to_device()
  {
    // Initialize the copy stream
    RETURN_ERROR(CubDebug(cudaStreamCreate(&copy_stream_)));

    // Allocate block starts and copy
    block_starts_device_ = allocate<std::uint32_t>(num_sources_);
    RETURN_ERROR_ALLOCATE(block_starts_device_);
    RETURN_ERROR(CubDebug(cudaMemcpyAsync(block_starts_device_,
                                          block_starts_host_.data(),
                                          sizeof(chunk_t) * num_sources_,
                                          cudaMemcpyHostToDevice,
                                          copy_stream_)));

    // Allocate source triples and copy
    src_triples_device_ = allocate<source_triple>(num_sources_);
    RETURN_ERROR_ALLOCATE(src_triples_device_);
    return CubDebug(cudaMemcpyAsync(src_triples_device_,
                                    src_triples_host_.data(),
                                    sizeof(source_triple) * num_sources_,
                                    cudaMemcpyHostToDevice,
                                    copy_stream_));
  }

  template <std::uint32_t BlockThreads>
  __host__ __forceinline__ cudaError_t init_scan_tile_state()
  {
    // Determine temporary storage requirements and allocate
    std::size_t scan_tile_state_storage_bytes = 0;
    scan_tile_state_t::AllocationSize(num_total_thread_blocks_, scan_tile_state_storage_bytes);
    scan_tile_state_storage_ = allocate<std::uint8_t>(scan_tile_state_storage_bytes);
    RETURN_ERROR_ALLOCATE(scan_tile_state_storage_);

    // Initialize the temporary storage
    RETURN_ERROR(CubDebug(scan_tile_state_.Init(num_total_thread_blocks_,
                                                scan_tile_state_storage_,
                                                scan_tile_state_storage_bytes)));

    // Initialize the init stream
    RETURN_ERROR(CubDebug(cudaStreamCreate(&init_stream_)));

    // Invoke the initialization kernel
    scan_tile_state_init()<<<cuda::ceil_div(num_total_thread_blocks_, BlockThreads),
                             BlockThreads,
                             0,
                             init_stream_>>>(scan_tile_state_, num_total_thread_blocks_);
    return CubDebug(cudaGetLastError());
  }
};