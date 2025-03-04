#pragma once

#include <cstddef>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

namespace stacker
{

constexpr std::uint32_t default_alignment = 128; // Default alignment in bytes

template <std::uint32_t Alignment = default_alignment>
class stack_allocator
{
  static constexpr std::uint32_t alignment = Alignment;

public:
  //--------------------------------------------------
  // Constructors & Destructors
  //--------------------------------------------------
  /**
   * 
   * @param pool_size_requested 
   */
  explicit stack_allocator(std::size_t pool_size_requested)
      : pool_size(cuda::ceil_div(pool_size_requested, static_cast<std::size_t>(alignment)) *
                  alignment)
  {
    CubDebug(cudaMalloc(&reinterpret_cast<void*>(base_ptr), pool_size));
  }

  // Disable copy constructor
  stack_allocator(const stack_allocator&)            = delete;
  stack_allocator& operator=(const stack_allocator&) = delete;

  // Move constructors
  stack_allocator(stack_allocator&& other) noexcept
      : pool_size(other.pool_size)
      , base_ptr(other.base_ptr)
      , current_offset(other.current_offset)
  {
    other.base_ptr       = nullptr; // Prevent double free
    other.current_offset = 0;
    other.pool_size      = 0;
  }
  stack_allocator& operator=(stack_allocator&& other) noexcept
  {
    if (this != &other)
    {
      // Free any existing memory before taking ownership
      if (base_ptr != nullptr)
      {
        CubDebug(cudaFree(base_ptr));
      }

      // Transfer ownership
      base_ptr       = other.base_ptr;
      pool_size      = other.pool_size;
      current_offset = other.current_offset;

      // Invalidate the moved-from object
      other.base_ptr       = nullptr;
      other.pool_size      = 0;
      other.current_offset = 0;
    }
    return *this;
  }

  // Destructor
  ~stack_allocator()
  {
    if (base_ptr != nullptr)
    {
      CubDebug(cudaFree(base_ptr));
    }
  }

  /**
   *
   * @tparam T
   * @param num
   * @return
   */
  template <typename T = std::uint8_t>
  __host__ T* allocate(std::size_t num)
  {
    // Ensure there is sufficient space
    const std::size_t allocation_size =
      cuda::ceil_div(num * sizeof(T), static_cast<std::size_t>(alignment)) * alignment;
    if (current_offset + allocation_size > pool_size)
    {
      std::cerr << "Error: Allocate request exceeds memory pool capacity.\n";
      return nullptr;
    }

    // Increment current_offset
    auto return_ptr = reinterpret_cast<T*>(base_ptr + current_offset);
    current_offset += allocation_size;
    return return_ptr;
  }

  /**
   * 
   * @tparam T 
   * @param num 
   */
  template <typename T = std::uint8_t>
  void deallocate(std::size_t num)
  {
    // Conservatively round down the deallocation
    const std::size_t deallocation_size = ((sizeof(T) * num) / alignment) * alignment;

    // Ensure you don't over-deallocate
    if (current_offset < deallocation_size)
    {
      std::cerr << "Error. Deallocation request exceeds allocated memory.\n";
    }

    // Decrement current_offset (aligned)
    current_offset -= deallocation_size;
  }

  /**
   *
   */
  void reset()
  {
    current_offset = 0;
  }

  //--------------------------------------------------
  // Thrust utilities
  //--------------------------------------------------
  /**
   *
   * @tparam T
   * @param num_items
   * @return
   */
  template <typename T>
  thrust::device_ptr<T> make_thrust_device_ptr(std::size_t num_items)
  {
    return thrust::device_pointer_cast(reinterpret_cast<T*>(allocate(num_items * sizeof(T))));
  }

  /**
   *
   * @tparam T
   * @param num_items
   * @param value
   * @return
   */
  template <typename T>
  thrust::device_ptr<T> make_thrust_device_ptr(std::size_t num_items, const T& value)
  {
    // Make thrust device pointer
    thrust::device_ptr<T> dev_ptr = make_thrust_device_ptr<T>(num_items);

    // Fill the memory region
    thrust::fill_n(dev_ptr, num_items, value);
    return dev_ptr;
  }

  /**
   *
   * @tparam T
   * @param num_items
   * @return
   */
  template <typename T>
  thrust::device_ptr<T> make_thrust_device_vector(std::size_t num_items)
  {
    auto dev_ptr = make_thrust_device_ptr<T>(num_items);
    return thrust::device_vector<T>(dev_ptr, dev_ptr + num_items);
  }

  /**
   *
   * @tparam T
   * @param num_items
   * @param value
   * @return
   */
  template <typename T>
  thrust::device_ptr<T> make_thrust_device_vector(std::size_t num_items, const T& value)
  {
    auto dev_ptr = make_thrust_device_ptr<T>(num_items, value);
    return thrust::device_vector<T>(dev_ptr, dev_ptr + num_items);
  }

private:
  std::size_t pool_size; // In bytes
  std::uint8_t* base_ptr     = nullptr;
  std::size_t current_offset = 0;
};

} // namespace stacker