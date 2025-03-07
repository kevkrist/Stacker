#=============================================================================
# Stacker (this is essentially lifted from NVIDIA's CuCollections library)
#=============================================================================
cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/STACKER_RAPIDS.cmake)
    file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-25.02/RAPIDS.cmake
         ${CMAKE_CURRENT_BINARY_DIR}/STACKER_RAPIDS.cmake)
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/STACKER_RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# * Determine GPU architectures
# * Enable the CMake CUDA language
rapids_cuda_init_architectures(STACKER)

project(STACKER VERSION 0.0.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

set(default_build_option_state OFF)
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
  set(default_build_option_state ON)
endif()

option(BUILD_TESTS "Configure CMake to build tests" ${default_build_option_state})
# option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" ${default_build_option_state})
option(BUILD_EXAMPLES "Configure CMake to build examples" ${default_build_option_state})
option(INSTALL_STACKER "Enable CMake install rules for stacker" ${default_build_option_state})

# Write the version header
rapids_cmake_write_version_file(include/stacker/version_config.hpp)

##############################################################################
# - build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# needed for clangd and clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################################
# - compiler options ---------------------------------------------------------

# * Find CUDAToolkit package
# * Offers support for CMAKE_CUDA_ARCHITECTURES=NATIVE
rapids_find_package(
    CUDAToolkit REQUIRED
    BUILD_EXPORT_SET stacker-exports
    INSTALL_EXPORT_SET stacker-exports
)

###################################################################################################
# - find packages we depend on --------------------------------------------------------------------

rapids_cpm_init()

include(cmake/thirdparty/get_cccl.cmake)

###################################################################################################
# - stacker target   -------------------------------------------------------------------------------

add_library(stacker INTERFACE)
add_library(stacker::stacker ALIAS stacker)
target_include_directories(stacker INTERFACE
    INTERFACE   $<BUILD_INTERFACE:${STACKER_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>)
target_link_libraries(stacker INTERFACE CCCL::CCCL CUDA::toolkit)
target_compile_features(stacker INTERFACE cxx_std_17 cuda_std_17)

###################################################################################################
# - optionally build tests ------------------------------------------------------------------------

if(BUILD_TESTS)
    add_subdirectory(tests)
endif(BUILD_TESTS)

###################################################################################################
# - Optionally build google benchmarks ------------------------------------------------------------

#if(BUILD_BENCHMARKS)
#    add_subdirectory(benchmarks)
#endif(BUILD_BENCHMARKS)

###################################################################################################
# - Optionally build examples ---------------------------------------------------------------------

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif(BUILD_EXAMPLES)

###################################################################################################
# - Install targets -------------------------------------------------------------------------------

install(TARGETS stacker EXPORT stacker-exports)

set(doc_string
    [=[
Provide targets for Stacker...

Stacker...

]=])

# build directory stacker-config generation
rapids_export(
    BUILD stacker
    EXPORT_SET stacker-exports
    GLOBAL_TARGETS stacker
    NAMESPACE stacker::
    DOCUMENTATION doc_string)

if(INSTALL_STACKER)
    install(DIRECTORY include/stacker/ DESTINATION include/stacker)
    install(FILES ${STACKER_BINARY_DIR}/include/stacker/version_config.hpp DESTINATION include/stacker)
    # install directory stacker-config generation
    rapids_export(
        INSTALL stacker
        EXPORT_SET stacker-exports
        GLOBAL_TARGETS stacker
        NAMESPACE stacker::
        DOCUMENTATION doc_string)
endif()