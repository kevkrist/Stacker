# Use CPM to find or clone CCCL
function(find_and_configure_cccl)
    include(${rapids-cmake-dir}/cpm/cccl.cmake)
    rapids_cpm_cccl(INSTALL_EXPORT_SET stacker-exports BUILD_EXPORT_SET stacker-exports)
endfunction()

find_and_configure_cccl()