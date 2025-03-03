#=============================================================================
# Examples
#=============================================================================
cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureExample EXAMPLE_NAME EXAMPLE_SRC)
    add_executable(${EXAMPLE_NAME} "${EXAMPLE_SRC}")
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
                                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
    target_include_directories(${EXAMPLE_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${EXAMPLE_NAME} PRIVATE --compiler-options=-Wall --compiler-options=-Wextra
      --expt-extended-lambda -Xcompiler -Wno-subobject-linkage)
    target_link_libraries(${EXAMPLE_NAME} PRIVATE stacker CUDA::cudart)
endfunction(ConfigureExample)

###################################################################################################
### Example sources ###############################################################################
###################################################################################################

ConfigureExample(STACKER_DUMMY "${CMAKE_CURRENT_SOURCE_DIR}/stacker.cu")