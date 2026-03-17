# Tests CMakeLists.txt

# Use Catch2 for testing
include(FetchContent)
FetchContent_Declare(
    Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.5.0.tar.gz
)
FetchContent_MakeAvailable(Catch2)

set(VOXCPM_DEFAULT_MODEL_PATH "${CMAKE_SOURCE_DIR}/models/voxcpm1.5.gguf" CACHE STRING "Default test model path")
set(VOXCPM_DEFAULT_TRACE_DIR "${CMAKE_SOURCE_DIR}/tests/fixtures/trace" CACHE STRING "Default trace directory")

function(voxcpm_add_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE voxcpm Catch2::Catch2WithMain)
    target_compile_definitions(${target} PRIVATE
        VOXCPM_DEFAULT_MODEL_PATH="${VOXCPM_DEFAULT_MODEL_PATH}"
        VOXCPM_DEFAULT_TRACE_DIR="${VOXCPM_DEFAULT_TRACE_DIR}"
    )
    add_test(NAME ${target} COMMAND ${target})
endfunction()

# FSQ tests
voxcpm_add_test(test_audio_vae test_audio_vae.cpp)
voxcpm_add_test(test_fsq test_fsq.cpp)

# Components tests
voxcpm_add_test(test_components test_components.cpp)

# MiniCPM tests
voxcpm_add_test(test_minicpm test_minicpm.cpp)

# LocEnc tests
voxcpm_add_test(test_localenc test_localenc.cpp)

# LocDiT tests
voxcpm_add_test(test_locdit test_locdit.cpp)

# UnifiedCFM tests
voxcpm_add_test(test_unified_cfm test_unified_cfm.cpp)

# VoxCPM prefill/decode tests
voxcpm_add_test(test_voxcpm test_voxcpm.cpp)

# Tokenizer tests
voxcpm_add_test(test_tokenizer test_tokenizer.cpp)

# Quantization tests
voxcpm_add_test(test_quantize test_quantize.cpp)

# imatrix tests
voxcpm_add_test(test_imatrix test_imatrix.cpp)
