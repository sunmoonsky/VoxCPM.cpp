#pragma once

#include <catch2/catch_test_macros.hpp>
#include <filesystem>

#include "test_config.h"

#define SKIP_IF_NO_MODEL()                                                   \
    do {                                                                     \
        const std::string _model_path = ::voxcpm::test::get_model_path();   \
        if (!std::filesystem::exists(_model_path)) {                         \
            WARN("Model file not found, skipping test: " << _model_path);   \
            return;                                                          \
        }                                                                    \
    } while (false)

#define SKIP_IF_NO_TRACE(trace_name)                                          \
    do {                                                                      \
        const std::string _trace_path =                                       \
            ::voxcpm::test::get_trace_path(trace_name);                       \
        if (!std::filesystem::exists(_trace_path)) {                          \
            WARN("Trace file not found, skipping test: " << _trace_path);     \
            return;                                                           \
        }                                                                     \
    } while (false)
