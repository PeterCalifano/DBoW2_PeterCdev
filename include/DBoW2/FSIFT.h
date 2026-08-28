/**
 * @file FSIFT.h
 * @brief SIFT descriptor policy and compatibility helpers.
 * @copyright See LICENSE.txt.
 */

#pragma once

#include <array>
#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "DescriptorPolicy.h"

namespace DBoW2
{
    /** @brief SIFT descriptor policy and compatibility helpers. */
    class FSIFT : public FixedFloatDescriptorPolicy<128>
    {
      public:
        using BasePolicy = FixedFloatDescriptorPolicy<128>;
        using Descriptor = BasePolicy::Descriptor;
        using Scalar = BasePolicy::Scalar;
        using TDescriptor = Descriptor;
        using pDescriptor = const TDescriptor *;

        static constexpr std::string_view kFamily = "sift";
        static constexpr std::uint32_t kDescriptorContractVersion = 1;

        static constexpr int L = 128;

        /** @brief Calculate the arithmetic mean of validated SIFT descriptors. */
        static void meanValue(const std::vector<pDescriptor> &descriptors,
                              TDescriptor &mean);

        /** @brief Calculate squared L2 distance between two SIFT descriptors. */
        static double distance(const TDescriptor &a, const TDescriptor &b);

        /** @brief Serialize one SIFT descriptor using strict policy text. */
        static std::string toString(const TDescriptor &a);

        /** @brief Parse strict policy text, throwing for malformed input. */
        static void fromString(TDescriptor &a, const std::string &s);

        /** @brief Convert SIFT descriptors into a row-major float matrix. */
        static void toMat32F(const std::vector<TDescriptor> &descriptors,
                             cv::Mat &mat);
    };
} // namespace DBoW2
