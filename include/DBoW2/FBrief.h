/**
 * File: FBrief.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: functions for BRIEF descriptors
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_F_BRIEF__
#define __D_T_F_BRIEF__

#include <bitset>
#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "DescriptorPolicy.h"

namespace DBoW2
{

    /** @brief BRIEF descriptor policy and compatibility helpers. */
    class FBrief : public BitsetDescriptorPolicy<256>
    {
      public:
        using BasePolicy = BitsetDescriptorPolicy<256>;
        using Descriptor = BasePolicy::Descriptor;
        using Scalar = BasePolicy::Scalar;
        static constexpr std::string_view kFamily = "brief";
        static constexpr std::uint32_t kDescriptorContractVersion = 1;
        static const int L = 256; // Descriptor length (in bits)
        typedef Descriptor TDescriptor;
        typedef const TDescriptor *pDescriptor;

        /**
         * Calculates the mean value of a set of descriptors
         * @param descriptors
         * @param mean mean descriptor
         */
        static void meanValue(const std::vector<pDescriptor> &descriptors,
                              TDescriptor &mean);

        /**
         * Calculates the distance between two descriptors
         * @param a
         * @param b
         * @return distance
         */
        static double distance(const TDescriptor &a, const TDescriptor &b);

        /**
         * Returns a string version of the descriptor
         * @param a descriptor
         * @return string version
         */
        static std::string toString(const TDescriptor &a);

        /**
         * Returns a descriptor from a string
         * @param a descriptor
         * @param s string version
         */
        static void fromString(TDescriptor &a, const std::string &s);

        /**
         * Returns a mat with the descriptors in float format
         * @param descriptors
         * @param mat (out) NxL 32F matrix
         */
        static void toMat32F(const std::vector<TDescriptor> &descriptors,
                             cv::Mat &mat);
    };

} // namespace DBoW2

#endif
