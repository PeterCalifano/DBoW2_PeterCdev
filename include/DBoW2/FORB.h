/**
 * @file FORB.h
 * @brief ORB descriptor policy and backward-compatible helper declarations.
 * @author Dorian Galvez-Lopez and Pietro Califano
 * @date 2026-08-28
 * @copyright See LICENSE.txt.
 */

#ifndef __D_T_F_ORB__
#define __D_T_F_ORB__

#include <cstddef>
#include <cstdint>
#include <opencv2/core.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "DescriptorPolicy.h"

namespace DBoW2
{
    /** @brief ORB descriptor policy and compatibility helpers. */
    class FORB
    {
      public:
        using Descriptor = cv::Mat;
        using Scalar = std::uint8_t;
        typedef Descriptor TDescriptor;

        static constexpr std::size_t kElementCount = 256;
        static constexpr EDescriptorStorage kStorage = EDescriptorStorage::packed_binary;
        static constexpr EDistanceMetric kMetric = EDistanceMetric::hamming;
        static constexpr EDescriptorNormalization kNormalization = EDescriptorNormalization::none;
        static constexpr std::string_view kFamily = "orb";
        static constexpr std::uint32_t kDescriptorContractVersion = 1;

        /**
         * @brief Return an independently allocated ORB descriptor copy.
         * @param descriptor Valid 1 x 32 CV_8U descriptor row.
         * @return Deep copy that does not alias the caller's OpenCV allocation.
         * @throws std::invalid_argument If the descriptor shape or scalar type is invalid.
         */
        [[nodiscard]] static Descriptor Clone(const Descriptor &descriptor);

        /**
         * @brief Validate one ORB descriptor at a public policy boundary.
         * @param descriptor Descriptor required to be a 1 x 32 CV_8U row.
         * @throws std::invalid_argument If the shape or scalar type is invalid.
         */
        static void Validate(const Descriptor &descriptor);

        /// Pointer to a single descriptor
        typedef const TDescriptor *pDescriptor;
        /// Descriptor length (in bytes)
        static const int L = 32;

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
         * @throws std::invalid_argument If the descriptor shape or type is invalid.
         */
        static std::string toString(const TDescriptor &a);

        /**
         * Returns a descriptor from a string
         * @param a descriptor
         * @param s string version
         * @throws std::invalid_argument If s is not exactly one valid ORB descriptor.
         */
        static void fromString(TDescriptor &a, const std::string &s);

        /**
         * Returns a mat with the descriptors in float format
         * @param descriptors
         * @param mat (out) NxL 32F matrix
         * @throws std::invalid_argument If any descriptor is malformed.
         */
        static void toMat32F(const std::vector<TDescriptor> &descriptors,
                             cv::Mat &mat);

        /**
         * Returns a mat with the descriptors in float format
         * @param descriptors NxL CV_8U matrix
         * @param mat (out) NxL 32F matrix
         * @throws std::invalid_argument If the matrix type or column count is invalid.
         */
        static void toMat32F(const cv::Mat &descriptors, cv::Mat &mat);

        /**
         * Returns a matrix with the descriptor in OpenCV format
         * @param descriptors vector of N row descriptors
         * @param mat (out) NxL CV_8U matrix
         * @throws std::invalid_argument If any descriptor is malformed.
         */
        static void toMat8U(const std::vector<TDescriptor> &descriptors,
                            cv::Mat &mat);

        /** @brief Calculate the historical ORB bit-majority centroid. */
        [[nodiscard]] static Descriptor Mean(
            std::span<const Descriptor *const> descriptors);

        /** @brief Calculate Hamming distance between validated ORB descriptors. */
        [[nodiscard]] static double Distance(const Descriptor &first,
                                             const Descriptor &second);

        /** @brief Serialize one validated ORB descriptor as 32 byte values. */
        [[nodiscard]] static std::string Serialize(const Descriptor &descriptor);

        /** @brief Strictly parse exactly 32 ORB byte values without partial output. */
        [[nodiscard]] static bool Deserialize(std::string_view serialized,
                                              Descriptor &descriptor);

        /** @brief Convert validated ORB descriptors into 256 zero/one float columns. */
        [[nodiscard]] static cv::Mat ToMat32F(std::span<const Descriptor> descriptors);

    };

} // namespace DBoW2

#endif
