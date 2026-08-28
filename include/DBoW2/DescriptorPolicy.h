/**
 * @file DescriptorPolicy.h
 * @brief C++20 descriptor contracts and fixed-float policy utilities.
 * @author Dorian Galvez-Lopez and Pietro Califano
 * @date 2026-08-28
 * @copyright See LICENSE.txt.
 */

#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <bitset>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <locale>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace DBoW2
{
    /** @brief Physical representation used by a descriptor policy. */
    enum class EDescriptorStorage
    {
        packed_binary, ///< Packed bits or bytes with a fixed logical bit count.
        fixed_float,   ///< Compile-time fixed-size float storage.
        dynamic_float  ///< Runtime container constrained to a fixed logical length.
    };

    /** @brief Distance function used by vocabulary clustering and lookup. */
    enum class EDistanceMetric
    {
        hamming,   ///< Number of differing logical bits.
        squared_l2 ///< Sum of squared element differences.
    };

    /** @brief Normalization required at a descriptor policy boundary. */
    enum class EDescriptorNormalization
    {
        none, ///< No deployment-input normalization requirement.
        l2    ///< Unit L2 norm required at the deployment-input boundary.
    };

    /**
     * @brief Static contract required by DBoW2 vocabularies and databases.
     *
     * A policy provides compile-time representation metadata, a default/copyable descriptor,
     * an independently owning clone, centroid and distance operations, and strict text
     * persistence. The explicit clone prevents shallow-copying view-backed descriptors.
     *
     * @tparam TPolicy Candidate descriptor policy.
     */
    template <typename TPolicy>
    concept DescriptorPolicy = requires {
        typename std::integral_constant<std::size_t, TPolicy::kElementCount>;
        typename std::integral_constant<EDescriptorStorage, TPolicy::kStorage>;
        typename std::integral_constant<EDistanceMetric, TPolicy::kMetric>;
        typename std::integral_constant<
            EDescriptorNormalization, TPolicy::kNormalization>;
    } && (TPolicy::kElementCount > 0) &&
        std::default_initializable<typename TPolicy::Descriptor> &&
        std::copyable<typename TPolicy::Descriptor> &&
        requires(const typename TPolicy::Descriptor &descriptor,
                 typename TPolicy::Descriptor &output,
                 std::span<const typename TPolicy::Descriptor *const> descriptors,
                 std::string_view serialized) {
        typename TPolicy::Descriptor;
        typename TPolicy::Scalar;
        { TPolicy::Validate(descriptor) } -> std::same_as<void>;
        { TPolicy::Clone(descriptor) } -> std::same_as<typename TPolicy::Descriptor>;
        { TPolicy::Mean(descriptors) } -> std::same_as<typename TPolicy::Descriptor>;
        { TPolicy::Distance(descriptor, descriptor) } -> std::convertible_to<double>;
        { TPolicy::Serialize(descriptor) } -> std::same_as<std::string>;
        { TPolicy::Deserialize(serialized, output) } -> std::same_as<bool>;
    };

    /**
     * @brief Descriptor contract that also supports classical local-feature matching.
     *
     * @tparam TPolicy Candidate local descriptor policy.
     */
    template <typename TPolicy>
    concept LocalDescriptorPolicy = DescriptorPolicy<TPolicy> &&
        requires(std::span<const typename TPolicy::Descriptor> descriptors) {
        { TPolicy::ToMat32F(descriptors) } -> std::same_as<cv::Mat>;
    };

    /** @brief Convenience alias for the descriptor associated with a policy. */
    template <DescriptorPolicy TPolicy>
    using DescriptorType = typename TPolicy::Descriptor;

    /**
     * @brief Fixed-size bitset descriptor operations using Hamming distance.
     * @tparam N Number of logical descriptor bits.
     */
    template <std::size_t N>
    class BitsetDescriptorPolicy
    {
      public:
        using Scalar = bool;
        using Descriptor = std::bitset<N>;

        static constexpr std::size_t kElementCount = N;
        static constexpr EDescriptorStorage kStorage = EDescriptorStorage::packed_binary;
        static constexpr EDistanceMetric kMetric = EDistanceMetric::hamming;
        static constexpr EDescriptorNormalization kNormalization = EDescriptorNormalization::none;

        /** @brief Validate a bitset descriptor; its type already enforces the full contract. */
        static void Validate(const Descriptor &) {}

        /**
         * @brief Return an independently owned copy of one descriptor.
         * @param descriptor Descriptor value to copy.
         * @return Copy with no caller-owned backing storage.
         */
        [[nodiscard]] static Descriptor Clone(const Descriptor &descriptor)
        {
            Validate(descriptor);
            return descriptor;
        }

        /**
         * @brief Calculate the strict bitwise majority descriptor.
         * @param descriptors Non-empty span of non-null descriptor pointers.
         * @return Descriptor containing bits set in more than half of the inputs.
         * @throws std::invalid_argument If the set is empty or contains a null pointer.
         */
        [[nodiscard]] static Descriptor Mean(
            const std::span<const Descriptor *const> descriptors)
        {
            if (descriptors.empty())
            {
                throw std::invalid_argument("Cannot calculate a centroid from an empty descriptor set.");
            }

            std::array<std::size_t, N> counts{};
            for (const Descriptor *descriptor : descriptors)
            {
                if (descriptor == nullptr)
                {
                    throw std::invalid_argument("Descriptor centroid input contains a null pointer.");
                }
                for (std::size_t index = 0; index < N; ++index)
                {
                    counts[index] += descriptor->test(index) ? 1U : 0U;
                }
            }

            Descriptor mean;
            const std::size_t half_count = descriptors.size() / 2U;
            for (std::size_t index = 0; index < N; ++index)
            {
                mean.set(index, counts[index] > half_count);
            }
            return mean;
        }

        /** @brief Return the Hamming distance between two bitset descriptors. */
        [[nodiscard]] static double Distance(const Descriptor &first, const Descriptor &second)
        {
            return static_cast<double>((first ^ second).count());
        }

        /** @brief Serialize a bitset descriptor as exactly N binary characters. */
        [[nodiscard]] static std::string Serialize(const Descriptor &descriptor)
        {
            return descriptor.to_string();
        }

        /**
         * @brief Parse exactly N binary characters without partial output.
         * @param serialized Binary descriptor text.
         * @param descriptor Output descriptor, modified only on success.
         * @return True when the input contains exactly N zero/one characters.
         */
        [[nodiscard]] static bool Deserialize(const std::string_view serialized,
                                              Descriptor &descriptor)
        {
            if (serialized.size() != N)
            {
                return false;
            }
            for (const char character : serialized)
            {
                if (character != '0' && character != '1')
                {
                    return false;
                }
            }

            descriptor = Descriptor(std::string(serialized));
            return true;
        }

        /** @brief Convert bitset descriptors to rows of zero/one float values. */
        [[nodiscard]] static cv::Mat ToMat32F(const std::span<const Descriptor> descriptors)
        {
            if (descriptors.empty())
            {
                return {};
            }
            if (descriptors.size() >
                static_cast<std::size_t>((std::numeric_limits<int>::max)()))
            {
                throw std::length_error("Descriptor count exceeds the OpenCV row range.");
            }

            cv::Mat matrix(static_cast<int>(descriptors.size()), static_cast<int>(N), CV_32F);
            for (std::size_t row = 0; row < descriptors.size(); ++row)
            {
                float *output = matrix.ptr<float>(static_cast<int>(row));
                for (std::size_t column = 0; column < N; ++column)
                {
                    output[column] = descriptors[row].test(column) ? 1.0F : 0.0F;
                }
            }
            return matrix;
        }
    };

    /**
     * @brief Fixed-length operations for legacy dynamically allocated float descriptors.
     * @tparam N Required number of float elements.
     */
    template <std::size_t N>
    class VectorFloatDescriptorPolicy
    {
      public:
        using Scalar = float;
        using Descriptor = std::vector<Scalar>;

        static constexpr std::size_t kElementCount = N;
        static constexpr EDescriptorStorage kStorage = EDescriptorStorage::dynamic_float;
        static constexpr EDistanceMetric kMetric = EDistanceMetric::squared_l2;
        static constexpr EDescriptorNormalization kNormalization = EDescriptorNormalization::none;

        /**
         * @brief Return an independently owned copy of one descriptor.
         * @param descriptor Descriptor value to copy.
         * @return Copy with no caller-owned backing storage.
         */
        [[nodiscard]] static Descriptor Clone(const Descriptor &descriptor)
        {
            Validate(descriptor);
            return descriptor;
        }

        /** @brief Calculate the element-wise centroid after validating every vector length. */
        [[nodiscard]] static Descriptor Mean(
            const std::span<const Descriptor *const> descriptors)
        {
            if (descriptors.empty())
            {
                throw std::invalid_argument("Cannot calculate a centroid from an empty descriptor set.");
            }

            Descriptor mean(N, 0.0F);
            for (const Descriptor *descriptor : descriptors)
            {
                ValidatePointer(descriptor);
                for (std::size_t index = 0; index < N; ++index)
                {
                    mean[index] += (*descriptor)[index];
                }
            }

            const float inverse_count = 1.0F / static_cast<float>(descriptors.size());
            for (float &value : mean)
            {
                value *= inverse_count;
            }
            return mean;
        }

        /** @brief Calculate squared L2 distance between two validated vectors. */
        [[nodiscard]] static double Distance(const Descriptor &first, const Descriptor &second)
        {
            if (first.size() != N || second.size() != N)
            {
                throw std::invalid_argument("Float descriptor has an incompatible element count.");
            }

            double squared_distance = 0.0;
            for (std::size_t index = 0; index < N; ++index)
            {
                const double difference = static_cast<double>(first[index]) -
                                          static_cast<double>(second[index]);
                squared_distance += difference * difference;
            }

            // A nonfinite input makes its difference and therefore the accumulated norm nonfinite.
            if (!std::isfinite(squared_distance))
            {
                throw std::invalid_argument("Float descriptor contains a nonfinite value.");
            }
            return squared_distance;
        }

        /** @brief Serialize exactly N finite float values. */
        [[nodiscard]] static std::string Serialize(const Descriptor &descriptor)
        {
            Validate(descriptor);

            std::ostringstream stream;
            stream.imbue(std::locale::classic());
            stream << std::setprecision(std::numeric_limits<float>::max_digits10);
            for (std::size_t index = 0; index < N; ++index)
            {
                if (index != 0)
                {
                    stream << ' ';
                }
                stream << descriptor[index];
            }
            return stream.str();
        }

        /** @brief Parse exactly N finite float values without partial output. */
        [[nodiscard]] static bool Deserialize(const std::string_view serialized,
                                              Descriptor &descriptor)
        {
            Descriptor parsed(N, 0.0F);
            std::istringstream stream{std::string(serialized)};
            stream.imbue(std::locale::classic());
            for (float &value : parsed)
            {
                if (!(stream >> value) || !std::isfinite(value))
                {
                    return false;
                }
            }

            stream >> std::ws;
            if (!stream.eof())
            {
                return false;
            }

            descriptor = std::move(parsed);
            return true;
        }

        /** @brief Convert validated vectors into one row-major OpenCV float matrix. */
        [[nodiscard]] static cv::Mat ToMat32F(const std::span<const Descriptor> descriptors)
        {
            if (descriptors.empty())
            {
                return {};
            }
            if (descriptors.size() >
                static_cast<std::size_t>((std::numeric_limits<int>::max)()))
            {
                throw std::length_error("Descriptor count exceeds the OpenCV row range.");
            }

            cv::Mat matrix(static_cast<int>(descriptors.size()), static_cast<int>(N), CV_32F);
            for (std::size_t row = 0; row < descriptors.size(); ++row)
            {
                Validate(descriptors[row]);
                float *output = matrix.ptr<float>(static_cast<int>(row));
                for (std::size_t column = 0; column < N; ++column)
                {
                    output[column] = descriptors[row][column];
                }
            }
            return matrix;
        }

        /**
         * @brief Validate the fixed logical length and finite values of one descriptor.
         * @param descriptor Descriptor received at a public policy boundary.
         * @throws std::invalid_argument If the length differs from N or a value is nonfinite.
         */
        static void Validate(const Descriptor &descriptor)
        {
            if (descriptor.size() != N)
            {
                throw std::invalid_argument("Float descriptor has an incompatible element count.");
            }
            for (const float value : descriptor)
            {
                if (!std::isfinite(value))
                {
                    throw std::invalid_argument("Float descriptor contains a nonfinite value.");
                }
            }
        }

      private:
        static void ValidatePointer(const Descriptor *descriptor)
        {
            if (descriptor == nullptr)
            {
                throw std::invalid_argument("Descriptor centroid input contains a null pointer.");
            }
            Validate(*descriptor);
        }

    };

    /**
     * @brief Fixed-size float descriptor operations using squared L2 distance.
     *
     * The policy rejects nonfinite values and malformed serialized data rather
     * than constructing a partially initialized descriptor.
     *
     * @tparam N Number of float elements in one descriptor.
     * @tparam TNormalization Required input normalization.
     */
    template <std::size_t N,
              EDescriptorNormalization TNormalization = EDescriptorNormalization::none>
    class FixedFloatDescriptorPolicy
    {
      public:
        using Scalar = float;
        using Descriptor = std::array<Scalar, N>;

        static constexpr std::size_t kElementCount = N;
        static constexpr EDescriptorStorage kStorage = EDescriptorStorage::fixed_float;
        static constexpr EDistanceMetric kMetric = EDistanceMetric::squared_l2;
        static constexpr EDescriptorNormalization kNormalization = TNormalization;

        /**
         * @brief Return an independently owned copy of one descriptor.
         * @param descriptor Descriptor value to copy.
         * @return Copy with no caller-owned backing storage.
         */
        [[nodiscard]] static Descriptor Clone(const Descriptor &descriptor)
        {
            Validate(descriptor);
            return descriptor;
        }

        /**
         * @brief Calculate the element-wise centroid of a descriptor set.
         * @param descriptors Non-empty span of non-null descriptor pointers.
         * @return Element-wise arithmetic mean.
         * @throws std::invalid_argument If the set is empty or contains null or nonfinite data.
         */
        [[nodiscard]] static Descriptor Mean(
            const std::span<const Descriptor *const> descriptors)
        {
            if (descriptors.empty())
            {
                throw std::invalid_argument("Cannot calculate a centroid from an empty descriptor set.");
            }

            Descriptor mean{};
            for (const Descriptor *descriptor : descriptors)
            {
                ValidatePointer(descriptor);
                for (std::size_t index = 0; index < N; ++index)
                {
                    mean[index] += (*descriptor)[index];
                }
            }

            const float inverse_count = 1.0F / static_cast<float>(descriptors.size());
            for (float &value : mean)
            {
                value *= inverse_count;
            }
            return mean;
        }

        /**
         * @brief Calculate squared L2 distance between two descriptors.
         * @param first First finite descriptor.
         * @param second Second finite descriptor.
         * @return Squared L2 distance.
         * @throws std::invalid_argument If either descriptor contains a nonfinite value.
         */
        [[nodiscard]] static double Distance(const Descriptor &first, const Descriptor &second)
        {
            double squared_distance = 0.0;
            for (std::size_t index = 0; index < N; ++index)
            {
                const double difference = static_cast<double>(first[index]) -
                                          static_cast<double>(second[index]);
                squared_distance += difference * difference;
            }

            // Fuse finite-value validation with this hot distance pass.
            if (!std::isfinite(squared_distance))
            {
                throw std::invalid_argument("Float descriptor contains a nonfinite value.");
            }
            return squared_distance;
        }

        /**
         * @brief Serialize one finite descriptor using locale-independent text.
         * @param descriptor Descriptor to serialize.
         * @return Space-delimited descriptor values.
         * @throws std::invalid_argument If the descriptor contains a nonfinite value.
         */
        [[nodiscard]] static std::string Serialize(const Descriptor &descriptor)
        {
            Validate(descriptor);

            std::ostringstream stream;
            stream.imbue(std::locale::classic());
            stream << std::setprecision(std::numeric_limits<float>::max_digits10);
            for (std::size_t index = 0; index < N; ++index)
            {
                if (index != 0)
                {
                    stream << ' ';
                }
                stream << descriptor[index];
            }
            return stream.str();
        }

        /**
         * @brief Parse one complete finite descriptor without partial output.
         * @param serialized Space-delimited descriptor values.
         * @param descriptor Output descriptor, modified only on success.
         * @return True when exactly N finite values were parsed.
         */
        [[nodiscard]] static bool Deserialize(const std::string_view serialized,
                                              Descriptor &descriptor)
        {
            Descriptor parsed{};
            std::istringstream stream{std::string(serialized)};
            stream.imbue(std::locale::classic());

            for (float &value : parsed)
            {
                if (!(stream >> value) || !std::isfinite(value))
                {
                    return false;
                }
            }

            stream >> std::ws;
            if (!stream.eof())
            {
                return false;
            }

            descriptor = parsed;
            return true;
        }

        /**
         * @brief Convert descriptors into one row-major OpenCV float matrix.
         * @param descriptors Finite descriptors to convert.
         * @return Matrix with one descriptor per row, or an empty matrix for empty input.
         * @throws std::invalid_argument If a descriptor contains a nonfinite value.
         */
        [[nodiscard]] static cv::Mat ToMat32F(const std::span<const Descriptor> descriptors)
        {
            if (descriptors.empty())
            {
                return {};
            }
            if (descriptors.size() >
                static_cast<std::size_t>((std::numeric_limits<int>::max)()))
            {
                throw std::length_error("Descriptor count exceeds the OpenCV row range.");
            }

            cv::Mat matrix(static_cast<int>(descriptors.size()), static_cast<int>(N), CV_32F);
            for (std::size_t row = 0; row < descriptors.size(); ++row)
            {
                Validate(descriptors[row]);
                float *output = matrix.ptr<float>(static_cast<int>(row));
                for (std::size_t column = 0; column < N; ++column)
                {
                    output[column] = descriptors[row][column];
                }
            }
            return matrix;
        }

        /**
         * @brief Validate that every value in one fixed-size descriptor is finite.
         * @param descriptor Descriptor received at a public policy boundary.
         * @throws std::invalid_argument If any value is nonfinite.
         */
        static void Validate(const Descriptor &descriptor)
        {
            for (const float value : descriptor)
            {
                if (!std::isfinite(value))
                {
                    throw std::invalid_argument("Float descriptor contains a nonfinite value.");
                }
            }
        }

      private:
        static void ValidatePointer(const Descriptor *descriptor)
        {
            if (descriptor == nullptr)
            {
                throw std::invalid_argument("Descriptor centroid input contains a null pointer.");
            }
            Validate(*descriptor);
        }

    };

    static_assert(DescriptorPolicy<FixedFloatDescriptorPolicy<1>>);
    static_assert(LocalDescriptorPolicy<FixedFloatDescriptorPolicy<1>>);
    static_assert(LocalDescriptorPolicy<BitsetDescriptorPolicy<1>>);
    static_assert(LocalDescriptorPolicy<VectorFloatDescriptorPolicy<1>>);
} // namespace DBoW2
