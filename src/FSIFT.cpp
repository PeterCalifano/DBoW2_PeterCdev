/**
 * @file FSIFT.cpp
 * @brief SIFT descriptor policy compatibility implementation.
 * @copyright See LICENSE.txt.
 */

#include <DBoW2/FSIFT.h>

#include <stdexcept>

namespace DBoW2
{
    void FSIFT::meanValue(const std::vector<pDescriptor> &descriptors,
                          TDescriptor &mean)
    {
        mean = Mean(descriptors);
    }

    double FSIFT::distance(const TDescriptor &a, const TDescriptor &b)
    {
        return Distance(a, b);
    }

    std::string FSIFT::toString(const TDescriptor &a)
    {
        return Serialize(a);
    }

    void FSIFT::fromString(TDescriptor &a, const std::string &s)
    {
        if (!Deserialize(s, a))
        {
            throw std::invalid_argument("Invalid serialized SIFT descriptor.");
        }
    }

    void FSIFT::toMat32F(const std::vector<TDescriptor> &descriptors,
                         cv::Mat &mat)
    {
        mat = ToMat32F(descriptors);
    }
} // namespace DBoW2
