/**
 * @file FORB.cpp
 * @brief ORB descriptor policy and backward-compatible helper definitions.
 * @author Dorian Galvez-Lopez and Pietro Califano
 * @date 2026-08-28
 * @copyright See LICENSE.txt.
 */
 
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>
#include <string>
#include <sstream>
#include <locale>
#include <stdexcept>
#include <utility>

#include <DBoW2/FORB.h>

using namespace std;

namespace DBoW2 {

FORB::Descriptor FORB::Clone(const Descriptor &descriptor)
{
  Validate(descriptor);
  return descriptor.clone();
}

// --------------------------------------------------------------------------

void FORB::meanValue(const std::vector<FORB::pDescriptor> &descriptors,
  FORB::TDescriptor &mean)
{
  if(descriptors.empty())
  {
    mean.release();
    return;
  }

  mean = Mean(descriptors);
}

// --------------------------------------------------------------------------
  
double FORB::distance(const FORB::TDescriptor &a,
  const FORB::TDescriptor &b)
{
  return Distance(a, b);
}

// --------------------------------------------------------------------------
  
std::string FORB::toString(const FORB::TDescriptor &a)
{
  return Serialize(a) + ' ';
}

// --------------------------------------------------------------------------
  
void FORB::fromString(FORB::TDescriptor &a, const std::string &s)
{
  if(!Deserialize(s, a))
  {
    throw std::invalid_argument("ORB descriptor text must contain exactly 32 byte values.");
  }
}

// --------------------------------------------------------------------------

void FORB::toMat32F(const std::vector<TDescriptor> &descriptors,
  cv::Mat &mat)
{
  mat = ToMat32F(descriptors);
}

// --------------------------------------------------------------------------

void FORB::toMat32F(const cv::Mat &descriptors, cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  if(descriptors.type() != CV_8UC1 || descriptors.cols != L)
  {
    throw std::invalid_argument("ORB descriptor matrix must have 32 CV_8U columns.");
  }
  descriptors.convertTo(mat, CV_32F);
}

// --------------------------------------------------------------------------

void FORB::toMat8U(const std::vector<TDescriptor> &descriptors,
  cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  if(descriptors.size() >
    static_cast<std::size_t>((std::numeric_limits<int>::max)()))
  {
    throw std::length_error("ORB descriptor count exceeds the OpenCV row range.");
  }
  for(const Descriptor &descriptor: descriptors)
  {
    Validate(descriptor);
  }

  mat.create(static_cast<int>(descriptors.size()), FORB::L, CV_8U);
  
  unsigned char *p = mat.ptr<unsigned char>();
  
  for(size_t i = 0; i < descriptors.size(); ++i, p += FORB::L)
  {
    const unsigned char *d = descriptors[i].ptr<unsigned char>();
    std::copy(d, d + FORB::L, p);
  }
  
}

// --------------------------------------------------------------------------

FORB::Descriptor FORB::Mean(const std::span<const Descriptor *const> descriptors)
{
  if(descriptors.empty())
  {
    throw std::invalid_argument("Cannot calculate a centroid from an empty ORB descriptor set.");
  }

  if(descriptors.size() == 1U)
  {
    if(descriptors.front() == nullptr)
    {
      throw std::invalid_argument("ORB centroid input contains a null descriptor pointer.");
    }
    return Clone(*descriptors.front());
  }

  std::array<std::size_t, kElementCount> bit_counts{};
  for(const Descriptor *descriptor: descriptors)
  {
    if(descriptor == nullptr)
    {
      throw std::invalid_argument("ORB centroid input contains a null descriptor pointer.");
    }
    Validate(*descriptor);

    const std::uint8_t *data = descriptor->ptr<std::uint8_t>();
    for(int byte_index = 0; byte_index < L; ++byte_index)
    {
      const std::size_t offset = static_cast<std::size_t>(byte_index) * 8U;
      if((data[byte_index] & (1U << 7)) != 0U) ++bit_counts[offset];
      if((data[byte_index] & (1U << 6)) != 0U) ++bit_counts[offset + 1U];
      if((data[byte_index] & (1U << 5)) != 0U) ++bit_counts[offset + 2U];
      if((data[byte_index] & (1U << 4)) != 0U) ++bit_counts[offset + 3U];
      if((data[byte_index] & (1U << 3)) != 0U) ++bit_counts[offset + 4U];
      if((data[byte_index] & (1U << 2)) != 0U) ++bit_counts[offset + 5U];
      if((data[byte_index] & (1U << 1)) != 0U) ++bit_counts[offset + 6U];
      if((data[byte_index] & 1U) != 0U) ++bit_counts[offset + 7U];
    }
  }

  Descriptor mean = cv::Mat::zeros(1, L, CV_8U);
  std::uint8_t *output = mean.ptr<std::uint8_t>();
  const std::size_t threshold = descriptors.size() / 2U + descriptors.size() % 2U;
  for(std::size_t bit_index = 0; bit_index < bit_counts.size(); ++bit_index)
  {
    if(bit_counts[bit_index] >= threshold)
    {
      output[bit_index / 8U] |= static_cast<std::uint8_t>(1U << (7U - bit_index % 8U));
    }
  }
  return mean;
}

// --------------------------------------------------------------------------

double FORB::Distance(const Descriptor &first, const Descriptor &second)
{
  Validate(first);
  Validate(second);

  const std::uint8_t *first_data = first.ptr<std::uint8_t>();
  const std::uint8_t *second_data = second.ptr<std::uint8_t>();
  std::size_t hamming_distance = 0U;
  for(std::size_t offset = 0; offset < static_cast<std::size_t>(L);
    offset += sizeof(std::uint64_t))
  {
    std::uint64_t first_word = 0U;
    std::uint64_t second_word = 0U;
    std::memcpy(&first_word, first_data + offset, sizeof(first_word));
    std::memcpy(&second_word, second_data + offset, sizeof(second_word));
    hamming_distance += std::popcount(first_word ^ second_word);
  }
  return static_cast<double>(hamming_distance);
}

// --------------------------------------------------------------------------

std::string FORB::Serialize(const Descriptor &descriptor)
{
  Validate(descriptor);

  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  const std::uint8_t *data = descriptor.ptr<std::uint8_t>();
  for(int index = 0; index < descriptor.cols; ++index)
  {
    if(index != 0)
    {
      stream << ' ';
    }
    stream << static_cast<unsigned int>(data[index]);
  }
  return stream.str();
}

// --------------------------------------------------------------------------

bool FORB::Deserialize(const std::string_view serialized, Descriptor &descriptor)
{
  Descriptor parsed(1, L, CV_8U);
  std::istringstream stream{std::string(serialized)};
  stream.imbue(std::locale::classic());

  std::uint8_t *data = parsed.ptr<std::uint8_t>();
  for(int index = 0; index < L; ++index)
  {
    unsigned int value = 0;
    if(!(stream >> value) || value > 255U)
    {
      return false;
    }
    data[index] = static_cast<std::uint8_t>(value);
  }

  stream >> std::ws;
  if(!stream.eof())
  {
    return false;
  }

  descriptor = std::move(parsed);
  return true;
}

// --------------------------------------------------------------------------

cv::Mat FORB::ToMat32F(const std::span<const Descriptor> descriptors)
{
  if(descriptors.empty())
  {
    return {};
  }
  if(descriptors.size() >
    static_cast<std::size_t>((std::numeric_limits<int>::max)()))
  {
    throw std::length_error("ORB descriptor count exceeds the OpenCV row range.");
  }

  cv::Mat matrix(static_cast<int>(descriptors.size()), L * 8, CV_32F);
  for(std::size_t row = 0; row < descriptors.size(); ++row)
  {
    const Descriptor &descriptor = descriptors[row];
    Validate(descriptor);

    const std::uint8_t *data = descriptor.ptr<std::uint8_t>();
    float *output = matrix.ptr<float>(static_cast<int>(row));
    for(int byte_index = 0; byte_index < L; ++byte_index)
    {
      output[0] = (data[byte_index] & (1U << 7)) != 0U ? 1.0F : 0.0F;
      output[1] = (data[byte_index] & (1U << 6)) != 0U ? 1.0F : 0.0F;
      output[2] = (data[byte_index] & (1U << 5)) != 0U ? 1.0F : 0.0F;
      output[3] = (data[byte_index] & (1U << 4)) != 0U ? 1.0F : 0.0F;
      output[4] = (data[byte_index] & (1U << 3)) != 0U ? 1.0F : 0.0F;
      output[5] = (data[byte_index] & (1U << 2)) != 0U ? 1.0F : 0.0F;
      output[6] = (data[byte_index] & (1U << 1)) != 0U ? 1.0F : 0.0F;
      output[7] = (data[byte_index] & 1U) != 0U ? 1.0F : 0.0F;
      output += 8;
    }
  }
  return matrix;
}

// --------------------------------------------------------------------------

void FORB::Validate(const Descriptor &descriptor)
{
  if(descriptor.type() != CV_8UC1 || descriptor.rows != 1 || descriptor.cols != L)
  {
    throw std::invalid_argument("ORB descriptor must be a 1x32 CV_8U matrix.");
  }
}

// --------------------------------------------------------------------------

} // namespace DBoW2

