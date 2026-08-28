/**
 * File: FORB.cpp
 * Date: June 2012
 * Author: Dorian Galvez-Lopez
 * Description: functions for ORB descriptors
 * License: see the LICENSE.txt file
 *
 */
 
#include <vector>
#include <string>
#include <sstream>
#include <stdint.h>
#include <limits.h>
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
  else if(descriptors.size() == 1)
  {
    mean = descriptors[0]->clone();
  }
  else
  {
    vector<int> sum(FORB::L * 8, 0);
    
    for(size_t i = 0; i < descriptors.size(); ++i)
    {
      const cv::Mat &d = *descriptors[i];
      const unsigned char *p = d.ptr<unsigned char>();
      
      for(int j = 0; j < d.cols; ++j, ++p)
      {
        if(*p & (1 << 7)) ++sum[ j*8     ];
        if(*p & (1 << 6)) ++sum[ j*8 + 1 ];
        if(*p & (1 << 5)) ++sum[ j*8 + 2 ];
        if(*p & (1 << 4)) ++sum[ j*8 + 3 ];
        if(*p & (1 << 3)) ++sum[ j*8 + 4 ];
        if(*p & (1 << 2)) ++sum[ j*8 + 5 ];
        if(*p & (1 << 1)) ++sum[ j*8 + 6 ];
        if(*p & (1))      ++sum[ j*8 + 7 ];
      }
    }
    
    mean = cv::Mat::zeros(1, FORB::L, CV_8U);
    unsigned char *p = mean.ptr<unsigned char>();
    
    const int N2 = (int)descriptors.size() / 2 + descriptors.size() % 2;
    for(size_t i = 0; i < sum.size(); ++i)
    {
      if(sum[i] >= N2)
      {
        // set bit
        *p |= 1 << (7 - (i % 8));
      }
      
      if(i % 8 == 7) ++p;
    }
  }
}

// --------------------------------------------------------------------------
  
double FORB::distance(const FORB::TDescriptor &a, 
  const FORB::TDescriptor &b)
{
  // Bit count function got from:
  // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
  // This implementation assumes that a.cols (CV_8U) % sizeof(uint64_t) == 0
  
  const uint64_t *pa, *pb;
  pa = a.ptr<uint64_t>(); // a & b are actually CV_8U
  pb = b.ptr<uint64_t>(); 
  
  uint64_t v, ret = 0;
  for(size_t i = 0; i < a.cols / sizeof(uint64_t); ++i, ++pa, ++pb)
  {
    v = *pa ^ *pb;
    v = v - ((v >> 1) & (uint64_t)~(uint64_t)0/3);
    v = (v & (uint64_t)~(uint64_t)0/15*3) + ((v >> 2) & 
      (uint64_t)~(uint64_t)0/15*3);
    v = (v + (v >> 4)) & (uint64_t)~(uint64_t)0/255*15;
    ret += (uint64_t)(v * ((uint64_t)~(uint64_t)0/255)) >> 
      (sizeof(uint64_t) - 1) * CHAR_BIT;
  }
  
  return static_cast<double>(ret);
  
  // // If uint64_t is not defined in your system, you can try this 
  // // portable approach (requires DUtils from DLib)
  // const unsigned char *pa, *pb;
  // pa = a.ptr<unsigned char>();
  // pb = b.ptr<unsigned char>();
  // 
  // int ret = 0;
  // for(int i = 0; i < a.cols; ++i, ++pa, ++pb)
  // {
  //   ret += DUtils::LUT::ones8bits[ *pa ^ *pb ];
  // }
  //  
  // return ret;
}

// --------------------------------------------------------------------------
  
std::string FORB::toString(const FORB::TDescriptor &a)
{
  stringstream ss;
  const unsigned char *p = a.ptr<unsigned char>();
  
  for(int i = 0; i < a.cols; ++i, ++p)
  {
    ss << (int)*p << " ";
  }
  
  return ss.str();
}

// --------------------------------------------------------------------------
  
void FORB::fromString(FORB::TDescriptor &a, const std::string &s)
{
  a.create(1, FORB::L, CV_8U);
  unsigned char *p = a.ptr<unsigned char>();
  
  stringstream ss(s);
  for(int i = 0; i < FORB::L; ++i, ++p)
  {
    int n;
    ss >> n;
    
    if(!ss.fail()) 
      *p = (unsigned char)n;
  }
  
}

// --------------------------------------------------------------------------

void FORB::toMat32F(const std::vector<TDescriptor> &descriptors, 
  cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  
  const size_t N = descriptors.size();
  
  mat.create(N, FORB::L*8, CV_32F);
  float *p = mat.ptr<float>();
  
  for(size_t i = 0; i < N; ++i)
  {
    const int C = descriptors[i].cols;
    const unsigned char *desc = descriptors[i].ptr<unsigned char>();
    
    for(int j = 0; j < C; ++j, p += 8)
    {
      p[0] = (desc[j] & (1 << 7) ? 1.f : 0.f);
      p[1] = (desc[j] & (1 << 6) ? 1.f : 0.f);
      p[2] = (desc[j] & (1 << 5) ? 1.f : 0.f);
      p[3] = (desc[j] & (1 << 4) ? 1.f : 0.f);
      p[4] = (desc[j] & (1 << 3) ? 1.f : 0.f);
      p[5] = (desc[j] & (1 << 2) ? 1.f : 0.f);
      p[6] = (desc[j] & (1 << 1) ? 1.f : 0.f);
      p[7] = (desc[j] & (1)      ? 1.f : 0.f);
    }
  } 
}

// --------------------------------------------------------------------------

void FORB::toMat32F(const cv::Mat &descriptors, cv::Mat &mat)
{
  descriptors.convertTo(mat, CV_32F);
}

// --------------------------------------------------------------------------

void FORB::toMat8U(const std::vector<TDescriptor> &descriptors,
  cv::Mat &mat)
{
  mat.create(descriptors.size(), FORB::L, CV_8U);
  
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

  std::vector<pDescriptor> legacy_descriptors;
  legacy_descriptors.reserve(descriptors.size());
  for(const Descriptor *descriptor: descriptors)
  {
    if(descriptor == nullptr)
    {
      throw std::invalid_argument("ORB centroid input contains a null descriptor pointer.");
    }
    Validate(*descriptor);
    legacy_descriptors.push_back(descriptor);
  }

  Descriptor mean;
  meanValue(legacy_descriptors, mean);
  return mean;
}

// --------------------------------------------------------------------------

double FORB::Distance(const Descriptor &first, const Descriptor &second)
{
  Validate(first);
  Validate(second);
  return distance(first, second);
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

  std::vector<Descriptor> legacy_descriptors;
  legacy_descriptors.reserve(descriptors.size());
  for(const Descriptor &descriptor: descriptors)
  {
    Validate(descriptor);
    legacy_descriptors.push_back(descriptor);
  }

  cv::Mat matrix;
  toMat32F(legacy_descriptors, matrix);
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

