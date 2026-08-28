/**
 * File: FBrief.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: functions for BRIEF descriptors
 * License: see the LICENSE.txt file
 *
 */
 
#include <vector>
#include <string>
#include <stdexcept>

#include <DBoW2/FBrief.h>

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

void FBrief::meanValue(const std::vector<FBrief::pDescriptor> &descriptors,
  FBrief::TDescriptor &mean)
{
  mean = Mean(descriptors);
}

// --------------------------------------------------------------------------
  
double FBrief::distance(const FBrief::TDescriptor &a, 
  const FBrief::TDescriptor &b)
{
  return Distance(a, b);
}

// --------------------------------------------------------------------------
  
std::string FBrief::toString(const FBrief::TDescriptor &a)
{
  return Serialize(a);
}

// --------------------------------------------------------------------------
  
void FBrief::fromString(FBrief::TDescriptor &a, const std::string &s)
{
  if(!Deserialize(s, a))
  {
    throw std::invalid_argument("Invalid serialized BRIEF descriptor.");
  }
}

// --------------------------------------------------------------------------

void FBrief::toMat32F(const std::vector<TDescriptor> &descriptors, 
  cv::Mat &mat)
{
  mat = ToMat32F(descriptors);
}

// --------------------------------------------------------------------------

} // namespace DBoW2

