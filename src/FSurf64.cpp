/**
 * File: FSurf64.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: functions for Surf64 descriptors
 * License: see the LICENSE.txt file
 *
 */
 
#include <vector>
#include <string>
#include <stdexcept>

#include "FSurf64.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

void FSurf64::meanValue(const std::vector<FSurf64::pDescriptor> &descriptors,
  FSurf64::TDescriptor &mean)
{
  mean = Mean(descriptors);
}

// --------------------------------------------------------------------------
  
double FSurf64::distance(const FSurf64::TDescriptor &a, const FSurf64::TDescriptor &b)
{
  return Distance(a, b);
}

// --------------------------------------------------------------------------

std::string FSurf64::toString(const FSurf64::TDescriptor &a)
{
  return Serialize(a);
}

// --------------------------------------------------------------------------
  
void FSurf64::fromString(FSurf64::TDescriptor &a, const std::string &s)
{
  if(!Deserialize(s, a))
  {
    throw std::invalid_argument("Invalid serialized SURF64 descriptor.");
  }
}

// --------------------------------------------------------------------------

void FSurf64::toMat32F(const std::vector<TDescriptor> &descriptors, 
    cv::Mat &mat)
{
  mat = ToMat32F(descriptors);
}

// --------------------------------------------------------------------------

} // namespace DBoW2

