/**
 * @file ScoringObject.h
 * @brief Polymorphic bag-of-words scoring strategies.
 * @author Dorian Galvez-Lopez and Pietro Califano
 * @date 2026-08-28
 * @copyright See LICENSE.txt.
 */

#ifndef __D_T_SCORING_OBJECT__
#define __D_T_SCORING_OBJECT__

#include <algorithm>
#include <cmath>
#include <limits>

#include "BowVector.h"

namespace DBoW2 {

namespace detail {

/** @internal Numerically stable KL contribution for positive finite operands. */
inline WordValue klDivergenceTerm(const WordValue first, const WordValue second) noexcept
{
  const WordValue ratio = first / second;
  if(ratio >= (std::numeric_limits<WordValue>::min)() &&
    ratio <= (std::numeric_limits<WordValue>::max)()) [[likely]]
  {
    return first * std::log(ratio);
  }

  // A log difference remains finite when the direct ratio overflows or underflows.
  return first * (std::log(first) - std::log(second));
}

/** @internal Numerically stable Chi-square product term for normalized operands. */
inline WordValue chiSquareProductTerm(const WordValue first,
  const WordValue second) noexcept
{
  const WordValue product = first * second;
  if(product >= (std::numeric_limits<WordValue>::min)()) [[likely]]
  {
    return product / (first + second);
  }

  // Scale by the larger operand when the direct product loses normal precision.
  const WordValue smaller = std::min(first, second);
  const WordValue larger = std::max(first, second);
  return smaller / (1.0 + smaller / larger);
}

/** @internal Numerically stable Bhattacharyya term for positive finite operands. */
inline WordValue bhattacharyyaTerm(const WordValue first, const WordValue second) noexcept
{
  const WordValue product = first * second;
  if(product >= (std::numeric_limits<WordValue>::min)()) [[likely]]
  {
    return std::sqrt(product);
  }

  // Separate roots preserve contributions whose direct product underflows.
  return std::sqrt(first) * std::sqrt(second);
}

} // namespace detail

/// Base class of scoring functions
class GeneralScoring
{
public:
  /**
   * @brief Compute the score between two sparse word vectors.
   * @param v First input vector.
   * @param w Second input vector.
   * @return Score defined by the concrete strategy.
   * @pre The vectors are normalized as required by mustNormalize(). Stored
   *   sparse weights are finite and strictly positive; zero is represented by
   *   the absence of a word.
   * @throws std::invalid_argument If a positive-domain strategy encounters an
   *   invalid weight in an arithmetic operand.
   */
  virtual double score(const BowVector &v, const BowVector &w) const = 0;

  /**
   * Returns whether a vector must be normalized before scoring according
   * to the scoring scheme
   * @param norm norm to use
   * @return true iff must normalize
   */
  virtual bool mustNormalize(LNorm &norm) const = 0;

  /// Log of epsilon
	static const double LOG_EPS; 
  // If you change the type of WordValue, make sure you change also the
	// epsilon value (this is needed by the KL method)
	
  virtual ~GeneralScoring() = default; //!< Required for virtual base classes
};

/** 
 * Macro for defining Scoring classes
 * @param NAME name of class
 * @param MUSTNORMALIZE if vectors must be normalized to compute the score
 * @param NORM type of norm to use when MUSTNORMALIZE
 */
#define __SCORING_CLASS(NAME, MUSTNORMALIZE, NORM) \
  NAME: public GeneralScoring \
  { public: \
    /** \
     * @copydoc GeneralScoring::score \
     */ \
    double score(const BowVector &v, const BowVector &w) const override; \
    \
    /** \
     * Says if a vector must be normalized according to the scoring function \
     * @param norm (out) if true, norm to use \
     * @return true iff vectors must be normalized \
     */ \
    inline bool mustNormalize(LNorm &norm) const override \
      { norm = NORM; return MUSTNORMALIZE; } \
  }
  
/// L1 Scoring object
class __SCORING_CLASS(L1Scoring, true, L1);

/// L2 Scoring object
class __SCORING_CLASS(L2Scoring, true, L2);

/// Chi square Scoring object
class __SCORING_CLASS(ChiSquareScoring, true, L1);

/// KL divergence Scoring object
class __SCORING_CLASS(KLScoring, true, L1);

/// Bhattacharyya Scoring object
class __SCORING_CLASS(BhattacharyyaScoring, true, L1);

/// Dot product Scoring object
class __SCORING_CLASS(DotProductScoring, false, L1);

#undef __SCORING_CLASS
  
} // namespace DBoW2

#endif
