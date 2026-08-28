/**
 * @file TemplatedDatabase.h
 * @brief Policy-typed visual database with validated persistence and indexes.
 * @author Dorian Galvez-Lopez and Pietro Califano
 * @date 2026-08-28
 * @copyright See LICENSE.txt.
 */
 
#ifndef __D_T_TEMPLATED_DATABASE__
#define __D_T_TEMPLATED_DATABASE__

#include <cmath>
#include <vector>
#include <numeric>
#include <fstream>
#include <limits>
#include <string>
#include <list>
#include <span>
#include <set>
#include <stdexcept>
#include <utility>

#include "TemplatedVocabulary.h"
#include "QueryResults.h"
#include "ScoringObject.h"
#include "BowVector.h"
#include "FeatureVector.h"

namespace DBoW2 {

// For query functions
static int MIN_COMMON_WORDS = 5;

/// @brief Generic image database operating on one descriptor policy.
/// @tparam TPolicy Static descriptor policy satisfying DescriptorPolicy.
template <DescriptorPolicy TPolicy>
class TemplatedDatabase
{
public:

  using Policy = TPolicy;
  using Descriptor = DescriptorType<TPolicy>;
  using DescriptorList = std::vector<Descriptor>;
  using Vocabulary = TemplatedVocabulary<TPolicy>;

  /**
   * Creates an empty database with an empty vocabulary.
   * @param use_di a direct index is used to store feature indexes
   * @param di_levels levels to go up the vocabulary tree to select the 
   *   node id to store in the direct index when adding images
   * @throws std::invalid_argument If di_levels is negative.
   */
  explicit TemplatedDatabase(bool use_di = true, int di_levels = 0);

  /**
   * Creates a database with the given vocabulary
   * @param voc vocabulary
   * @param use_di a direct index is used to store feature indexes
   * @param di_levels levels to go up the vocabulary tree to select the 
   *   node id to store in the direct index when adding images
   * @throws std::invalid_argument If di_levels is negative.
   */
  explicit TemplatedDatabase(const Vocabulary &voc, bool use_di = true,
    int di_levels = 0);

  /**
   * Copy constructor. Copies the vocabulary too
   * @param db object to copy
   */
  TemplatedDatabase(const TemplatedDatabase<TPolicy> &db) = default;

  /** @brief Move a database and all of its owned indexes without copying. */
  TemplatedDatabase(TemplatedDatabase<TPolicy> &&db) noexcept = default;

  /** 
   * Creates the database from a file
   * @param filename
   */
  TemplatedDatabase(const std::string &filename);

  /** 
   * Creates the database from a file
   * @param filename
   */
  TemplatedDatabase(const char *filename);

  /**
   * Destructor
   */
  virtual ~TemplatedDatabase() = default;

  /**
   * Copies the given database and its vocabulary
   * @param db database to copy
   */
  TemplatedDatabase<TPolicy>& operator=(const TemplatedDatabase<TPolicy> &db) = default;

  /** @brief Replace this database by moving another owned database snapshot. */
  TemplatedDatabase<TPolicy>& operator=(TemplatedDatabase<TPolicy> &&db) noexcept = default;

  /**
   * Sets the vocabulary to use and clears the content of the database.
   * @param voc vocabulary to copy
   */
  inline void setVocabulary(const Vocabulary &voc);
  
  /**
   * Sets the vocabulary to use and the direct index parameters, and clears
   * the content of the database
   * @param voc vocabulary to copy
   * @param use_di a direct index is used to store feature indexes
   * @param di_levels levels to go up the vocabulary tree to select the 
   *   node id to store in the direct index when adding images
   * @throws std::invalid_argument If di_levels is negative.
   */
  void setVocabulary(const Vocabulary& voc, bool use_di, int di_levels = 0);
  
  /**
   * Returns a pointer to the vocabulary used
   * @return vocabulary
   */
  inline const Vocabulary* getVocabulary() const;

  /** 
   * Allocates some memory for the direct and inverted indexes
   * @param nd number of expected image entries in the database 
   * @param ni number of expected words per image
   * @note Use 0 to ignore a parameter
   */
  void allocate(int nd = 0, int ni = 0);

  /**
   * Adds an entry to the database and returns its index
   * @param features features of the new entry
   * @param bowvec if given, the bow vector of these features is returned
   * @param fvec if given, the vector of nodes and feature indexes is returned
   * @return id of new entry
   */
  EntryId add(const std::vector<DescriptorType<TPolicy>> &features,
    BowVector *bowvec = NULL, FeatureVector *fvec = NULL);

  /**
   * @brief Add a non-owning descriptor batch without copying descriptor storage.
   * @param features Descriptor span consumed synchronously.
   * @param bowvec Optional output BoW vector.
   * @param fvec Optional output direct-index feature vector.
   * @return Dense database entry identifier.
   */
  EntryId add(std::span<const DescriptorType<TPolicy>> features,
    BowVector *bowvec = nullptr, FeatureVector *fvec = nullptr);

  /**
   * Adds an entry to the database and returns its index.
   * @param vec Normalized sparse BoW vector.
   * @param fec feature vector to add the entry. Only necessary if using the
   *   direct index
   * @return id of new entry
   * @throws std::out_of_range If a word or direct-index node ID is invalid.
   * @throws std::invalid_argument If a sparse word weight is nonfinite or not
   *   strictly positive.
   */
  EntryId add(const BowVector &vec, 
    const FeatureVector &fec = FeatureVector() );

  /**
   * Empties the database
   */
  inline void clear();

  /**
   * Returns the number of entries in the database 
   * @return number of entries in the database
   */
  inline unsigned int size() const;
  
  /**
   * Checks if the direct index is being used
   * @return true iff using direct index
   */
  inline bool usingDirectIndex() const;
  
  /**
   * Returns the di levels when using direct index
   * @return di levels
   */
  inline int getDirectIndexLevels() const;
  
  /**
   * Queries the database with some features
   * @param features query features
   * @param ret (out) query results
   * @param max_results number of results to return. <= 0 means all
   * @param max_id only entries with id < max_id are returned in ret.
   *   < 0 means all
   */
  void query(const std::vector<DescriptorType<TPolicy>> &features, QueryResults &ret,
    int max_results = 1, int max_id = -1) const;

  /**
   * @brief Query with a non-owning descriptor batch.
   * @param features Descriptor span consumed synchronously.
   * @param ret Output ranked results.
   * @param max_results Maximum returned results; nonpositive means all.
   * @param max_id Exclusive upper entry-ID bound; negative means all.
   */
  void query(std::span<const DescriptorType<TPolicy>> features, QueryResults &ret,
    int max_results = 1, int max_id = -1) const;
  
  /**
   * Queries the database with a vector
   * @param vec bow vector already normalized
   * @param ret results
   * @param max_results number of results to return. <= 0 means all
   * @param max_id only entries with id < max_id are returned in ret.
   *   < 0 means all
   * @throws std::out_of_range If a word ID is outside the vocabulary.
   * @throws std::invalid_argument If a sparse word weight is nonfinite or not
   *   strictly positive.
   */
  void query(const BowVector &vec, QueryResults &ret, 
    int max_results = 1, int max_id = -1) const;

  /**
   * Returns the a feature vector associated with a database entry
   * @param id entry id (must be < size())
   * @return const reference to map of nodes and their associated features in
   *   the given entry
   */
  const FeatureVector& retrieveFeatures(EntryId id) const;

  /**
   * Stores the database in a file
   * @param filename
   */
  void save(const std::string &filename) const;
  
  /**
   * Loads the database from a file
   * @param filename
   */
  void load(const std::string &filename);
  
  /** 
   * Stores the database in the given file storage structure
   * @param fs
   * @param name node name
   */
  virtual void save(cv::FileStorage &fs, 
    const std::string &name = "database") const;
  
  /** 
   * Loads the database from the given file storage structure
   * @param fs
   * @param name node name
   */
  virtual void load(const cv::FileStorage &fs, 
    const std::string &name = "database");

protected:

  /// Validate an externally supplied BoW vector before indexing the inverted file.
  void validateBowVector(const BowVector &vec) const;

  /// Validate direct-index node IDs against the owned vocabulary.
  void validateFeatureVector(const FeatureVector &features) const;

  /// Validate the requested number of direct-index parent levels.
  static void validateDirectIndexLevels(int di_levels);
  
  /// Query with L1 scoring
  void queryL1(const BowVector &vec, QueryResults &ret, 
    int max_results, int max_id) const;
  
  /// Query with L2 scoring
  void queryL2(const BowVector &vec, QueryResults &ret, 
    int max_results, int max_id) const;
  
  /// Query with Chi square scoring
  void queryChiSquare(const BowVector &vec, QueryResults &ret, 
    int max_results, int max_id) const;
  
  /// Query with Bhattacharyya scoring
  void queryBhattacharyya(const BowVector &vec, QueryResults &ret, 
    int max_results, int max_id) const;
  
  /// Query with KL divergence scoring  
  void queryKL(const BowVector &vec, QueryResults &ret, 
    int max_results, int max_id) const;
  
  /// Query with dot product scoring
  void queryDotProduct(const BowVector &vec, QueryResults &ret, 
    int max_results, int max_id) const;

protected:

  /* Inverted file declaration */
  
  /// Item of IFRow
  struct IFPair
  {
    /// Entry id
    EntryId entry_id;
    
    /// Word weight in this entry
    WordValue word_weight;
    
    /**
     * Creates an empty pair
     */
    IFPair(){}
    
    /**
     * Creates an inverted file pair
     * @param eid entry id
     * @param wv word weight
     */
    IFPair(EntryId eid, WordValue wv): entry_id(eid), word_weight(wv) {}
    
    /**
     * Compares the entry ids
     * @param eid
     * @return true iff this entry id is the same as eid
     */
    inline bool operator==(EntryId eid) const { return entry_id == eid; }
  };
  
  /// Row of InvertedFile
  typedef std::list<IFPair> IFRow;
  // IFRows are sorted in ascending entry_id order
  
  /// Inverted index
  typedef std::vector<IFRow> InvertedFile; 
  // InvertedFile[word_id] --> inverted file of that word
  
  /* Direct file declaration */

  /// Direct index
  typedef std::vector<FeatureVector> DirectFile;
  // DirectFile[entry_id] --> [ directentry, ... ]

protected:

  /// Associated vocabulary
  Vocabulary m_voc;
  
  /// Flag to use direct index
  bool m_use_di;
  
  /// Levels to go up the vocabulary tree to select nodes to store
  /// in the direct index
  int m_dilevels;
  
  /// Inverted file (must have size() == |words|)
  InvertedFile m_ifile;
  
  /// Direct file (resized for allocation)
  DirectFile m_dfile;
  
  /// Number of valid entries in m_dfile
  int m_nentries;
  
};

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
TemplatedDatabase<TPolicy>::TemplatedDatabase
  (bool use_di, int di_levels)
  : m_voc(), m_use_di(use_di), m_dilevels(di_levels), m_nentries(0)
{
  validateDirectIndexLevels(di_levels);
  clear();
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
TemplatedDatabase<TPolicy>::TemplatedDatabase
  (const Vocabulary &voc, bool use_di, int di_levels)
  : m_voc(voc), m_use_di(use_di), m_dilevels(di_levels), m_nentries(0)
{
  validateDirectIndexLevels(di_levels);
  clear();
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
TemplatedDatabase<TPolicy>::TemplatedDatabase
  (const std::string &filename)
  : m_voc(), m_use_di(true), m_dilevels(0), m_nentries(0)
{
  load(filename);
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
TemplatedDatabase<TPolicy>::TemplatedDatabase
  (const char *filename)
  : m_voc(), m_use_di(true), m_dilevels(0), m_nentries(0)
{
  load(filename);
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
EntryId TemplatedDatabase<TPolicy>::add(
  const std::vector<DescriptorType<TPolicy>> &features,
  BowVector *bowvec, FeatureVector *fvec)
{
  return add(std::span<const DescriptorType<TPolicy>>(features), bowvec, fvec);
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
EntryId TemplatedDatabase<TPolicy>::add(
  const std::span<const DescriptorType<TPolicy>> features,
  BowVector *bowvec, FeatureVector *fvec)
{
  BowVector aux;
  BowVector& v = (bowvec ? *bowvec : aux);
  
  if(m_use_di && fvec != NULL)
  {
    m_voc.transform(features, v, *fvec, m_dilevels); // with features
    return add(v, *fvec);
  }
  else if(m_use_di)
  {
    FeatureVector fv;
    m_voc.transform(features, v, fv, m_dilevels); // with features
    return add(v, fv);
  }
  else if(fvec != NULL)
  {
    m_voc.transform(features, v, *fvec, m_dilevels); // with features
    return add(v);
  }
  else
  {
    m_voc.transform(features, v); // with features
    return add(v);
  }
}

// ---------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
EntryId TemplatedDatabase<TPolicy>::add(const BowVector &v,
  const FeatureVector &fv)
{
  validateBowVector(v);
  if(m_use_di)
  {
    validateFeatureVector(fv);
  }
  if(m_nentries == (std::numeric_limits<int>::max)())
  {
    throw std::length_error("Database entry count exceeds the supported range.");
  }

  const EntryId entry_id = static_cast<EntryId>(m_nentries);

  // Reject an inconsistent direct index before changing either index.
  if(m_use_di)
  {
    if(m_dfile.size() != static_cast<std::size_t>(entry_id))
    {
      throw std::logic_error("Database direct index is inconsistent with its entry count.");
    }
  }

  try
  {
    for(const auto &[word_id, word_weight]: v)
    {
      m_ifile[word_id].emplace_back(entry_id, word_weight);
    }
    if(m_use_di)
    {
      m_dfile.push_back(fv);
    }
  }
  catch(...)
  {
    // Roll back rows already appended if a later allocation fails.
    for(const auto &[word_id, word_weight]: v)
    {
      (void)word_weight;
      IFRow &row = m_ifile[word_id];
      if(!row.empty() && row.back().entry_id == entry_id)
      {
        row.pop_back();
      }
    }
    throw;
  }

  ++m_nentries;
  return entry_id;
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::validateBowVector(const BowVector &vec) const
{
  for(const auto &[word_id, word_weight]: vec)
  {
    if(static_cast<std::size_t>(word_id) >= m_ifile.size())
    {
      throw std::out_of_range("BoW vector references a word outside the vocabulary.");
    }
    if(!std::isfinite(word_weight) || word_weight <= 0.0)
    {
      throw std::invalid_argument(
        "BoW vector sparse weights must be finite and strictly positive.");
    }
  }
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::validateFeatureVector(const FeatureVector &features) const
{
  for(const auto &[node_id, feature_ids]: features)
  {
    (void)feature_ids;
    if(!m_voc.isValidNodeId(node_id))
    {
      throw std::out_of_range("Feature vector references a node outside the vocabulary.");
    }
  }
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::validateDirectIndexLevels(const int di_levels)
{
  if(di_levels < 0)
  {
    throw std::invalid_argument("Direct-index levels must be nonnegative.");
  }
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
inline void TemplatedDatabase<TPolicy>::setVocabulary(const Vocabulary& voc)
{
  m_voc = voc;
  clear();
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
inline void TemplatedDatabase<TPolicy>::setVocabulary(
  const Vocabulary& voc, bool use_di, int di_levels)
{
  validateDirectIndexLevels(di_levels);
  m_use_di = use_di;
  m_dilevels = di_levels;
  setVocabulary(voc);
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
inline const typename TemplatedDatabase<TPolicy>::Vocabulary*
TemplatedDatabase<TPolicy>::getVocabulary() const
{
  return &m_voc;
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
inline void TemplatedDatabase<TPolicy>::clear()
{
  m_ifile.assign(m_voc.size(), IFRow{});
  m_dfile.clear();
  m_nentries = 0;
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::allocate(int nd, int ni)
{
  // m_ifile already contains |words| items
  if(ni > 0)
  {
    typename std::vector<IFRow>::iterator rit;
    for(rit = m_ifile.begin(); rit != m_ifile.end(); ++rit)
    {
      int n = (int)rit->size();
      if(ni > n)
      {
        rit->resize(ni);
        rit->resize(n);
      }
    }
  }
  
  if(m_use_di && (int)m_dfile.size() < nd)
  {
    m_dfile.resize(nd);
  }
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
inline unsigned int TemplatedDatabase<TPolicy>::size() const
{
  return m_nentries;
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
inline bool TemplatedDatabase<TPolicy>::usingDirectIndex() const
{
  return m_use_di;
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
inline int TemplatedDatabase<TPolicy>::getDirectIndexLevels() const
{
  return m_dilevels;
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::query(
  const std::vector<DescriptorType<TPolicy>> &features,
  QueryResults &ret, int max_results, int max_id) const
{
  query(std::span<const DescriptorType<TPolicy>>(features), ret, max_results, max_id);
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::query(
  const std::span<const DescriptorType<TPolicy>> features,
  QueryResults &ret, int max_results, int max_id) const
{
  BowVector vec;
  m_voc.transform(features, vec);
  query(vec, ret, max_results, max_id);
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::query(
  const BowVector &vec, 
  QueryResults &ret, int max_results, int max_id) const
{
  validateBowVector(vec);
  ret.resize(0);
  
  switch(m_voc.getScoringType())
  {
    case L1_NORM:
      queryL1(vec, ret, max_results, max_id);
      break;
      
    case L2_NORM:
      queryL2(vec, ret, max_results, max_id);
      break;
      
    case CHI_SQUARE:
      queryChiSquare(vec, ret, max_results, max_id);
      break;
      
    case KL:
      queryKL(vec, ret, max_results, max_id);
      break;
      
    case BHATTACHARYYA:
      queryBhattacharyya(vec, ret, max_results, max_id);
      break;
      
    case DOT_PRODUCT:
      queryDotProduct(vec, ret, max_results, max_id);
      break;
  }
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::queryL1(const BowVector &vec,
  QueryResults &ret, int max_results, int max_id) const
{
  BowVector::const_iterator vit;
  typename IFRow::const_iterator rit;
    
  std::map<EntryId, double> pairs;
  std::map<EntryId, double>::iterator pit;
  
  for(vit = vec.begin(); vit != vec.end(); ++vit)
  {
    const WordId word_id = vit->first;
    const WordValue& qvalue = vit->second;
        
    const IFRow& row = m_ifile[word_id];
    
    // IFRows are sorted in ascending entry_id order
    
    for(rit = row.begin(); rit != row.end(); ++rit)
    {
      const EntryId entry_id = rit->entry_id;
      const WordValue& dvalue = rit->word_weight;
      
      if((int)entry_id < max_id || max_id == -1)
      {
        double value = fabs(qvalue - dvalue) - fabs(qvalue) - fabs(dvalue);
        
        pit = pairs.lower_bound(entry_id);
        if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
        {
          pit->second += value;
        }
        else
        {
          pairs.insert(pit, 
            std::map<EntryId, double>::value_type(entry_id, value));
        }
      }
      
    } // for each inverted row
  } // for each query word
	
  // move to vector
  ret.reserve(pairs.size());
  for(pit = pairs.begin(); pit != pairs.end(); ++pit)
  {
    ret.push_back(Result(pit->first, pit->second));
  }
	
  // resulting "scores" are now in [-2 best .. 0 worst]	
  
  // sort vector in ascending order of score
  std::sort(ret.begin(), ret.end());
  // (ret is inverted now --the lower the better--)

  // cut vector
  if(max_results > 0 && (int)ret.size() > max_results)
    ret.resize(max_results);
  
  // complete and scale score to [0 worst .. 1 best]
  // ||v - w||_{L1} = 2 + Sum(|v_i - w_i| - |v_i| - |w_i|) 
  //		for all i | v_i != 0 and w_i != 0 
  // (Nister, 2006)
  // scaled_||v - w||_{L1} = 1 - 0.5 * ||v - w||_{L1}
  QueryResults::iterator qit;
  for(qit = ret.begin(); qit != ret.end(); qit++) 
    qit->Score = -qit->Score/2.0;
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::queryL2(const BowVector &vec,
  QueryResults &ret, int max_results, int max_id) const
{
  BowVector::const_iterator vit;
  typename IFRow::const_iterator rit;
  
  std::map<EntryId, double> pairs;
  std::map<EntryId, double>::iterator pit;
  
  //map<EntryId, int> counters;
  //map<EntryId, int>::iterator cit;
  
  for(vit = vec.begin(); vit != vec.end(); ++vit)
  {
    const WordId word_id = vit->first;
    const WordValue& qvalue = vit->second;
    
    const IFRow& row = m_ifile[word_id];
    
    // IFRows are sorted in ascending entry_id order
    
    for(rit = row.begin(); rit != row.end(); ++rit)
    {
      const EntryId entry_id = rit->entry_id;
      const WordValue& dvalue = rit->word_weight;
      
      if((int)entry_id < max_id || max_id == -1)
      {
        double value = - qvalue * dvalue; // minus sign for sorting trick
        
        pit = pairs.lower_bound(entry_id);
        //cit = counters.lower_bound(entry_id);
        if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
        {
          pit->second += value; 
          //cit->second += 1;
        }
        else
        {
          pairs.insert(pit, 
            std::map<EntryId, double>::value_type(entry_id, value));
          
          //counters.insert(cit, 
          //  map<EntryId, int>::value_type(entry_id, 1));
        }
      }
      
    } // for each inverted row
  } // for each query word
	
  // move to vector
  ret.reserve(pairs.size());
  //cit = counters.begin();
  for(pit = pairs.begin(); pit != pairs.end(); ++pit)//, ++cit)
  {
    ret.push_back(Result(pit->first, pit->second));// / cit->second));
  }
	
  // resulting "scores" are now in [-1 best .. 0 worst]	
  
  // sort vector in ascending order of score
  std::sort(ret.begin(), ret.end());
  // (ret is inverted now --the lower the better--)

  // cut vector
  if(max_results > 0 && (int)ret.size() > max_results)
    ret.resize(max_results);

  // complete and scale score to [0 worst .. 1 best]
  // ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) 
	//		for all i | v_i != 0 and w_i != 0 )
	// (Nister, 2006)
	QueryResults::iterator qit;
  for(qit = ret.begin(); qit != ret.end(); qit++) 
  {
    if(qit->Score <= -1.0) // rounding error
      qit->Score = 1.0;
    else
      qit->Score = 1.0 - sqrt(1.0 + qit->Score); // [0..1]
      // the + sign is ok, it is due to - sign in 
      // value = - qvalue * dvalue
  }
  
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::queryChiSquare(const BowVector &vec,
  QueryResults &ret, int max_results, int max_id) const
{
  BowVector::const_iterator vit;
  typename IFRow::const_iterator rit;
  
  std::map<EntryId, std::pair<double, int> > pairs;
  std::map<EntryId, std::pair<double, int> >::iterator pit;
  
  std::map<EntryId, std::pair<double, double> > sums; // < sum vi, sum wi >
  std::map<EntryId, std::pair<double, double> >::iterator sit;
  
  // In the current implementation, we suppose vec is not normalized
  
  //map<EntryId, double> expected;
  //map<EntryId, double>::iterator eit;
  
  for(vit = vec.begin(); vit != vec.end(); ++vit)
  {
    const WordId word_id = vit->first;
    const WordValue& qvalue = vit->second;
    
    const IFRow& row = m_ifile[word_id];
    
    // IFRows are sorted in ascending entry_id order
    
    for(rit = row.begin(); rit != row.end(); ++rit)
    {
      const EntryId entry_id = rit->entry_id;
      const WordValue& dvalue = rit->word_weight;
      
      if((int)entry_id < max_id || max_id == -1)
      {
        // (v-w)^2/(v+w) - v - w = -4 vw/(v+w)
        // we move the 4 out
        const double value = -detail::chiSquareProductTerm(qvalue, dvalue);
        
        pit = pairs.lower_bound(entry_id);
        sit = sums.lower_bound(entry_id);
        //eit = expected.lower_bound(entry_id);
        if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
        {
          pit->second.first += value;
          pit->second.second += 1;
          //eit->second += dvalue;
          sit->second.first += qvalue;
          sit->second.second += dvalue;
        }
        else
        {
          pairs.insert(pit, 
            std::map<EntryId, std::pair<double, int> >::value_type(entry_id,
              std::make_pair(value, 1) ));
          //expected.insert(eit, 
          //  map<EntryId, double>::value_type(entry_id, dvalue));
          
          sums.insert(sit, 
            std::map<EntryId, std::pair<double, double> >::value_type(entry_id,
              std::make_pair(qvalue, dvalue) ));
        }
      }
      
    } // for each inverted row
  } // for each query word
	
  // move to vector
  ret.reserve(pairs.size());
  sit = sums.begin();
  for(pit = pairs.begin(); pit != pairs.end(); ++pit, ++sit)
  {
    if(pit->second.second >= MIN_COMMON_WORDS)
    {
      ret.push_back(Result(pit->first, pit->second.first));
      ret.back().nWords = pit->second.second;
      ret.back().sumCommonVi = sit->second.first;
      ret.back().sumCommonWi = sit->second.second;
      ret.back().expectedChiScore = 
        2 * sit->second.second / (1 + sit->second.second);
    }
  
    //ret.push_back(Result(pit->first, pit->second));
  }
	
  // resulting "scores" are now in [-2 best .. 0 worst]	
  // we have to add +2 to the scores to obtain the chi square score
  
  // sort vector in ascending order of score
  std::sort(ret.begin(), ret.end());
  // (ret is inverted now --the lower the better--)

  // cut vector
  if(max_results > 0 && (int)ret.size() > max_results)
    ret.resize(max_results);

  // complete and scale score to [0 worst .. 1 best]
  QueryResults::iterator qit;
  for(qit = ret.begin(); qit != ret.end(); qit++)
  {
    // this takes the 4 into account
    qit->Score = - 2. * qit->Score; // [0..1]
    
    qit->chiScore = qit->Score;
  }
  
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::queryKL(const BowVector &vec,
  QueryResults &ret, int max_results, int max_id) const
{
  BowVector::const_iterator vit;
  typename IFRow::const_iterator rit;
  
  std::map<EntryId, double> pairs;
  std::map<EntryId, double>::iterator pit;
  
  for(vit = vec.begin(); vit != vec.end(); ++vit)
  {
    const WordId word_id = vit->first;
    const WordValue& vi = vit->second;
    
    const IFRow& row = m_ifile[word_id];
    
    // IFRows are sorted in ascending entry_id order
    
    for(rit = row.begin(); rit != row.end(); ++rit)
    {    
      const EntryId entry_id = rit->entry_id;
      const WordValue& wi = rit->word_weight;
      
      if((int)entry_id < max_id || max_id == -1)
      {
        const double value = detail::klDivergenceTerm(vi, wi);
        
        pit = pairs.lower_bound(entry_id);
        if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
        {
          pit->second += value;
        }
        else
        {
          pairs.insert(pit, 
            std::map<EntryId, double>::value_type(entry_id, value));
        }
      }
      
    } // for each inverted row
  } // for each query word
	
  // resulting "scores" are now in [-X worst .. 0 best .. X worst]
  // but we cannot make sure which ones are better without calculating
  // the complete score

  // complete scores and move to vector
  ret.reserve(pairs.size());
  for(pit = pairs.begin(); pit != pairs.end(); ++pit)
  {
    EntryId eid = pit->first;
    double value = 0.0;

    for(vit = vec.begin(); vit != vec.end(); ++vit)
    {
      const WordValue &vi = vit->second;
      const IFRow& row = m_ifile[vit->first];

      if(vi > 0.0)
      {
        if(row.end() == find(row.begin(), row.end(), eid ))
        {
          value += vi * (log(vi) - GeneralScoring::LOG_EPS);
        }
      }
    }
    
    pit->second += value;
    
    // to vector
    ret.push_back(Result(pit->first, pit->second));
  }
  
  // real scores are now in [0 best .. X worst]

  // sort vector in ascending order
  // (scores are inverted now --the lower the better--)
  std::sort(ret.begin(), ret.end());

  // cut vector
  if(max_results > 0 && (int)ret.size() > max_results)
    ret.resize(max_results);

  // cannot scale scores
    
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::queryBhattacharyya(
  const BowVector &vec, QueryResults &ret, int max_results, int max_id) const
{
  BowVector::const_iterator vit;
  typename IFRow::const_iterator rit;
  
  //map<EntryId, double> pairs;
  //map<EntryId, double>::iterator pit;
  
  std::map<EntryId, std::pair<double, int> > pairs; // <eid, <score, counter> >
  std::map<EntryId, std::pair<double, int> >::iterator pit;
  
  for(vit = vec.begin(); vit != vec.end(); ++vit)
  {
    const WordId word_id = vit->first;
    const WordValue& qvalue = vit->second;
    
    const IFRow& row = m_ifile[word_id];
    
    // IFRows are sorted in ascending entry_id order
    
    for(rit = row.begin(); rit != row.end(); ++rit)
    {
      const EntryId entry_id = rit->entry_id;
      const WordValue& dvalue = rit->word_weight;
      
      if((int)entry_id < max_id || max_id == -1)
      {
        const double value = detail::bhattacharyyaTerm(qvalue, dvalue);
        
        pit = pairs.lower_bound(entry_id);
        if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
        {
          pit->second.first += value;
          pit->second.second += 1;
        }
        else
        {
          pairs.insert(pit, 
            std::map<EntryId, std::pair<double, int> >::value_type(entry_id,
              std::make_pair(value, 1)));
        }
      }
      
    } // for each inverted row
  } // for each query word
	
  // move to vector
  ret.reserve(pairs.size());
  for(pit = pairs.begin(); pit != pairs.end(); ++pit)
  {
    if(pit->second.second >= MIN_COMMON_WORDS)
    {
      ret.push_back(Result(pit->first, pit->second.first));
      ret.back().nWords = pit->second.second;
      ret.back().bhatScore = pit->second.first;
    }
  }
	
  // scores are already in [0..1]

  // sort vector in descending order
  std::sort(ret.begin(), ret.end(), Result::gt);

  // cut vector
  if(max_results > 0 && (int)ret.size() > max_results)
    ret.resize(max_results);

}

// ---------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::queryDotProduct(
  const BowVector &vec, QueryResults &ret, int max_results, int max_id) const
{
  BowVector::const_iterator vit;
  typename IFRow::const_iterator rit;
  
  std::map<EntryId, double> pairs;
  std::map<EntryId, double>::iterator pit;
  
  for(vit = vec.begin(); vit != vec.end(); ++vit)
  {
    const WordId word_id = vit->first;
    const WordValue& qvalue = vit->second;
    
    const IFRow& row = m_ifile[word_id];
    
    // IFRows are sorted in ascending entry_id order
    
    for(rit = row.begin(); rit != row.end(); ++rit)
    {
      const EntryId entry_id = rit->entry_id;
      const WordValue& dvalue = rit->word_weight;
      
      if((int)entry_id < max_id || max_id == -1)
      {
        double value; 
        if(this->m_voc.getWeightingType() == BINARY)
          value = 1;
        else
          value = qvalue * dvalue;
        
        pit = pairs.lower_bound(entry_id);
        if(pit != pairs.end() && !(pairs.key_comp()(entry_id, pit->first)))
        {
          pit->second += value;
        }
        else
        {
          pairs.insert(pit, 
            std::map<EntryId, double>::value_type(entry_id, value));
        }
      }
      
    } // for each inverted row
  } // for each query word
	
  // move to vector
  ret.reserve(pairs.size());
  for(pit = pairs.begin(); pit != pairs.end(); ++pit)
  {
    ret.push_back(Result(pit->first, pit->second));
  }
	
  // scores are the greater the better

  // sort vector in descending order
  std::sort(ret.begin(), ret.end(), Result::gt);

  // cut vector
  if(max_results > 0 && (int)ret.size() > max_results)
    ret.resize(max_results);

  // these scores cannot be scaled
}

// ---------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
const FeatureVector& TemplatedDatabase<TPolicy>::retrieveFeatures
  (EntryId id) const
{
  assert(id < size());
  return m_dfile[id];
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::save(const std::string &filename) const
{
  cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
  if(!fs.isOpened()) throw std::string("Could not open file ") + filename;
  
  save(fs);
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::save(cv::FileStorage &fs,
  const std::string &name) const
{
  // Format YAML:
  // vocabulary { ... see TemplatedVocabulary::save }
  // database 
  // {
  //   nEntries: 
  //   usingDI: 
  //   diLevels: 
  //   invertedIndex
  //   [
  //     [
  //        { 
  //          imageId: 
  //          weight: 
  //        }
  //     ]
  //   ]
  //   directIndex
  //   [
  //      [
  //        {
  //          nodeId:
  //          features: [ ]
  //        }
  //      ]
  //   ]

  // invertedIndex[i] is for the i-th word
  // directIndex[i] is for the i-th entry
  // directIndex may be empty if not using direct index
  //
  // imageId's and nodeId's must be stored in ascending order
  // (according to the construction of the indexes)

  m_voc.save(fs);
 
  fs << name << "{";
  
  fs << "nEntries" << m_nentries;
  fs << "usingDI" << (m_use_di ? 1 : 0);
  fs << "diLevels" << m_dilevels;
  
  fs << "invertedIndex" << "[";
  
  typename InvertedFile::const_iterator iit;
  typename IFRow::const_iterator irit;
  for(iit = m_ifile.begin(); iit != m_ifile.end(); ++iit)
  {
    fs << "["; // word of IF
    for(irit = iit->begin(); irit != iit->end(); ++irit)
    {
      fs << "{:" 
        << "imageId" << (int)irit->entry_id
        << "weight" << irit->word_weight
        << "}";
    }
    fs << "]"; // word of IF
  }
  
  fs << "]"; // invertedIndex
  
  fs << "directIndex" << "[";
  
  typename DirectFile::const_iterator dit;
  typename FeatureVector::const_iterator drit;
  for(dit = m_dfile.begin(); dit != m_dfile.end(); ++dit)
  {
    fs << "["; // entry of DF
    
    for(drit = dit->begin(); drit != dit->end(); ++drit)
    {
      NodeId nid = drit->first;
      const std::vector<unsigned int>& features = drit->second;
      
      // save info of last_nid
      fs << "{";
      fs << "nodeId" << (int)nid;
      // Write each value explicitly to avoid aliasing vector<unsigned int> as vector<int>.
      fs << "features" << "[" << "[";
      for(const unsigned int feature: features)
      {
        if(feature > static_cast<unsigned int>((std::numeric_limits<int>::max)()))
        {
          throw std::overflow_error("Feature index exceeds the persistence integer range.");
        }
        fs << static_cast<int>(feature);
      }
      fs << "]" << "]";
      fs << "}";
    }
    
    fs << "]"; // entry of DF
  }
  
  fs << "]"; // directIndex
  
  fs << "}"; // database
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::load(const std::string &filename)
{
  cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened())
  {
    throw std::runtime_error("Could not open database file " + filename);
  }
  
  load(fs);
}

// --------------------------------------------------------------------------

template <DescriptorPolicy TPolicy>
void TemplatedDatabase<TPolicy>::load(const cv::FileStorage &fs,
  const std::string &name)
{
  Vocabulary loaded_vocabulary;
  loaded_vocabulary.load(fs);

  const cv::FileNode database_node = fs[name];
  if(database_node.empty() || !database_node.isMap())
  {
    throw std::runtime_error("Database storage node is missing or is not a map.");
  }

  const auto read_integer = [](const cv::FileNode &node, const char *key) {
    const cv::FileNode field = node[key];
    if(field.empty() || !field.isInt())
    {
      throw std::runtime_error(std::string("Database field '") + key +
        "' must be an integer.");
    }
    return static_cast<int>(field);
  };

  const int entry_count = read_integer(database_node, "nEntries");
  const int direct_index_value = read_integer(database_node, "usingDI");
  const int direct_index_levels = read_integer(database_node, "diLevels");
  if(entry_count < 0 || (direct_index_value != 0 && direct_index_value != 1) ||
    direct_index_levels < 0)
  {
    throw std::runtime_error("Database metadata is outside supported ranges.");
  }
  const bool use_direct_index = direct_index_value != 0;

  const cv::FileNode serialized_inverted_file = database_node["invertedIndex"];
  if(!serialized_inverted_file.isSeq() ||
    serialized_inverted_file.size() != loaded_vocabulary.size())
  {
    throw std::runtime_error(
      "Database inverted index size must match the vocabulary word count.");
  }

  // Validate all persisted entry references before changing the live database.
  InvertedFile loaded_inverted_file(loaded_vocabulary.size());
  for(std::size_t word_index = 0; word_index < serialized_inverted_file.size();
    ++word_index)
  {
    const cv::FileNode serialized_row =
      serialized_inverted_file[static_cast<int>(word_index)];
    if(!serialized_row.isSeq())
    {
      throw std::runtime_error("Every database inverted-index row must be a sequence.");
    }

    bool has_previous_entry = false;
    EntryId previous_entry = 0U;
    for(std::size_t row_index = 0; row_index < serialized_row.size(); ++row_index)
    {
      const cv::FileNode serialized_pair = serialized_row[static_cast<int>(row_index)];
      if(!serialized_pair.isMap())
      {
        throw std::runtime_error("Every inverted-index item must be a map.");
      }

      const int entry_id_value = read_integer(serialized_pair, "imageId");
      const cv::FileNode weight_node = serialized_pair["weight"];
      if(entry_id_value < 0 || entry_id_value >= entry_count || weight_node.empty() ||
        (!weight_node.isInt() && !weight_node.isReal()))
      {
        throw std::runtime_error("Database inverted-index item is outside valid ranges.");
      }

      const EntryId entry_id = static_cast<EntryId>(entry_id_value);
      const WordValue weight = static_cast<double>(weight_node);
      if(!std::isfinite(weight) || weight <= 0.0)
      {
        throw std::runtime_error(
          "Database inverted-index sparse weights must be finite and strictly positive.");
      }
      if(has_previous_entry && entry_id <= previous_entry)
      {
        throw std::runtime_error(
          "Database inverted-index entries must be strictly ordered.");
      }

      loaded_inverted_file[word_index].emplace_back(entry_id, weight);
      previous_entry = entry_id;
      has_previous_entry = true;
    }
  }

  const cv::FileNode serialized_direct_file = database_node["directIndex"];
  if(!serialized_direct_file.isSeq() ||
    (use_direct_index && serialized_direct_file.size() !=
      static_cast<std::size_t>(entry_count)) ||
    (!use_direct_index && serialized_direct_file.size() != 0U))
  {
    throw std::runtime_error(
      "Database direct index is inconsistent with its metadata.");
  }

  DirectFile loaded_direct_file;
  if(use_direct_index)
  {
    loaded_direct_file.resize(static_cast<std::size_t>(entry_count));
    for(std::size_t entry_index = 0; entry_index < serialized_direct_file.size();
      ++entry_index)
    {
      const cv::FileNode serialized_entry =
        serialized_direct_file[static_cast<int>(entry_index)];
      if(!serialized_entry.isSeq())
      {
        throw std::runtime_error("Every database direct-index entry must be a sequence.");
      }

      FeatureVector &loaded_entry = loaded_direct_file[entry_index];
      for(std::size_t item_index = 0; item_index < serialized_entry.size(); ++item_index)
      {
        const cv::FileNode serialized_item =
          serialized_entry[static_cast<int>(item_index)];
        if(!serialized_item.isMap())
        {
          throw std::runtime_error("Every direct-index item must be a map.");
        }

        const int node_id_value = read_integer(serialized_item, "nodeId");
        if(node_id_value < 0 ||
          !loaded_vocabulary.isValidNodeId(static_cast<NodeId>(node_id_value)))
        {
          throw std::runtime_error("Database direct index references an unknown node.");
        }

        const cv::FileNode serialized_features = serialized_item["features"];
        if(!serialized_features.isSeq())
        {
          throw std::runtime_error("Direct-index features must be a sequence.");
        }

        // Accept both the historical nested sequence and the current flat sequence.
        cv::FileNode feature_values = serialized_features;
        if(serialized_features.size() == 1U && serialized_features[0].isSeq())
        {
          feature_values = serialized_features[0];
        }
        else if(!serialized_features.empty() && serialized_features[0].isSeq())
        {
          throw std::runtime_error("Direct-index features have an invalid nested layout.");
        }

        std::vector<unsigned int> features;
        features.reserve(feature_values.size());
        for(const cv::FileNode &feature_node: feature_values)
        {
          if(!feature_node.isInt() || static_cast<int>(feature_node) < 0)
          {
            throw std::runtime_error("Direct-index feature IDs must be nonnegative integers.");
          }
          features.push_back(static_cast<unsigned int>(static_cast<int>(feature_node)));
        }

        const NodeId node_id = static_cast<NodeId>(node_id_value);
        const auto [position, inserted] = loaded_entry.emplace(node_id, std::move(features));
        (void)position;
        if(!inserted)
        {
          throw std::runtime_error("Database direct index contains a duplicate node ID.");
        }
      }
    }
  }

  // Assemble a complete snapshot and publish it with one non-throwing move assignment.
  TemplatedDatabase loaded_database(false);
  loaded_database.m_voc = std::move(loaded_vocabulary);
  loaded_database.m_use_di = use_direct_index;
  loaded_database.m_dilevels = direct_index_levels;
  loaded_database.m_ifile = std::move(loaded_inverted_file);
  loaded_database.m_dfile = std::move(loaded_direct_file);
  loaded_database.m_nentries = entry_count;
  *this = std::move(loaded_database);
}

// --------------------------------------------------------------------------

/**
 * Writes printable information of the database
 * @param os stream to write to
 * @param db
 */
template <DescriptorPolicy TPolicy>
std::ostream& operator<<(std::ostream &os, 
  const TemplatedDatabase<TPolicy> &db)
{
  os << "Database: Entries = " << db.size() << ", "
    "Using direct index = " << (db.usingDirectIndex() ? "yes" : "no");
  
  if(db.usingDirectIndex())
    os << ", Direct index levels = " << db.getDirectIndexLevels();
  
  os << ". " << *db.getVocabulary();
  return os;
}

// --------------------------------------------------------------------------

} // namespace DBoW2

#endif
