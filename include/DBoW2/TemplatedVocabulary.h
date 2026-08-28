/**
 * @file TemplatedVocabulary.h
 * @brief Policy-typed hierarchical visual vocabulary.
 * @author Dorian Galvez-Lopez and Pietro Califano
 * @date 2026-08-28
 * @copyright See LICENSE.txt.
 */

#ifndef __D_T_TEMPLATED_VOCABULARY__
#define __D_T_TEMPLATED_VOCABULARY__

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "BowVector.h"
#include "DescriptorPolicy.h"
#include "FeatureVector.h"
#include "ScoringObject.h"

namespace DBoW2
{

    /// @brief Generic visual vocabulary operating on one descriptor policy.
    /// @tparam TPolicy Static descriptor policy satisfying DescriptorPolicy.
    template <DescriptorPolicy TPolicy>
    class TemplatedVocabulary
    {
      public:
        using Policy = TPolicy;
        using Descriptor = DescriptorType<TPolicy>;
        using DescriptorList = std::vector<Descriptor>;
        using TrainingFeatures = std::vector<DescriptorList>;

        static constexpr bool kIsBinary =
            TPolicy::kStorage == EDescriptorStorage::packed_binary;
        static constexpr std::size_t kDescriptorLength = TPolicy::kElementCount;

        /** @brief Return whether the policy uses packed binary descriptors. */
        [[nodiscard]] static constexpr bool isBinary() { return kIsBinary; }

        /** @brief Return the number of logical elements in each descriptor. */
        [[nodiscard]] static constexpr std::size_t getDescriptorLength()
        {
            return kDescriptorLength;
        }

      public:
        /**
         * Initiates an empty vocabulary
         * @param k branching factor
         * @param L depth levels
         * @param weighting weighting type
         * @param scoring scoring type
         */
        TemplatedVocabulary(int k = 10, int L = 5,
                            WeightingType weighting = TF_IDF, ScoringType scoring = L1_NORM);

        /**
         * Creates the vocabulary by loading a file
         * @param filename
         */
        TemplatedVocabulary(const std::string &filename);

        /**
         * Creates the vocabulary by loading a file
         * @param filename
         */
        TemplatedVocabulary(const char *filename);

        /**
         * Copy constructor
         * @param voc
         */
        TemplatedVocabulary(const TemplatedVocabulary<TPolicy> &voc);

        /**
         * @brief Move a vocabulary while preserving its node-pointer index.
         * @param voc Vocabulary whose owned storage is transferred.
         */
        TemplatedVocabulary(TemplatedVocabulary<TPolicy> &&voc) noexcept = default;

        /**
         * Destructor
         */
        virtual ~TemplatedVocabulary() = default;

        /**
         * Assigns the given vocabulary to this by copying its data and removing
         * all the data contained by this vocabulary before
         * @param voc
         * @return reference to this vocabulary
         */
        TemplatedVocabulary<TPolicy> &operator=(
            const TemplatedVocabulary<TPolicy> &voc);

        /**
         * @brief Replace this vocabulary by transferring another vocabulary's storage.
         * @param voc Vocabulary whose owned storage is transferred.
         * @return Reference to this vocabulary.
         */
        TemplatedVocabulary<TPolicy> &operator=(
            TemplatedVocabulary<TPolicy> &&voc) noexcept = default;

        /**
         * Creates a vocabulary from the training features with the already
         * defined parameters
         * @param training_features
         */
        virtual void create(const std::vector<std::vector<DescriptorType<TPolicy>>> &training_features);

        /**
         * Creates a vocabulary from the training features, setting the branching
         * factor and the depth levels of the tree
         * @param training_features
         * @param k branching factor
         * @param L depth levels
         */
        virtual void create(const std::vector<std::vector<DescriptorType<TPolicy>>> &training_features,
                            int k, int L);

        /**
         * Creates a vocabulary from the training features, setting the branching
         * factor nad the depth levels of the tree, and the weighting and scoring
         * schemes
         */
        virtual void create(const std::vector<std::vector<DescriptorType<TPolicy>>> &training_features,
                            int k, int L, WeightingType weighting, ScoringType scoring);

        /**
         * Returns the number of words in the vocabulary
         * @return number of words
         */
        virtual inline unsigned int size() const;

        /**
         * Returns whether the vocabulary is empty (i.e. it has not been trained)
         * @return true iff the vocabulary is empty
         */
        virtual inline bool empty() const;

        /**
         * Transforms a set of descriptores into a bow vector
         * @param features
         * @param v (out) bow vector of weighted words
         */
        virtual void transform(const std::vector<DescriptorType<TPolicy>> &features, BowVector &v)
            const;

        /**
         * @brief Transform a non-owning descriptor batch into a BoW vector.
         * @param features Descriptor span consumed synchronously.
         * @param v Output normalized word vector.
         */
        void transform(std::span<const DescriptorType<TPolicy>> features, BowVector &v) const;

        /**
         * Transform a set of descriptors into a bow vector and a feature vector
         * @param features
         * @param v (out) bow vector
         * @param fv (out) feature vector of nodes and feature indexes
         * @param levelsup levels to go up the vocabulary tree to get the node index
         * @throws std::invalid_argument If levelsup is negative.
         */
        virtual void transform(const std::vector<DescriptorType<TPolicy>> &features,
                               BowVector &v, FeatureVector &fv, int levelsup) const;

        /**
         * @brief Transform a non-owning descriptor batch into BoW and direct-index vectors.
         * @param features Descriptor span consumed synchronously.
         * @param v Output normalized word vector.
         * @param fv Output node-to-feature-index map.
         * @param levelsup Levels above each word used for direct-index nodes.
         * @throws std::invalid_argument If levelsup is negative.
         */
        void transform(std::span<const DescriptorType<TPolicy>> features,
                       BowVector &v, FeatureVector &fv, int levelsup) const;

        /**
         * Transforms a single feature into a word (without weight)
         * @param feature
         * @return word id
         */
        virtual WordId transform(const DescriptorType<TPolicy> &feature) const;

        /**
         * @brief Return the score between two sparse word vectors.
         * @param a First input vector.
         * @param b Second input vector.
         * @return Score defined by the configured scoring strategy.
         * @pre The vectors are normalized as required by the scoring strategy.
         *   Stored sparse weights are finite and strictly positive; zero is
         *   represented by the absence of a word.
         * @throws std::invalid_argument If a positive-domain strategy encounters
         *   an invalid weight in an arithmetic operand.
         */
        inline double score(const BowVector &a, const BowVector &b) const;

        /**
         * Returns the id of the node that is "levelsup" levels from the word given
         * @param wid word id
         * @param levelsup 0..L
         * @return node id. if levelsup is 0, returns the node id associated to the
         *   word id
         * @throws std::out_of_range If wid does not belong to the vocabulary.
         */
        virtual NodeId getParentNode(WordId wid, int levelsup) const;

        /**
         * Returns the ids of all the words that are under the given node id,
         * by traversing any of the branches that goes down from the node
         * @param nid starting node id
         * @param words ids of words
         * @throws std::out_of_range If nid does not belong to the vocabulary.
         */
        void getWordsFromNode(NodeId nid, std::vector<WordId> &words) const;

        /**
         * Returns the branching factor of the tree (k)
         * @return k
         */
        inline int getBranchingFactor() const { return m_k; }

        /**
         * Returns the depth levels of the tree (L)
         * @return L
         */
        inline int getDepthLevels() const { return m_L; }

        /**
         * Returns the real depth levels of the tree on average
         * @return Average depth of the leaves, or zero for an empty vocabulary.
         */
        float getEffectiveLevels() const;

        /**
         * Returns the descriptor of a word
         * @param wid word id
         * @return descriptor
         * @throws std::out_of_range If wid does not belong to the vocabulary.
         */
        virtual inline DescriptorType<TPolicy> getWord(WordId wid) const;

        /**
         * @brief Return whether a node identifier belongs to this vocabulary.
         * @param nid Candidate node identifier.
         * @return True when nid can be used to index the vocabulary tree.
         */
        [[nodiscard]] inline bool isValidNodeId(NodeId nid) const;

        /**
         * Returns the weight of a word
         * @param wid word id
         * @return weight
         * @throws std::out_of_range If wid does not belong to the vocabulary.
         */
        virtual inline WordValue getWordWeight(WordId wid) const;

        /**
         * Returns the weighting method
         * @return weighting method
         */
        inline WeightingType getWeightingType() const { return m_weighting; }

        /**
         * Returns the scoring method
         * @return scoring method
         */
        inline ScoringType getScoringType() const { return m_scoring; }

        /**
         * Changes the weighting method
         * @param type new weighting type
         */
        inline void setWeightingType(WeightingType type);

        /**
         * Changes the scoring method
         * @param type new scoring type
         */
        void setScoringType(ScoringType type);

        /**
         * Saves the vocabulary into a file
         * @param filename
         */
        void save(const std::string &filename) const;

        /**
         * Loads the vocabulary from a file
         * @param filename
         */
        void load(const std::string &filename);

        /**
         * Saves the vocabulary to a file storage structure
         * @param fs Writable OpenCV file storage.
         * @param name Top-level node name used for the vocabulary.
         */
        virtual void save(cv::FileStorage &fs,
                          const std::string &name = "vocabulary") const;

        /**
         * Loads the vocabulary from a file storage node
         * @param fs Readable OpenCV file storage.
         * @param name Top-level node name containing the vocabulary.
         * @throws std::runtime_error If parameters, tree topology, descriptors, or words are
         *         missing, malformed, or mutually inconsistent.
         */
        virtual void load(const cv::FileStorage &fs,
                          const std::string &name = "vocabulary");

        /**
         * Stops those words whose weight is below minWeight.
         * Words are stopped by setting their weight to 0. There are not returned
         * later when transforming image features into vectors.
         * Note that when using IDF or TF_IDF, the weight is the idf part, which
         * is equivalent to -log(f), where f is the frequency of the word
         * (f = Ni/N, Ni: number of training images where the word is present,
         * N: number of training images).
         * Note that the old weight is forgotten, and subsequent calls to this
         * function with a lower minWeight have no effect.
         * @return number of words stopped now
         */
        virtual int stopWords(double minWeight);

      protected:
        /// Pointer to descriptor
        typedef const DescriptorType<TPolicy> *pDescriptor;

        /// Tree node
        struct Node
        {
            /// Node id
            NodeId id;
            /// Weight if the node is a word
            WordValue weight;
            /// Children
            std::vector<NodeId> children;
            /// Parent node (undefined in case of root)
            NodeId parent;
            /// Node descriptor
            DescriptorType<TPolicy> descriptor;

            /// Word id if the node is a word
            WordId word_id;

            /**
             * Empty constructor
             */
            Node() : id(0), weight(0), parent(0), descriptor(), word_id(0) {}

            /**
             * Constructor
             * @param _id node id
             */
            Node(NodeId _id)
                : id(_id), weight(0), parent(0), descriptor(), word_id(0)
            {
            }

            /**
             * Returns whether the node is a leaf node
             * @return true iff the node is a leaf
             */
            inline bool isLeaf() const { return children.empty(); }
        };

      protected:
        /**
         * Creates an instance of the scoring object accoring to m_scoring
         */
        void createScoringObject();

        /** Validate tree parameters and enumeration-backed strategy values. */
        static void ValidateConfiguration(int k, int L, WeightingType weighting,
                                          ScoringType scoring);

        /** Validate that training input is non-empty and satisfies the descriptor policy. */
        static void ValidateTrainingFeatures(
            const std::vector<std::vector<DescriptorType<TPolicy>>> &training_features);

        /** Validate a public word identifier before indexing vocabulary-owned storage. */
        void validateWordId(WordId wid) const;

        /** Build a vocabulary after all external inputs have been validated. */
        void createValidated(
            const std::vector<std::vector<DescriptorType<TPolicy>>> &training_features);

        /**
         * Returns a set of pointers to descriptores
         * @param training_features all the features
         * @param features (out) pointers to the training features
         */
        void getFeatures(
            const std::vector<std::vector<DescriptorType<TPolicy>>> &training_features,
            std::vector<pDescriptor> &features) const;

        /**
         * Returns the word id associated to a feature
         * @param feature
         * @param id (out) word id
         * @param weight (out) word weight
         * @param nid (out) if given, id of the node "levelsup" levels up
         * @param levelsup
         */
        virtual void transform(const DescriptorType<TPolicy> &feature,
                               WordId &id, WordValue &weight, NodeId *nid = NULL, int levelsup = 0) const;

        /**
         * Returns the word id associated to a feature
         * @param feature
         * @param id (out) word id
         */
        virtual void transform(const DescriptorType<TPolicy> &feature, WordId &id) const;

        /**
         * Creates a level in the tree, under the parent, by running kmeans with
         * a descriptor set, and recursively creates the subsequent levels too
         * @param parent_id id of parent node
         * @param descriptors descriptors to run the kmeans on
         * @param current_level current level in the tree
         */
        void HKmeansStep(NodeId parent_id, const std::vector<pDescriptor> &descriptors,
                         int current_level);

        /**
         * Creates k clusters from the given descriptors with some seeding algorithm.
         * @note In this class, kmeans++ is used, but this function should be
         *   overriden by inherited classes.
         */
        virtual void initiateClusters(const std::vector<pDescriptor> &descriptors,
                                      std::vector<DescriptorType<TPolicy>> &clusters) const;

        /**
         * Creates k clusters from the given descriptor sets by running the
         * initial step of kmeans++
         * @param descriptors
         * @param clusters resulting clusters
         */
        void initiateClustersKMpp(const std::vector<pDescriptor> &descriptors,
                                  std::vector<DescriptorType<TPolicy>> &clusters) const;

        /**
         * Create the words of the vocabulary once the tree has been built
         */
        void createWords();

        /**
         * Sets the weights of the nodes of tree according to the given features.
         * Before calling this function, the nodes and the words must be already
         * created (by calling HKmeansStep and createWords)
         * @param features
         */
        void setNodeWeights(const std::vector<std::vector<DescriptorType<TPolicy>>> &features);

        /**
         * Returns a random number in the range [min..max]
         * @param min
         * @param max
         * @return random T number in [min..max]
         */
        template <class T>
        static T RandomValue(T min, T max)
        {
            return ((T)rand() / (T)RAND_MAX) * (max - min) + min;
        }

        /**
         * Returns a random int in the range [min..max]
         * @param min
         * @param max
         * @return random int in [min..max]
         */
        static int RandomInt(int min, int max)
        {
            int d = max - min + 1;
            return int(((double)rand() / ((double)RAND_MAX + 1.0)) * d) + min;
        }

      protected:
        /// Branching factor
        int m_k;

        /// Depth levels
        int m_L;

        /// Weighting method
        WeightingType m_weighting;

        /// Scoring method
        ScoringType m_scoring;

        /// Object for computing scores
        std::unique_ptr<GeneralScoring> m_scoring_object;

        /// Tree nodes
        std::vector<Node> m_nodes;

        /// Words of the vocabulary (tree leaves)
        /// this condition holds: m_words[wid]->word_id == wid
        std::vector<Node *> m_words;
    };

    // TEMPLATE IMPLEMENTATION
    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    TemplatedVocabulary<TPolicy>::TemplatedVocabulary(
        int k, int L, WeightingType weighting, ScoringType scoring)
        : m_k(k), m_L(L), m_weighting(weighting), m_scoring(scoring),
          m_scoring_object(nullptr)
    {
        ValidateConfiguration(k, L, weighting, scoring);
        createScoringObject();
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    TemplatedVocabulary<TPolicy>::TemplatedVocabulary(const std::string &filename)
        : m_scoring_object(nullptr)
    {
        load(filename);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    TemplatedVocabulary<TPolicy>::TemplatedVocabulary(const char *filename)
        : m_scoring_object(nullptr)
    {
        load(filename);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::createScoringObject()
    {
        switch (m_scoring)
        {
        case L1_NORM:
            m_scoring_object = std::make_unique<L1Scoring>();
            break;

        case L2_NORM:
            m_scoring_object = std::make_unique<L2Scoring>();
            break;

        case CHI_SQUARE:
            m_scoring_object = std::make_unique<ChiSquareScoring>();
            break;

        case KL:
            m_scoring_object = std::make_unique<KLScoring>();
            break;

        case BHATTACHARYYA:
            m_scoring_object = std::make_unique<BhattacharyyaScoring>();
            break;

        case DOT_PRODUCT:
            m_scoring_object = std::make_unique<DotProductScoring>();
            break;

        default:
            throw std::invalid_argument("Unsupported vocabulary scoring type.");
        }
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::setScoringType(ScoringType type)
    {
        ValidateConfiguration(m_k, m_L, m_weighting, type);
        m_scoring = type;
        createScoringObject();
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::setWeightingType(WeightingType type)
    {
        ValidateConfiguration(m_k, m_L, type, m_scoring);
        this->m_weighting = type;
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    TemplatedVocabulary<TPolicy>::TemplatedVocabulary(
        const TemplatedVocabulary<TPolicy> &voc)
        : m_k(voc.m_k), m_L(voc.m_L), m_weighting(voc.m_weighting),
          m_scoring(voc.m_scoring), m_scoring_object(nullptr), m_nodes(voc.m_nodes)
    {
        createScoringObject();
        for (std::size_t index = 1; index < m_nodes.size(); ++index)
        {
            m_nodes[index].descriptor = TPolicy::Clone(voc.m_nodes[index].descriptor);
        }

        // Preserve the persisted word-ID mapping instead of renumbering leaves by node order.
        m_words.resize(voc.m_words.size());
        for (std::size_t word_id = 0; word_id < m_words.size(); ++word_id)
        {
            m_words[word_id] = &m_nodes[voc.m_words[word_id]->id];
        }
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    TemplatedVocabulary<TPolicy> &
    TemplatedVocabulary<TPolicy>::operator=(const TemplatedVocabulary<TPolicy> &voc)
    {
        if (this == &voc)
        {
            return *this;
        }

        TemplatedVocabulary copied(voc);
        *this = std::move(copied);

        return *this;
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::ValidateConfiguration(
        const int k, const int L, const WeightingType weighting, const ScoringType scoring)
    {
        if (k < 2 || L < 1 || weighting < TF_IDF || weighting > BINARY ||
            scoring < L1_NORM || scoring > DOT_PRODUCT)
        {
            throw std::invalid_argument(
                "Vocabulary requires k >= 2, L >= 1, and supported weighting/scoring types.");
        }
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::ValidateTrainingFeatures(
        const std::vector<std::vector<DescriptorType<TPolicy>>> &training_features)
    {
        std::size_t descriptor_count = 0U;
        for (const auto &frame_descriptors : training_features)
        {
            for (const DescriptorType<TPolicy> &descriptor : frame_descriptors)
            {
                if (descriptor_count ==
                    static_cast<std::size_t>((std::numeric_limits<int>::max)()))
                {
                    throw std::length_error(
                        "Vocabulary training descriptor count exceeds the index range.");
                }
                TPolicy::Validate(descriptor);
                ++descriptor_count;
            }
        }
        if (descriptor_count == 0U)
        {
            throw std::invalid_argument(
                "Vocabulary training requires at least one valid descriptor.");
        }
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::validateWordId(const WordId wid) const
    {
        if (static_cast<std::size_t>(wid) >= m_words.size())
        {
            throw std::out_of_range("Word ID does not belong to this vocabulary.");
        }
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::createValidated(
        const std::vector<std::vector<DescriptorType<TPolicy>>> &training_features)
    {
        // Compute the complete k-ary tree bound without floating-point overflow or division.
        std::size_t expected_nodes = 1U;
        std::size_t nodes_at_level = 1U;
        for (int level = 1; level <= m_L; ++level)
        {
            const std::size_t branching_factor = static_cast<std::size_t>(m_k);
            if (nodes_at_level >
                    ((std::numeric_limits<std::size_t>::max)() - expected_nodes) /
                        branching_factor)
            {
                throw std::length_error("Vocabulary tree parameters exceed the size range.");
            }
            nodes_at_level *= branching_factor;
            expected_nodes += nodes_at_level;
        }
        if (expected_nodes > m_nodes.max_size() ||
            expected_nodes > static_cast<std::size_t>((std::numeric_limits<NodeId>::max)()))
        {
            throw std::length_error("Vocabulary tree exceeds the supported node range.");
        }

        m_nodes.clear();
        m_words.clear();

        m_nodes.reserve(expected_nodes); // avoid allocations when creating the tree

        std::vector<pDescriptor> features;
        getFeatures(training_features, features);

        // create root
        m_nodes.push_back(Node(0)); // root

        // create the tree
        HKmeansStep(0, features, 1);

        // create the words
        createWords();

        // and set the weight of each node of the tree
        setNodeWeights(training_features);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::create(
        const std::vector<std::vector<DescriptorType<TPolicy>>> &training_features)
    {
        ValidateConfiguration(m_k, m_L, m_weighting, m_scoring);
        ValidateTrainingFeatures(training_features);
        createValidated(training_features);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::create(
        const std::vector<std::vector<DescriptorType<TPolicy>>> &training_features,
        int k, int L)
    {
        ValidateConfiguration(k, L, m_weighting, m_scoring);
        ValidateTrainingFeatures(training_features);
        m_k = k;
        m_L = L;

        createValidated(training_features);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::create(
        const std::vector<std::vector<DescriptorType<TPolicy>>> &training_features,
        int k, int L, WeightingType weighting, ScoringType scoring)
    {
        ValidateConfiguration(k, L, weighting, scoring);
        ValidateTrainingFeatures(training_features);
        m_k = k;
        m_L = L;
        m_weighting = weighting;
        m_scoring = scoring;
        createScoringObject();

        createValidated(training_features);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::getFeatures(
        const std::vector<std::vector<DescriptorType<TPolicy>>> &training_features,
        std::vector<pDescriptor> &features) const
    {
        features.resize(0);

        typename std::vector<std::vector<DescriptorType<TPolicy>>>::const_iterator vvit;
        typename std::vector<DescriptorType<TPolicy>>::const_iterator vit;
        for (vvit = training_features.begin(); vvit != training_features.end(); ++vvit)
        {
            features.reserve(features.size() + vvit->size());
            for (vit = vvit->begin(); vit != vvit->end(); ++vit)
            {
                features.push_back(&(*vit));
            }
        }
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::HKmeansStep(
        NodeId parent_id,
        const std::vector<pDescriptor> &descriptors,
        int current_level)
    {
        if (descriptors.empty())
            return;

        // features associated to each cluster
        std::vector<DescriptorType<TPolicy>> clusters;
        // groups[i] stores descriptor indices associated with cluster i.
        std::vector<std::vector<unsigned int>> groups;

        clusters.reserve(m_k);
        groups.reserve(m_k);

        // const int msizes[] = { m_k, descriptors.size() };
        // cv::SparseMat assoc(2, msizes, CV_8U);
        // cv::SparseMat last_assoc(2, msizes, CV_8U);
        //// assoc.row(cluster_idx).col(descriptor_idx) = 1 iif associated

        if ((int)descriptors.size() <= m_k)
        {
            // trivial case: one cluster per feature
            groups.resize(descriptors.size());

            for (unsigned int i = 0; i < descriptors.size(); i++)
            {
                groups[i].push_back(i);
                clusters.push_back(TPolicy::Clone(*descriptors[i]));
            }
        }
        else
        {
            // select clusters and groups with kmeans

            bool first_time = true;
            bool goon = true;

            // to check if clusters move after iterations
            std::vector<int> last_association, current_association;

            while (goon)
            {
                // 1. Calculate clusters

                if (first_time)
                {
                    // random sample
                    initiateClusters(descriptors, clusters);
                }
                else
                {
                    // calculate cluster centres

                    for (unsigned int c = 0; c < clusters.size(); ++c)
                    {
                        std::vector<pDescriptor> cluster_descriptors;
                        cluster_descriptors.reserve(groups[c].size());

                        /*
                        for(unsigned int d = 0; d < descriptors.size(); ++d)
                        {
                          if( assoc.find<unsigned char>(c, d) )
                          {
                            cluster_descriptors.push_back(descriptors[d]);
                          }
                        }
                        */

                        std::vector<unsigned int>::const_iterator vit;
                        for (vit = groups[c].begin(); vit != groups[c].end(); ++vit)
                        {
                            cluster_descriptors.push_back(descriptors[*vit]);
                        }

                        clusters[c] = TPolicy::Mean(cluster_descriptors);
                    }

                } // if(!first_time)

                // 2. Associate features with clusters

                // calculate distances to cluster centers
                groups.clear();
                groups.resize(clusters.size(), std::vector<unsigned int>());
                current_association.resize(descriptors.size());

                // assoc.clear();

                typename std::vector<pDescriptor>::const_iterator fit;
                // unsigned int d = 0;
                for (fit = descriptors.begin(); fit != descriptors.end(); ++fit) //, ++d)
                {
                    double best_dist = TPolicy::Distance(*(*fit), clusters[0]);
                    unsigned int icluster = 0;

                    for (unsigned int c = 1; c < clusters.size(); ++c)
                    {
                        double dist = TPolicy::Distance(*(*fit), clusters[c]);
                        if (dist < best_dist)
                        {
                            best_dist = dist;
                            icluster = c;
                        }
                    }

                    // assoc.ref<unsigned char>(icluster, d) = 1;

                    const auto feature_index = static_cast<unsigned int>(
                        std::distance(descriptors.begin(), fit));
                    groups[icluster].push_back(feature_index);
                    current_association[feature_index] = icluster;
                }

                // kmeans++ ensures all the clusters has any feature associated with them

                // 3. check convergence
                if (first_time)
                {
                    first_time = false;
                }
                else
                {
                    // goon = !eqUChar(last_assoc, assoc);

                    goon = false;
                    for (unsigned int i = 0; i < current_association.size(); i++)
                    {
                        if (current_association[i] != last_association[i])
                        {
                            goon = true;
                            break;
                        }
                    }
                }

                if (goon)
                {
                    // copy last feature-cluster association
                    last_association = current_association;
                    // last_assoc = assoc.clone();
                }

            } // while(goon)

        } // if must run kmeans

        // create nodes
        for (unsigned int i = 0; i < clusters.size(); ++i)
        {
            const NodeId id = static_cast<NodeId>(m_nodes.size());
            m_nodes.push_back(Node(id));
            m_nodes.back().descriptor = TPolicy::Clone(clusters[i]);
            m_nodes.back().parent = parent_id;
            m_nodes[parent_id].children.push_back(id);
        }

        // go on with the next level
        if (current_level < m_L)
        {
            // iterate again with the resulting clusters
            const std::vector<NodeId> &children_ids = m_nodes[parent_id].children;
            for (unsigned int i = 0; i < clusters.size(); ++i)
            {
                NodeId id = children_ids[i];

                std::vector<pDescriptor> child_features;
                child_features.reserve(groups[i].size());

                std::vector<unsigned int>::const_iterator vit;
                for (vit = groups[i].begin(); vit != groups[i].end(); ++vit)
                {
                    child_features.push_back(descriptors[*vit]);
                }

                if (child_features.size() > 1)
                {
                    HKmeansStep(id, child_features, current_level + 1);
                }
            }
        }
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::initiateClusters(
        const std::vector<pDescriptor> &descriptors,
        std::vector<DescriptorType<TPolicy>> &clusters) const
    {
        initiateClustersKMpp(descriptors, clusters);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::initiateClustersKMpp(
        const std::vector<pDescriptor> &pfeatures,
        std::vector<DescriptorType<TPolicy>> &clusters) const
    {
        // Implements kmeans++ seeding algorithm
        // Algorithm:
        // 1. Choose one center uniformly at random from among the data points.
        // 2. For each data point x, compute D(x), the distance between x and the nearest
        //    center that has already been chosen.
        // 3. Add one new data point as a center. Each point x is chosen with probability
        //    proportional to D(x)^2.
        // 4. Repeat Steps 2 and 3 until k centers have been chosen.
        // 5. Now that the initial centers have been chosen, proceed using standard k-means
        //    clustering.

        clusters.resize(0);
        clusters.reserve(m_k);
        std::vector<double> min_dists(pfeatures.size(), std::numeric_limits<double>::max());

        // 1.

        std::size_t ifeature = static_cast<std::size_t>(
            RandomInt(0, static_cast<int>(pfeatures.size() - 1U)));

        // create first cluster
        clusters.push_back(TPolicy::Clone(*pfeatures[ifeature]));

        // compute the initial distances
        typename std::vector<pDescriptor>::const_iterator fit;
        std::vector<double>::iterator dit;
        dit = min_dists.begin();
        for (fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
        {
            *dit = TPolicy::Distance(*(*fit), clusters.back());
        }

        while ((int)clusters.size() < m_k)
        {
            // 2.
            dit = min_dists.begin();
            for (fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
            {
                if (*dit > 0)
                {
                    double dist = TPolicy::Distance(*(*fit), clusters.back());
                    if (dist < *dit)
                        *dit = dist;
                }
            }

            // 3.
            double dist_sum = std::accumulate(min_dists.begin(), min_dists.end(), 0.0);

            if (dist_sum > 0)
            {
                double cut_d;
                do
                {
                    cut_d = RandomValue<double>(0, dist_sum);
                } while (cut_d <= 0.0);

                double d_up_now = 0;
                for (dit = min_dists.begin(); dit != min_dists.end(); ++dit)
                {
                    d_up_now += *dit;
                    if (d_up_now >= cut_d)
                        break;
                }

                if (dit == min_dists.end())
                    ifeature = pfeatures.size() - 1U;
                else
                    ifeature = static_cast<std::size_t>(
                        std::distance(min_dists.begin(), dit));

                clusters.push_back(TPolicy::Clone(*pfeatures[ifeature]));

            } // if dist_sum > 0
            else
                break;

        } // while(used_clusters < m_k)
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::createWords()
    {
        m_words.clear();

        if (!m_nodes.empty())
        {
            // Reserve the single-pass upper bound instead of traversing all nodes twice merely
            // to determine the exact number of leaves.
            m_words.reserve(m_nodes.size() - 1U);

            typename std::vector<Node>::iterator nit;

            nit = m_nodes.begin(); // ignore root
            for (++nit; nit != m_nodes.end(); ++nit)
            {
                if (nit->isLeaf())
                {
                    nit->word_id = static_cast<WordId>(m_words.size());
                    m_words.push_back(&(*nit));
                }
            }
        }
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::setNodeWeights(
        const std::vector<std::vector<DescriptorType<TPolicy>>> &training_features)
    {
        const std::size_t word_count = m_words.size();
        const double document_count = static_cast<double>(training_features.size());

        if (m_weighting == TF || m_weighting == BINARY)
        {
            // idf part must be 1 always
            for (std::size_t i = 0; i < word_count; ++i)
                m_words[i]->weight = 1;
        }
        else if (m_weighting == IDF || m_weighting == TF_IDF)
        {
            // IDF and TF-IDF: we calculte the idf path now

            // Note: this actually calculates the idf part of the tf-idf score.
            // The complete tf-idf score is calculated in ::transform

            std::vector<std::size_t> document_frequencies(word_count, 0U);
            std::vector<bool> counted(word_count, false);

            typename std::vector<std::vector<DescriptorType<TPolicy>>>::const_iterator mit;
            typename std::vector<DescriptorType<TPolicy>>::const_iterator fit;

            for (mit = training_features.begin(); mit != training_features.end(); ++mit)
            {
                fill(counted.begin(), counted.end(), false);

                for (fit = mit->begin(); fit < mit->end(); ++fit)
                {
                    WordId word_id;
                    transform(*fit, word_id);

                    if (!counted[word_id])
                    {
                        ++document_frequencies[word_id];
                        counted[word_id] = true;
                    }
                }
            }

            // set ln(N/Ni)
            for (std::size_t i = 0; i < word_count; ++i)
            {
                if (document_frequencies[i] > 0U)
                {
                    m_words[i]->weight =
                        log(document_count / static_cast<double>(document_frequencies[i]));
                } // else // This cannot occur if using kmeans++
            }
        }
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    inline unsigned int TemplatedVocabulary<TPolicy>::size() const
    {
        return static_cast<unsigned int>(m_words.size());
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    inline bool TemplatedVocabulary<TPolicy>::empty() const
    {
        return m_words.empty();
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    float TemplatedVocabulary<TPolicy>::getEffectiveLevels() const
    {
        if (m_words.empty())
        {
            return 0.0F;
        }

        std::size_t depth_sum = 0U;
        typename std::vector<Node *>::const_iterator wit;
        for (wit = m_words.begin(); wit != m_words.end(); ++wit)
        {
            const Node *p = *wit;

            for (; p->id != 0; ++depth_sum)
                p = &m_nodes[p->parent];
        }

        return static_cast<float>(static_cast<double>(depth_sum) /
                                  static_cast<double>(m_words.size()));
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    DescriptorType<TPolicy> TemplatedVocabulary<TPolicy>::getWord(WordId wid) const
    {
        validateWordId(wid);
        return TPolicy::Clone(m_words[wid]->descriptor);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    bool TemplatedVocabulary<TPolicy>::isValidNodeId(const NodeId nid) const
    {
        return static_cast<std::size_t>(nid) < m_nodes.size();
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    WordValue TemplatedVocabulary<TPolicy>::getWordWeight(WordId wid) const
    {
        validateWordId(wid);
        return m_words[wid]->weight;
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    WordId TemplatedVocabulary<TPolicy>::transform(const DescriptorType<TPolicy> &feature) const
    {
        if (empty())
        {
            return 0;
        }

        WordId wid;
        transform(feature, wid);
        return wid;
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::transform(
        const std::vector<DescriptorType<TPolicy>> &features, BowVector &v) const
    {
        transform(std::span<const DescriptorType<TPolicy>>(features), v);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::transform(
        const std::span<const DescriptorType<TPolicy>> features, BowVector &v) const
    {
        v.clear();

        if (empty())
        {
            return;
        }

        // normalize
        LNorm norm;
        bool must = m_scoring_object->mustNormalize(norm);

        if (m_weighting == TF || m_weighting == TF_IDF)
        {
            for (const DescriptorType<TPolicy> &feature : features)
            {
                WordId id;
                WordValue w;
                // w is the idf value if TF_IDF, 1 if TF

                transform(feature, id, w);

                // not stopped
                if (w > 0)
                    v.addWeight(id, w);
            }

            if (!v.empty() && !must)
            {
                // unnecessary when normalizing
                const double nd = static_cast<double>(v.size());
                for (BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
                    vit->second /= nd;
            }
        }
        else // IDF || BINARY
        {
            for (const DescriptorType<TPolicy> &feature : features)
            {
                WordId id;
                WordValue w;
                // w is idf if IDF, or 1 if BINARY

                transform(feature, id, w);

                // not stopped
                if (w > 0)
                    v.addIfNotExist(id, w);

            } // if add_features
        } // if m_weighting == ...

        if (must)
            v.normalize(norm);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::transform(
        const std::vector<DescriptorType<TPolicy>> &features,
        BowVector &v, FeatureVector &fv, int levelsup) const
    {
        transform(std::span<const DescriptorType<TPolicy>>(features), v, fv, levelsup);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::transform(
        const std::span<const DescriptorType<TPolicy>> features,
        BowVector &v, FeatureVector &fv, int levelsup) const
    {
        if (levelsup < 0)
        {
            throw std::invalid_argument("Direct-index levels must be nonnegative.");
        }

        v.clear();
        fv.clear();

        if (empty()) // safe for subclasses
        {
            return;
        }

        // normalize
        LNorm norm;
        bool must = m_scoring_object->mustNormalize(norm);

        if (m_weighting == TF || m_weighting == TF_IDF)
        {
            unsigned int i_feature = 0;
            for (const DescriptorType<TPolicy> &feature : features)
            {
                WordId id;
                NodeId nid;
                WordValue w;
                // w is the idf value if TF_IDF, 1 if TF

                transform(feature, id, w, &nid, levelsup);

                if (w > 0) // not stopped
                {
                    v.addWeight(id, w);
                    fv.addFeature(nid, i_feature);
                }
                ++i_feature;
            }

            if (!v.empty() && !must)
            {
                // unnecessary when normalizing
                const double nd = static_cast<double>(v.size());
                for (BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
                    vit->second /= nd;
            }
        }
        else // IDF || BINARY
        {
            unsigned int i_feature = 0;
            for (const DescriptorType<TPolicy> &feature : features)
            {
                WordId id;
                NodeId nid;
                WordValue w;
                // w is idf if IDF, or 1 if BINARY

                transform(feature, id, w, &nid, levelsup);

                if (w > 0) // not stopped
                {
                    v.addIfNotExist(id, w);
                    fv.addFeature(nid, i_feature);
                }
                ++i_feature;
            }
        } // if m_weighting == ...

        if (must)
            v.normalize(norm);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    inline double TemplatedVocabulary<TPolicy>::score(const BowVector &v1, const BowVector &v2) const
    {
        return m_scoring_object->score(v1, v2);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::transform(const DescriptorType<TPolicy> &feature, WordId &id) const
    {
        WordValue weight;
        transform(feature, id, weight);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::transform(
        const DescriptorType<TPolicy> &feature,
        WordId &word_id,
        WordValue &weight,
        NodeId *nid,
        int levelsup) const
    {
        if (levelsup < 0)
        {
            throw std::invalid_argument("Direct-index levels must be nonnegative.");
        }

        // propagate the feature down the tree
        NodeId final_id = 0; // root

        do
        {
            const std::vector<NodeId> &children = m_nodes[final_id].children;
            final_id = children.front();

            double best_d = TPolicy::Distance(feature, m_nodes[final_id].descriptor);

            for (auto child = children.begin() + 1; child != children.end(); ++child)
            {
                const NodeId id = *child;
                const double d = TPolicy::Distance(feature, m_nodes[id].descriptor);
                if (d < best_d)
                {
                    best_d = d;
                    final_id = id;
                }
            }

        } while (!m_nodes[final_id].isLeaf());

        // Resolve the direct-index node from the actual leaf so shallow branches remain valid.
        if (nid != nullptr)
        {
            *nid = final_id;
            for (int level = 0; level < levelsup && *nid != 0U; ++level)
            {
                *nid = m_nodes[*nid].parent;
            }
        }

        // turn node id into word id
        word_id = m_nodes[final_id].word_id;
        weight = m_nodes[final_id].weight;
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    NodeId TemplatedVocabulary<TPolicy>::getParentNode(WordId wid, int levelsup) const
    {
        validateWordId(wid);
        NodeId ret = m_words[wid]->id;   // node id
        while (levelsup > 0 && ret != 0) // ret == 0 --> root
        {
            --levelsup;
            ret = m_nodes[ret].parent;
        }
        return ret;
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::getWordsFromNode(NodeId nid, std::vector<WordId> &words) const
    {
        if (!isValidNodeId(nid))
        {
            throw std::out_of_range("Node ID does not belong to this vocabulary.");
        }

        words.clear();

        if (m_nodes[nid].isLeaf())
        {
            words.push_back(m_nodes[nid].word_id);
        }
        else
        {
            words.reserve(m_k); // ^1, ^2, ...

            std::vector<NodeId> parents;
            parents.push_back(nid);

            while (!parents.empty())
            {
                NodeId parentid = parents.back();
                parents.pop_back();

                const std::vector<NodeId> &child_ids = m_nodes[parentid].children;
                std::vector<NodeId>::const_iterator cit;

                for (cit = child_ids.begin(); cit != child_ids.end(); ++cit)
                {
                    const Node &child_node = m_nodes[*cit];

                    if (child_node.isLeaf())
                        words.push_back(child_node.word_id);
                    else
                        parents.push_back(*cit);

                } // for each child
            } // while !parents.empty
        }
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    int TemplatedVocabulary<TPolicy>::stopWords(double minWeight)
    {
        int c = 0;
        typename std::vector<Node *>::iterator wit;
        for (wit = m_words.begin(); wit != m_words.end(); ++wit)
        {
            if ((*wit)->weight < minWeight)
            {
                ++c;
                (*wit)->weight = 0;
            }
        }
        return c;
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::save(const std::string &filename) const
    {
        cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
        if (!fs.isOpened())
            throw std::string("Could not open file ") + filename;

        save(fs);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::load(const std::string &filename)
    {
        cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            throw std::string("Could not open file ") + filename;
        }

        this->load(fs);
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::save(cv::FileStorage &f,
                                                          const std::string &name) const
    {
        // Format YAML:
        // vocabulary
        // {
        //   k:
        //   L:
        //   scoringType:
        //   weightingType:
        //   nodes
        //   [
        //     {
        //       nodeId:
        //       parentId:
        //       weight:
        //       descriptor:
        //     }
        //   ]
        //   words
        //   [
        //     {
        //       wordId:
        //       nodeId:
        //     }
        //   ]
        // }
        //
        // The root node (index 0) is not included in the node vector
        //

        f << name << "{";

        f << "k" << m_k;
        f << "L" << m_L;
        f << "scoringType" << m_scoring;
        f << "weightingType" << m_weighting;

        // tree
        f << "nodes" << "[";
        std::vector<NodeId> parents, children;
        std::vector<NodeId>::const_iterator pit;

        parents.push_back(0); // root

        while (!parents.empty())
        {
            NodeId pid = parents.back();
            parents.pop_back();

            const Node &parent = m_nodes[pid];
            children = parent.children;

            for (pit = children.begin(); pit != children.end(); pit++)
            {
                const Node &child = m_nodes[*pit];

                // save node data
                f << "{:";
                f << "nodeId" << (int)child.id;
                f << "parentId" << (int)pid;
                f << "weight" << (double)child.weight;
                f << "descriptor" << TPolicy::Serialize(child.descriptor);
                f << "}";

                // add to parent list
                if (!child.isLeaf())
                {
                    parents.push_back(*pit);
                }
            }
        }

        f << "]"; // nodes

        // words
        f << "words" << "[";

        typename std::vector<Node *>::const_iterator wit;
        for (wit = m_words.begin(); wit != m_words.end(); wit++)
        {
            const WordId id = static_cast<WordId>(std::distance(m_words.begin(), wit));
            f << "{:";
            f << "wordId" << (int)id;
            f << "nodeId" << (int)(*wit)->id;
            f << "}";
        }

        f << "]"; // words

        f << "}";
    }

    // --------------------------------------------------------------------------

    template <DescriptorPolicy TPolicy>
    void TemplatedVocabulary<TPolicy>::load(const cv::FileStorage &fs,
                                            const std::string &name)
    {
        const cv::FileNode vocabulary_node = fs[name];
        if (vocabulary_node.empty() || !vocabulary_node.isMap())
        {
            throw std::runtime_error("Vocabulary storage node is missing or is not a map.");
        }

        const auto read_integer = [](const cv::FileNode &node, const char *key) {
            const cv::FileNode field = node[key];
            if (field.empty() || !field.isInt())
            {
                throw std::runtime_error(
                    std::string("Vocabulary field '") + key + "' must be an integer.");
            }
            return static_cast<int>(field);
        };
        const int branching_factor = read_integer(vocabulary_node, "k");
        const int depth_levels = read_integer(vocabulary_node, "L");
        const int scoring_value = read_integer(vocabulary_node, "scoringType");
        const int weighting_value = read_integer(vocabulary_node, "weightingType");
        if (branching_factor < 2 || depth_levels <= 0 ||
            scoring_value < static_cast<int>(L1_NORM) ||
            scoring_value > static_cast<int>(DOT_PRODUCT) ||
            weighting_value < static_cast<int>(TF_IDF) ||
            weighting_value > static_cast<int>(BINARY))
        {
            throw std::runtime_error("Vocabulary tree parameters are outside supported ranges.");
        }

        const cv::FileNode serialized_nodes = vocabulary_node["nodes"];
        if (serialized_nodes.empty() || !serialized_nodes.isSeq())
        {
            throw std::runtime_error("Vocabulary nodes must be a non-empty sequence.");
        }
        const std::size_t node_count = serialized_nodes.size();
        if (node_count > static_cast<std::size_t>((std::numeric_limits<NodeId>::max)()) ||
            node_count > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        {
            throw std::runtime_error("Vocabulary node count exceeds the NodeId range.");
        }

        // Parse into temporary storage so malformed input cannot partially mutate this vocabulary.
        std::vector<Node> loaded_nodes(node_count + 1U);
        std::vector<bool> seen_nodes(node_count + 1U, false);
        std::vector<std::size_t> node_depths(node_count + 1U, 0U);
        loaded_nodes.front().id = 0;
        seen_nodes.front() = true;
        for (std::size_t index = 0; index < node_count; ++index)
        {
            const cv::FileNode serialized_node =
                serialized_nodes[static_cast<int>(index)];
            if (!serialized_node.isMap())
            {
                throw std::runtime_error("Every vocabulary node must be a map.");
            }

            const int node_id_value = read_integer(serialized_node, "nodeId");
            const int parent_id_value = read_integer(serialized_node, "parentId");
            if (node_id_value <= 0 ||
                static_cast<std::size_t>(node_id_value) > node_count)
            {
                throw std::runtime_error("Vocabulary node ID is outside the serialized node range.");
            }
            if (parent_id_value < 0 ||
                static_cast<std::size_t>(parent_id_value) > node_count)
            {
                throw std::runtime_error("Vocabulary parent ID is outside the serialized node range.");
            }

            const NodeId node_id = static_cast<NodeId>(node_id_value);
            const NodeId parent_id = static_cast<NodeId>(parent_id_value);
            if (seen_nodes[node_id])
            {
                throw std::runtime_error("Vocabulary contains a duplicate node ID.");
            }
            if (!seen_nodes[parent_id])
            {
                throw std::runtime_error(
                    "Vocabulary parent must precede its child and connect to the root.");
            }

            const cv::FileNode weight_node = serialized_node["weight"];
            const cv::FileNode descriptor_node = serialized_node["descriptor"];
            if (weight_node.empty() || (!weight_node.isInt() && !weight_node.isReal()) ||
                descriptor_node.empty() || !descriptor_node.isString())
            {
                throw std::runtime_error(
                    "Vocabulary node weight and descriptor have invalid storage types.");
            }
            const WordValue weight = static_cast<double>(weight_node);
            if (!std::isfinite(weight) || weight < 0.0)
            {
                throw std::runtime_error(
                    "Vocabulary node weight must be finite and nonnegative.");
            }

            Node &node = loaded_nodes[node_id];
            node.id = node_id;
            node.parent = parent_id;
            node.weight = weight;
            if (!TPolicy::Deserialize(static_cast<std::string>(descriptor_node),
                                      node.descriptor))
            {
                throw std::runtime_error("Invalid descriptor for vocabulary node " +
                                         std::to_string(node_id));
            }

            const std::size_t node_depth = node_depths[parent_id] + 1U;
            if (node_depth > static_cast<std::size_t>(depth_levels))
            {
                throw std::runtime_error("Vocabulary node exceeds the declared tree depth.");
            }
            node_depths[node_id] = node_depth;
            loaded_nodes[parent_id].children.push_back(node_id);
            if (loaded_nodes[parent_id].children.size() >
                static_cast<std::size_t>(branching_factor))
            {
                throw std::runtime_error(
                    "Vocabulary node exceeds the declared branching factor.");
            }
            seen_nodes[node_id] = true;
        }

        const cv::FileNode serialized_words = vocabulary_node["words"];
        if (serialized_words.empty() || !serialized_words.isSeq())
        {
            throw std::runtime_error("Vocabulary words must be a non-empty sequence.");
        }
        const std::size_t word_count = serialized_words.size();
        if (word_count > static_cast<std::size_t>((std::numeric_limits<WordId>::max)()) ||
            word_count > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        {
            throw std::runtime_error("Vocabulary word count exceeds the WordId range.");
        }

        std::vector<NodeId> word_nodes(word_count, 0U);
        std::vector<bool> seen_words(word_count, false);
        std::vector<bool> node_has_word(node_count + 1U, false);
        for (std::size_t index = 0; index < word_count; ++index)
        {
            const cv::FileNode serialized_word =
                serialized_words[static_cast<int>(index)];
            if (!serialized_word.isMap())
            {
                throw std::runtime_error("Every vocabulary word must be a map.");
            }

            const int word_id_value = read_integer(serialized_word, "wordId");
            const int node_id_value = read_integer(serialized_word, "nodeId");
            if (word_id_value < 0 ||
                static_cast<std::size_t>(word_id_value) >= word_count)
            {
                throw std::runtime_error("Vocabulary word ID is outside the serialized word range.");
            }
            if (node_id_value <= 0 ||
                static_cast<std::size_t>(node_id_value) > node_count)
            {
                throw std::runtime_error("Vocabulary word node ID is outside the node range.");
            }

            const WordId word_id = static_cast<WordId>(word_id_value);
            const NodeId node_id = static_cast<NodeId>(node_id_value);
            if (seen_words[word_id] || node_has_word[node_id])
            {
                throw std::runtime_error("Vocabulary word or word-node ID is duplicated.");
            }
            if (!loaded_nodes[node_id].isLeaf())
            {
                throw std::runtime_error("Vocabulary words must reference leaf nodes.");
            }

            loaded_nodes[node_id].word_id = word_id;
            word_nodes[word_id] = node_id;
            seen_words[word_id] = true;
            node_has_word[node_id] = true;
        }

        for (std::size_t node_index = 1; node_index <= node_count; ++node_index)
        {
            if (loaded_nodes[node_index].isLeaf() != node_has_word[node_index])
            {
                throw std::runtime_error(
                    "Vocabulary leaf and word-node sets do not match.");
            }
        }

        TemplatedVocabulary loaded_vocabulary(
            branching_factor, depth_levels,
            static_cast<WeightingType>(weighting_value),
            static_cast<ScoringType>(scoring_value));
        loaded_vocabulary.m_nodes = std::move(loaded_nodes);
        loaded_vocabulary.m_words.resize(word_count);
        for (std::size_t word_id = 0; word_id < word_count; ++word_id)
        {
            loaded_vocabulary.m_words[word_id] =
                &loaded_vocabulary.m_nodes[word_nodes[word_id]];
        }

        // Publish only after every allocation and pointer relationship is complete.
        *this = std::move(loaded_vocabulary);
    }

    // --------------------------------------------------------------------------

    /**
     * Writes printable information of the vocabulary
     * @param os stream to write to
     * @param voc
     */
    template <DescriptorPolicy TPolicy>
    std::ostream &operator<<(std::ostream &os,
                             const TemplatedVocabulary<TPolicy> &voc)
    {
        os << "Vocabulary: k = " << voc.getBranchingFactor()
           << ", L = " << voc.getDepthLevels()
           << ", Weighting = ";

        switch (voc.getWeightingType())
        {
        case TF_IDF:
            os << "tf-idf";
            break;
        case TF:
            os << "tf";
            break;
        case IDF:
            os << "idf";
            break;
        case BINARY:
            os << "binary";
            break;
        }

        os << ", Scoring = ";
        switch (voc.getScoringType())
        {
        case L1_NORM:
            os << "L1-norm";
            break;
        case L2_NORM:
            os << "L2-norm";
            break;
        case CHI_SQUARE:
            os << "Chi square distance";
            break;
        case KL:
            os << "KL-divergence";
            break;
        case BHATTACHARYYA:
            os << "Bhattacharyya coefficient";
            break;
        case DOT_PRODUCT:
            os << "Dot product";
            break;
        }

        os << ", Number of words = " << voc.size();

        return os;
    }

} // namespace DBoW2

#endif
