DBoW2
=====

DBoW2 is an improved version of the DBow library, an open source C++ library for indexing and converting images into a bag-of-word representation. It implements a hierarchical tree for approximating nearest neighbours in the image feature space and creating a visual vocabulary. DBoW2 also implements an image database with inverted and direct files to index images and enabling quick queries and feature comparisons. The main differences with the previous DBow library are:

  * DBoW2 classes are templated, so it can work with any type of descriptor.
  * DBoW2 is shipped with policies for ORB, BRIEF, SURF64, and SIFT descriptors.
  * DBoW2 adds a direct file to the image database to do fast feature comparison. This is used by DLoopDetector.
  * DBoW2 does not use a binary format any longer. On the other hand, it uses the OpenCV storage system to save vocabularies and databases. This means that these files can be stored as plain text in YAML format, making compatibility easier, or compressed in gunzip format (.gz) to reduce disk usage.
  * Some pieces of code have been rewritten to optimize speed. The interface of DBoW2 has been simplified.
  * For performance reasons, DBoW2 does not support stop words.

This fork requires C++20 and OpenCV 4.2 or newer. BRIEF uses `std::bitset` and has no Boost
dependency.

DBoW2, along with DLoopDetector, has been tested on several real datasets, yielding an execution time of 3 ms to convert the BRIEF features of an image into a bag-of-words vector and 5 ms to look for image matches in a database with more than 19000 images.

## Citing

If you use this software in an academic work, please cite:

    @ARTICLE{GalvezTRO12,
      author={G\'alvez-L\'opez, Dorian and Tard\'os, J. D.},
      journal={IEEE Transactions on Robotics},
      title={Bags of Binary Words for Fast Place Recognition in Image Sequences},
      year={2012},
      month={October},
      volume={28},
      number={5},
      pages={1188--1197},
      doi={10.1109/TRO.2012.2197158},
      ISSN={1552-3098}
    }
}

## Usage notes

### Weighting and Scoring

DBoW2 implements the same weighting and scoring mechanisms as DBow. Check them here. The only difference is that DBoW2 scales all the scores to [0..1], so that the scaling flag is not used any longer.

### Save & Load

All vocabularies and databases can be saved to and load from disk with the save and load member functions. When a database is saved, the vocabulary it is associated with is also embedded in the file, so that vocabulary and database files are completely independent.

You can also add the vocabulary or database data to any file opened with a `cv::FileStorage` structure.

You can save the vocabulary or the database with any file extension. If you use .gz, the file is automatically compressed (OpenCV behaviour).

## Implementation notes

### Template parameters

DBoW2 has two main classes: `TemplatedVocabulary` and `TemplatedDatabase`. These implement the visual vocabulary to convert images into bag-of-words vectors and the database to index images. These classes are templated:

    template<DescriptorPolicy TPolicy>
    class TemplatedVocabulary
    {
      ...
    };

    template<DescriptorPolicy TPolicy>
    class TemplatedDatabase
    {
      ...
    };

A single policy type provides the descriptor representation and the static operations required by
DBoW2. The C++20 `DescriptorPolicy` concept checks the associated descriptor/scalar types,
compile-time shape and representation metadata, default/copy semantics, independent ownership via
`Clone`, centroid, distance, and strict text persistence. `Clone` is intentionally distinct from a
copy expression because view-backed descriptors such as `cv::Mat` require a deep copy.
`LocalDescriptorPolicy` additionally requires conversion to a row-major OpenCV float matrix for
classical local-feature matching.

For example, `FORB::Descriptor` is a `cv::Mat` single row containing 32 8-bit values. When features
are extracted from an image, callers provide a
`std::vector<typename TPolicy::Descriptor>`. `FBrief`, `FSurf64`, and `FSIFT` provide equivalent
policies for their historical descriptor representations.

Applications can define new policy types without editing a feature-name registry or inheriting from
a marker base. Generic `BitsetDescriptorPolicy<N>`, `FixedFloatDescriptorPolicy<N>`, and
`VectorFloatDescriptorPolicy<N>` utilities cover the built-in storage and distance strategies.

### Predefined Vocabularies and Databases

To make it easier to use, this fork defines predefined vocabularies and
databases for the built-in descriptor helpers: `OrbVocabulary`,
`OrbDatabase`, `BriefVocabulary`, `BriefDatabase`, `Surf64Vocabulary`,
`Surf64Database`, `SiftVocabulary`, and `SiftDatabase`. Please, check the
demo application to see how they are created and used.
