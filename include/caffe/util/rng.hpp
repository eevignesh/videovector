#ifndef CAFFE_RNG_CPP_HPP_
#define CAFFE_RNG_CPP_HPP_

#include <algorithm>
#include <iterator>

#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"

#include "caffe/common.hpp"

namespace caffe {

typedef boost::mt19937 rng_t;

inline rng_t* caffe_rng() {
  return static_cast<caffe::rng_t*>(Caffe::rng_stream().generator());
}

// Fisher–Yates algorithm
template <class RandomAccessIterator, class RandomGenerator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end,
                    RandomGenerator* gen) {
  typedef typename std::iterator_traits<RandomAccessIterator>::difference_type
      difference_type;
  typedef typename boost::uniform_int<difference_type> dist_type;

  difference_type length = std::distance(begin, end);
  if (length <= 0) return;

  for (difference_type i = length - 1; i > 0; --i) {
    dist_type dist(0, i);
    std::iter_swap(begin + i, begin + dist(*gen));
  }
}

template <class RandomAccessIterator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end) {
  shuffle(begin, end, caffe_rng());
}

// For random selection of elements
template<class RandomAccessIterator>
RandomAccessIterator random_unique(RandomAccessIterator first, RandomAccessIterator last, int num_random) {
    int left = std::distance(first, last);
    while (num_random--) {
        RandomAccessIterator r = first;
        std::advance(r, rand()%left);
        std::swap(*first, *r);
        ++first;
        --left;
    }
    return first;
}

}  // namespace caffe

#endif  // CAFFE_RNG_HPP_
