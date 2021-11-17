/**
* Copyright 2021 wss
* Created by wss on 11月,17, 2021
*/
#ifndef STENSOR_PUBLIC_STENSOR_RANDOM_HPP_
#define STENSOR_PUBLIC_STENSOR_RANDOM_HPP_
#include <boost/random.hpp>
#include <memory>
#include <glog/logging.h>
//#include <random>
namespace stensor{
typedef boost::mt19937 rng_t;
int64_t cluster_seed_gen();
class RNG {
 public:
  RNG();
  explicit RNG(unsigned int seed);
  RNG &operator=(const RNG &);
  void *generator();
 private:
  class Generator;
  std::shared_ptr<Generator> generator_;
};

}
#endif //STENSOR_PUBLIC_STENSOR_RANDOM_HPP_
