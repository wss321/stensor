/**
* Copyright 2021 wss
* Created by wss on 11月,17, 2021
*/
#include "stensor_random.hpp"

namespace stensor {

typedef boost::mt19937 rng_t;

// random seeding
int64_t cluster_seed_gen() {
  int64_t s, seed, pid;
  FILE *f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }
  LOG(INFO) << "System entropy source not available, "
               "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);

  pid = getpid();
  s = time(nullptr);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}

class RNG::Generator {
 public:
  Generator() : rng_(new stensor::rng_t(cluster_seed_gen())) {}
  explicit Generator(unsigned int seed) : rng_(new stensor::rng_t(seed)) {}
  stensor::rng_t *rng() { return rng_.get(); }
 private:
  std::shared_ptr<stensor::rng_t> rng_;
};

RNG::RNG() : generator_(new Generator()) {}
RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) {}

RNG &RNG::operator=(const RNG &other) {
  generator_.reset(other.generator_.get());
  return *this;
}

void *RNG::generator() {
  return static_cast<void *>(generator_->rng());
}

}