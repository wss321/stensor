#include "common.hpp"

namespace stensor {

uint32_t count(const std::vector<uint32_t> &shape) {
  if (shape.empty()) return 0;
  uint32_t num = 1;
  for (int i = 0; i < shape.size(); ++i) {
    num *= shape[i];
  }
  return num;
}

template<typename Dshape>
std::vector<Dshape> broadcast(std::vector<Dshape> &shape1, std::vector<Dshape> &shape2) {
  bool GT = shape1.size() > shape2.size();
  const std::vector<Dshape> *p1 = &shape1;
  const std::vector<Dshape> *p2 = &shape2;
  if (!GT) {
    p1 = &shape2;
    p2 = &shape1;
  }
  std::vector<Dshape> out_shape(p1->size());
  int diff = std::abs(p1->size() - p2->size());
  for (int i = 0; i < diff; ++i) {
    out_shape[i] = (*p1)[i];
  }
  for (int i1 = diff, i2 = 0; i1 < p1->size(); ++i1, ++i2) {
    Dshape s1 = (*p1)[i1];
    Dshape s2 = (*p2)[i2];
    if (s1 != s2)
      CHECK(s1 == 1 || s2 == 1)
              << "Cannot broadcast between tensors with shape " << shape1 << " and " << shape2;
    out_shape[i1] = std::max(s1, s2);
  }
  // change shape
  if (!GT) {
    std::vector<Dshape> temp(out_shape.size());
    for (int i = 0; i < diff; ++i)
      temp[i] = 1;
    for (int i = diff, j = 0; i < temp.size() && j < shape1.size(); ++i, ++j) {
      temp[i] = shape1[j];
    }
    shape1 = temp;
  } else {
    std::vector<Dshape> temp(out_shape.size());
    for (int i = 0; i < diff; ++i)
      temp[i] = 1;
    for (int i = diff, j = 0; i < temp.size() && j < shape1.size(); ++i, ++j) {
      temp[i] = shape2[j];
    }
    shape2 = temp;
  }
  return out_shape;
}
template std::vector<int> broadcast(std::vector<int> &shape1, std::vector<int> &shape2);
template std::vector<uint32_t> broadcast(std::vector<uint32_t> &shape1, std::vector<uint32_t> &shape2);

template<typename Dtype>
std::ostream &operator<<(std::ostream &out, std::vector<Dtype> vector) {
  out << "[";
  for (int i = 0; i < vector.size(); ++i) {
    out << vector[i] << " ";
  }
  out << "]";
  return out;
}
template std::ostream &operator<<<int>(std::ostream &out, std::vector<int> vector);
template std::ostream &operator<<<float>(std::ostream &out, std::vector<float> vector);
template std::ostream &operator<<<double>(std::ostream &out, std::vector<double> vector);
template std::ostream &operator<<<uint32_t>(std::ostream &out, std::vector<uint32_t> vector);
template std::ostream &operator<<<bool>(std::ostream &out, std::vector<bool> vector);
template std::ostream &operator<<<std::string>(std::ostream &out, std::vector<std::string> vector);

// Make sure each thread can have different values.
static boost::thread_specific_ptr<Config> thread_instance_;
Config &Config::GetInstance() {
  if (!thread_instance_.get()) {
    thread_instance_.reset(new Config());
  }
  return *(thread_instance_.get());
}

Config::Config()
    : cublas_handle_(nullptr),
      curand_generator_(nullptr),
      random_generator_(),
      multiprocess_(false) {
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
  if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
  }
  // Try to create a curand handler.
  if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
      != CURAND_STATUS_SUCCESS ||
      curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seed_gen())
          != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Curand generator. Curand won't be available.";
  }
}

Config::~Config() {
  if (cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  }
}

void Config::set_random_seed(const unsigned int seed) {
  // curand seed
  static bool g_curand_availability_logged = false;
  if (GetInstance().curand_generator_) {
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(),
                                                    seed));
    CURAND_CHECK(curandSetGeneratorOffset(curand_generator(), 0));
  } else {
    if (!g_curand_availability_logged) {
      LOG(ERROR) <<
                 "Curand not available. Skipping setting the curand seed.";
      g_curand_availability_logged = true;
    }
  }
  // RNG seed
  GetInstance().random_generator_.reset(new RNG(seed));
}

}//namespace stensor
