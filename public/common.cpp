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

std::vector<uint32_t> broadcast(const std::vector<uint32_t> &shape1, const std::vector<uint32_t> &shape2) {
  bool GT = shape1.size() > shape2.size();
  const std::vector<uint32_t>* p1= &shape1;
  const std::vector<uint32_t>* p2= &shape2;
  if (!GT) {
    p1 = &shape2;
    p2= &shape1;
  }
  std::vector<uint32_t> out_shape(p1->size());
  int diff = std::abs(p1->size() - p2->size());
  for (int i = 0; i < diff; ++i) {
    out_shape[i] = (*p1)[i];
  }
  for (int i1 = diff, i2 = 0; i1 < p1->size(); ++i1, ++i2) {
    uint32_t s1 = (*p1)[i1];
    uint32_t s2 = (*p2)[i2];
    if (s1 != s1)
      CHECK(s1 == 1 || s2 == 1);
    out_shape[i1] = std::max(s1, s2);
  }
  return out_shape;
}

template<typename Dtype>
std::ostream &operator<<(std::ostream &out, std::vector<Dtype> vector) {
  out << "[";
  for (int i = 0; i < vector.size(); ++i) {
    out << vector[i] << " ";
  }
  out << "]" << std::endl;
  return out;
}
template std::ostream &operator<< <int>(std::ostream &out, std::vector<int> vector);
template std::ostream &operator<< <float>(std::ostream &out, std::vector<float> vector);
template std::ostream &operator<< <double>(std::ostream &out, std::vector<double> vector);
template std::ostream &operator<< <uint32_t>(std::ostream &out, std::vector<uint32_t> vector);
template std::ostream &operator<< <bool>(std::ostream &out, std::vector<bool> vector);
template std::ostream &operator<< <std::string>(std::ostream &out, std::vector<std::string> vector);

}//namespace stensor
