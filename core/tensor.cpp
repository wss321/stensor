#include <public/synmem.hpp>
#include "tensor.hpp"
#include "proto/tensor.pb.h"
#include "math_base_cpu.hpp"

namespace stensor {

void Tensor::CopyFrom(const Tensor &source, bool copy_grad, bool reshape, Mode mode) {
  CopyFrom(&source, copy_grad, reshape, mode);
}

void Tensor::CopyFrom(const Tensor *source, bool copy_grad, bool reshape, Mode mode) {
  if (source->size() != _size || source->shape() != _shape) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to cpu_copy Tensors of different sizes.";
    }
  }
  // TODO:CopyFrom CPU
  switch (mode) {
    case stensor::CPU:
      if (copy_grad) {
        cpu_copy(_size, source->cpu_grad(), static_cast<Dtype *>(_grad->mutable_cpu_data()));
      } else {
        cpu_copy(_size, source->cpu_data(), static_cast<Dtype *>(_data->mutable_cpu_data()));
      }
      break;
    default:LOG(FATAL) << "Unknown device mode.";
  }
}

void Tensor::Reshape(const ShapeType &shape) {
  CHECK_LE(shape.size(), kMaxTensorAxes);
  _size = 1;
  _shape.resize(shape.size());

  for (int i = 0; i < shape.size(); ++i) {
    if (_size != 0) {
      CHECK_LE(shape[i], INT_MAX / _size)
        << "Tensor size exceeds INT_MAX";
    }
    _size *= shape[i];
    _shape[i] = shape[i];
  }
  if (_size > _capacity) {
    _capacity = _size;
    _data.reset(new SynMem(_capacity * sizeof(Dtype)));
    if (_require_grad)
      _grad.reset(new SynMem(_capacity * sizeof(Dtype)));
  }
}

void Tensor::ReshapeLike(const Tensor &other) {
  Reshape(other.shape());
}
void Tensor::ReshapeLike(const Tensor *other) {
  Reshape(other->shape());
}

void Tensor::flatten() {
  Reshape(ShapeType{_size});
}

Tensor::Tensor(const ShapeType &shape, bool require_grad) :
    _capacity(0ul), _require_grad(require_grad) {
  Reshape(shape);
}

bool Tensor::ShapeEquals(const Tensor *other) {
  ShapeType shapeOther;
  stensor::RepeatTypeToVector(other->shape(), shapeOther);
  if (_shape.size() != shapeOther.size()) return false;
  for (int i = 0; i < _shape.size(); ++i) {
    if (_shape[i] != shapeOther[i]) return false;
  }
  return true;
}

bool Tensor::ShapeEquals(const Tensor &other) {
  return ShapeEquals(&other);
}
bool Tensor::ShapeEquals(const TensorProto *other) {
  ShapeType shapeOther;
  stensor::RepeatTypeToVector(other->shape(), shapeOther);
  if (_shape.size() != shapeOther.size()) return false;
  for (int i = 0; i < _shape.size(); ++i) {
    if (_shape[i] != shapeOther[i]) return false;
  }
  return true;
}

bool Tensor::ShapeEquals(const TensorProto &other) {
  return ShapeEquals(&other);
}

void Tensor::ToProto(TensorProto *proto, bool write_grad) const {
  proto->clear_shape();
  // 1. shape
  for (int i = 0; i < _shape.size(); ++i) {
    proto->add_shape(_shape[i]);
  }
  proto->clear_data();
  proto->clear_grad();
  // 2. data
  const float *data_vec = cpu_data();
  for (int i = 0; i < _size; ++i) {
    proto->add_data(data_vec[i]);
  }
  // 3. grad
  if (write_grad) {
    const float *grad_vec = cpu_grad();
    for (int i = 0; i < _size; ++i) {
      proto->add_grad(grad_vec[i]);
    }
  }
  // 4. size & name & isparam
  proto->set_size(_size);
  proto->set_name(_name);
  proto->set_require_grad(_require_grad);
  // 5. neighbor & operations
  proto->clear_neighbors();
  proto->clear_operations();
  for (int i = 0; i < _neighbors.size(); ++i) {
    proto->add_neighbors(_neighbors[i]);
    proto->add_operations(_operations[i]);
  }
}
void Tensor::ToProto(TensorProto &proto, bool write_grad) const {
  ToProto(&proto, write_grad);
}

void Tensor::FromProto(const TensorProto *proto, bool reshape) {
  // 1. reshape
  _size = proto->size();
  _require_grad = proto->require_grad();
  _name = proto->name();
  if (reshape) {
    Tensor::ShapeType new_shape;
    stensor::RepeatTypeToVector(proto->shape(), new_shape);
    Reshape(new_shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch.";
  }
  // 2. data
  // copy data
  Dtype *data_vec = mutable_cpu_data();
  if (proto->data_size() > 0) {
    CHECK_EQ(_size, proto->data_size()) << "data size mismatch.";
//    for (int i = 0; i < _size; ++i) {
//      data_vec[i] = proto->data(i);
//    }
    stensor::cpu_copy(_size, proto->data().data(), data_vec);
  }
  // 3. grad
  // copy grad
  if (proto->grad_size() > 0) {
    CHECK_EQ(_size, proto->grad_size()) << "gradiant size mismatch.";
    Dtype *grad_vec = mutable_cpu_grad();
//    for (int i = 0; i < _size; ++i) {
//      grad_vec[i] = proto->grad(i);
//    }
    stensor::cpu_copy(_size, proto->grad().data(), grad_vec);
  }
  // 4. neighbors & operations
  stensor::RepeatTypeToVector(proto->neighbors(), _neighbors);
  stensor::RepeatTypeToVector(proto->operations(), _operations);
}

void Tensor::FromProto(const TensorProto &proto, bool reshape) {
  FromProto(&proto, reshape);
}

std::string Tensor::data_string() const {
  std::ostringstream out;
  const Tensor::ShapeType shape = this->shape();
  const Tensor::Dtype *data = this->cpu_data();
  int dim = this->num_axes();
  int count = this->size();
  for (int i = 0; i < dim; ++i) {
    out << "[";
  }
  for (int i = 0; i < count; ++i) {
    out << *data++ << " ";
    int n_bracket = 0;
    int acc = 1;
    for (int j = dim - 1; j >= 0; --j) {
      acc *= static_cast<int>(shape[j]);
      if (((i + 1) % acc) == 0) {
        out << "]";
        n_bracket++;
      } else break;
    }
    if (n_bracket >= 1)
      out << "\n";
    if ((i + 2) <= count) {
      for (int j = 0; j < n_bracket; ++j) {
        out << "[";
      }
    }
  }
  out << "{shape:" << this->shape_string() << ", dtype:"
      << abi::__cxa_demangle(typeid(Dtype).name(), nullptr, nullptr, nullptr)
      << "}" << std::endl;
  return out.str();
}

std::ostream &operator<<(std::ostream &out, const Tensor *tensor) {
  out << tensor->data_string();
  return out;
}

std::ostream &operator<<(std::ostream &out, const Tensor &tensor) {
  return stensor::operator<<(out, &tensor);
}

Tensor::Tensor(const Tensor &other, bool require_grad) :
    _data(),
    _grad(), _size(0ul),
    _capacity(0ul), _name(),
    _require_grad(require_grad) {
  CopyFrom(other, false, true, other.state());
}

Tensor::Tensor(const Tensor *other, bool require_grad) :
    _data(),
    _grad(), _size(0ul),
    _capacity(0ul), _name(),
    _require_grad(require_grad) {

  CopyFrom(other, false, true, other->state());
}

//TODO:GPU Mode
Tensor::Tensor(const std::vector<Tensor::Dtype> &other,
               const ShapeType &shape, bool require_grad,
               const Mode mode) :
    _data(),
    _grad(),
    _size(0ul),
    _capacity(0ul),
    _name(),
    _require_grad(require_grad) {
  CHECK(!other.empty()) << "Attempting assign an empty vector to tensor";
  uint32_t num = stensor::count(shape);
  CHECK_EQ(other.size(), num) << "Shape mismatch";
  Reshape(shape);
  Tensor::Dtype *data = mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
    data[i] = other[i];
  }
}
//TODO:operater overload
Tensor &Tensor::operator=(const Tensor &other) {
  CopyFrom(other, false, true, other.state());
  return (*this);
}
Tensor &Tensor::operator=(const Tensor *other) {
  CopyFrom(other, false, true, other->state());
  return (*this);
}

Tensor::Dtype Tensor::operator[](std::vector<int> indices) const {
  CHECK_EQ(indices.size(), num_axes()) << "indices size must be equal with num axes";
  Tensor::ShapeType canonicalIndex(num_axes(), 0);
  for (int i = 0; i < num_axes(); ++i) {
    if (indices[i] < 0) {
      CHECK_GE(indices[i], shape(i));
      canonicalIndex[i] = indices[i] + shape(i);
    } else canonicalIndex[i] = indices[i];
  }

  int out = 1;
  for (int i = 0; i < indices.size(); ++i) {
    out *= static_cast<int>(canonicalIndex[i] + 1);
  }
  return cpu_data()[out - 1];
}

/* Tensor Generator end*/
/* save and load*/
void save(const Tensor *tensor, const std::string &path) {
  TensorProto proto;
  tensor->ToProto(proto);
  std::fstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);
  bool success = proto.SerializeToOstream(&output);
  CHECK(success) << "Failed to save tensor to " << path;
}
Tensor *load(const std::string &path) {
  TensorProto proto;
  std::fstream input(path, std::ios::in | std::ios::binary);
  bool success = proto.ParseFromIstream(&input);
  CHECK(success) << "Failed to load tensor from " << path;
  Tensor *new_tensor = new Tensor();
  new_tensor->FromProto(proto);
  return new_tensor;
}
/* save and load end*/
}//namespace stensor
