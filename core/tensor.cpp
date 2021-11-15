#include <public/synmem.hpp>
#include <utility>
#include "tensor.hpp"
#include "proto/tensor.pb.h"
#include "math.hpp"

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
void Tensor::ToProto(TensorProto &proto, bool write_grad) const{
  ToProto(&proto, write_grad);
}

void Tensor::FromProto(const TensorProto *proto, bool reshape) {
  // 1. reshape
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
    for (int i = 0; i < _size; ++i) {
      data_vec[i] = proto->data(i);
    }
  }
  // 3. grad
  // copy grad
  if (proto->grad_size() > 0) {
    CHECK_EQ(_size, proto->grad_size()) << "gradiant size mismatch.";
    Dtype *grad_vec = mutable_cpu_grad();
    for (int i = 0; i < _size; ++i) {
      grad_vec[i] = proto->grad(i);
    }
  }
  // 4. name
  _size = proto->size();
  _require_grad = proto->require_grad();
  _name = proto->name();
  // 5. neighbors & operations
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
    out << data[i] << " ";
    for (int j = dim - 1; j >= 0; --j) {
      if (((i + 1) % shape[j]) == 0) {
        out << "]";
        if ((i + 2) < count) out << "\n[";
      } else break;
    }

  }
  out << "\nshape=" << this->shape_string() << std::endl;
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

/* math of Tensor */
//TODO:GPU Mode
void set(Tensor &tensor, float val) {
  float *data = tensor.mutable_cpu_data();
  stensor::set(tensor.size(), val, data);
}

void set(Tensor *tensor, float val) {
  float *data = tensor->mutable_cpu_data();
  stensor::set(tensor->size(), val, data);
}

Tensor *add(Tensor &tensor, float val, bool inplace) {
  if (inplace) {
    float *data = tensor.mutable_cpu_data();
    stensor::add(tensor.size(), data, val, data);
    return &tensor;
  } else {
    const float *data = tensor.cpu_data();
    Tensor *out_tensor = new Tensor(tensor, tensor.require_grad());
    float *out_data = out_tensor->mutable_cpu_data();
    stensor::add(out_tensor->size(), data, val, out_data);
    return out_tensor;
  }
}

Tensor *sub(Tensor &tensor, float val, bool inplace) {
  if (inplace) {
    float *data = tensor.mutable_cpu_data();
    stensor::sub(tensor.size(), data, val, data);
    return &tensor;
  } else {
    const float *data = tensor.cpu_data();
    Tensor *out_tensor = new Tensor(tensor, tensor.require_grad());
    float *out_data = out_tensor->mutable_cpu_data();
    stensor::sub(out_tensor->size(), data, val, out_data);
    return out_tensor;
  }
}
Tensor *scale(Tensor &tensor, float val, bool inplace) {
  if (inplace) {
    float *data = tensor.mutable_cpu_data();
    stensor::scale(tensor.size(), data, val, data);
    return &tensor;
  } else {
    const float *data = tensor.cpu_data();
    Tensor *out_tensor = new Tensor(tensor, tensor.require_grad());
    float *out_data = out_tensor->mutable_cpu_data();
    stensor::scale(out_tensor->size(), data, val, out_data);
    return out_tensor;
  }
}

Tensor *pow(Tensor &tensor, float val, bool inplace) {
  if (inplace) {
    float *data = tensor.mutable_cpu_data();
    stensor::pow(tensor.size(), data, val, data);
    return &tensor;
  } else {
    const float *data = tensor.cpu_data();
    Tensor *out_tensor = new Tensor(tensor, tensor.require_grad());
    float *out_data = out_tensor->mutable_cpu_data();
    stensor::pow(out_tensor->size(), data, val, out_data);
    return out_tensor;
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


// Tensor op Tensor
// TODO:broadcast op

Tensor *add(const Tensor &a, const Tensor &b){
  const Tensor::ShapeType out_shape = stensor::broadcast(a.shape(), b.shape());
  bool require_grad = (a.require_grad()||b.require_grad());
  Tensor *out_tensor = new Tensor(out_shape, require_grad);
  NOT_IMPLEMENTED;
  return out_tensor;

}
Tensor *sub(const Tensor &a, const Tensor &b){
  NOT_IMPLEMENTED;
}
Tensor *scale(const Tensor &a, const Tensor &b){
  NOT_IMPLEMENTED;
}
Tensor *div(const Tensor &a, const Tensor &b){
  NOT_IMPLEMENTED;
}
Tensor *exp(const Tensor &a, const Tensor &b){
  NOT_IMPLEMENTED;
}

/* math of Tensor end */

/* Tensor Generator*/

Tensor *random(const Tensor::ShapeType &shape, bool require_grad, float a, float b) {
  Tensor *new_t = new Tensor(shape, require_grad);
  stensor::rng_uniform<Tensor::Dtype>(new_t->size(), Tensor::Dtype(a), Tensor::Dtype(b), new_t->mutable_cpu_data());
  return new_t;
}
Tensor *random(const Tensor::ShapeType &shape, bool require_grad) {
  return random(shape, require_grad, 0.0, 1.0);
}
Tensor *random(const Tensor::ShapeType &shape, float a, float b) {
  return random(shape, false, a, b);
}

Tensor *random_gaussian(const Tensor::ShapeType &shape, bool require_grad, float mu, float sigma) {
  Tensor *new_t = new Tensor(shape, require_grad);
  stensor::rng_gaussian<Tensor::Dtype>(new_t->size(),
                                       Tensor::Dtype(mu),
                                       Tensor::Dtype(sigma),
                                       new_t->mutable_cpu_data());
  return new_t;
}

Tensor *random_gaussian(const Tensor::ShapeType &shape, bool require_grad) {
  return random_gaussian(shape, require_grad, 0.0, 1.0);
}
Tensor *random_gaussian(const Tensor::ShapeType &shape, float mu, float sigma) {
  return random_gaussian(shape, false, mu, sigma);
}
Tensor *constants(const Tensor::ShapeType &shape, Tensor::Dtype val, bool require_grad) {
  stensor::Tensor *new_t = new Tensor(shape, require_grad);
  stensor::set(new_t, val);
  return new_t;
}

Tensor *zeros(const Tensor::ShapeType &shape, bool require_grad) {
  return constants(shape, 0.0, require_grad);
}

Tensor *ones(const Tensor::ShapeType &shape, bool require_grad) {
  return constants(shape, 1.0, require_grad);
}
Tensor *constants_like(Tensor *other, Tensor::Dtype val, bool require_grad) {
  Tensor *new_t = new Tensor(other, require_grad);
  stensor::set(new_t, val);
  return new_t;
}

Tensor *zeros_like(Tensor *other, bool require_grad) {
  return constants_like(other, 0.0, require_grad);
}

Tensor *ones_like(Tensor *other, bool require_grad) {
  return constants_like(other, 1.0, require_grad);
}

/* Tensor Generator end*/

}//namespace stensor
