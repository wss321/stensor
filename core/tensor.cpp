#include <public/synmem.hpp>
#include "tensor.hpp"
#include "proto/tensor.pb.h"
#include "math/math_base_cpu.hpp"

namespace stensor {

template<typename Dtype>
void Tensor<Dtype>::update_state() {
  if (_data) {
    if (_data->gpu_data()) {
      _current_data = static_cast<Dtype *>(_data->gpu_data());
      _device = _data->device();
    } else if (_data->cpu_data()) {
      _current_data = static_cast<Dtype *>(_data->cpu_data());
      _device = -1;
    }
    _size = _data->size() / sizeof(Dtype);
  } else {
    _current_data = nullptr;
    _capacity = 0;
  }
  if (_grad) {
    if (_grad->gpu_data()) {
      _current_grad = static_cast<Dtype *>(_grad->gpu_data());
      _device = _grad->device();
    } else if (_grad->cpu_data()) {
      _current_grad = static_cast<Dtype *>(_grad->cpu_data());
      _device = -1;
    }
    if (size() == 0) _size = _grad->size() / sizeof(Dtype);
  } else _current_grad = nullptr;
}

template<typename Dtype>
void Tensor<Dtype>::to_cpu() {
  if (_capacity != 0) {
    if (!_data->cpu_data())
      _data->alloc_cpu();
    CHECK(_data->gpu_data()) << "GPU data is none";
    _data->copy_gpu_to_cpu();
    _data->free_gpu();
    if (require_grad()) {
      if (!_grad->cpu_data())
        _grad->alloc_cpu();
      CHECK(_grad->gpu_data()) << "GPU grad is none";
      _grad->copy_gpu_to_cpu();
      _grad->free_gpu();
    }
  }
  update_state();
}

template<typename Dtype>
void Tensor<Dtype>::to_gpu() {
  if (_capacity != 0) {
    if (!_data->gpu_data())
      _data->alloc_gpu();
    CHECK(_data->cpu_data()) << "CPU data is none";
    _data->copy_cpu_to_gpu();
    _data->free_cpu();
    if (require_grad()) {
      if (!_grad->gpu_data())
        _grad->alloc_gpu();
      CHECK(_grad->cpu_data()) << "CPU grad is none";
      _grad->copy_cpu_to_gpu();
      _grad->free_cpu();
    }
  }
  update_state();
}

template<typename Dtype>
void Tensor<Dtype>::copy_from(const Tensor<Dtype> &source, bool copy_grad, bool reset) {
  copy_from(&source, copy_grad, reset);
}

template<typename Dtype>
void Tensor<Dtype>::copy_data_from(const Tensor<Dtype> *other, bool reset) {
  if (other->size() != _size) {
    if (!reset)
      LOG(FATAL) << "Trying to copy tensor of different sizes."
                 << _size << " vs " << other->size();
    Reset(other->shape());
  }
  switch (state()) {
    case stensor::CPU:
      if (other->state() == stensor::GPU)
        LOG(FATAL) << "Trying to copy tensor of different sizes."
                   << "CPU" << " vs " << "GPU:" << other->device();
      cpu_copy(_size, other->const_data(), data());
      break;
    case stensor::GPU:
      if (other->state() == stensor::CPU)
        LOG(FATAL) << "Trying to copy tensor of different sizes."
                   << "GPU:" << device() << " vs " << "CPU";
      gpu_copy(_size, other->const_data(), grad());
      break;
    default:LOG(FATAL) << "Unknown device mode.";
  }
}

template<typename Dtype>
void Tensor<Dtype>::copy_grad_from(const Tensor<Dtype> *other, bool reset) {
  if (other->size() != _size) {
    if (!reset)
      LOG(FATAL) << "Trying to copy tensor of different sizes."
                 << _size << " vs " << other->size();
    Reset(other->shape());
  }
  switch (state()) {
    case stensor::CPU:
      if (other->state() == stensor::GPU)
        LOG(FATAL) << "Trying to copy tensor of different sizes."
                   << "CPU" << " vs " << "GPU:" << other->device();
      cpu_copy(_size, other->const_grad(), grad());
      break;
    case stensor::GPU:
      if (other->state() == stensor::CPU)
        LOG(FATAL) << "Trying to copy tensor of different sizes."
                   << "GPU:" << device() << " vs " << "CPU";
      gpu_copy(_size, other->const_grad(), grad());
      break;
    default:LOG(FATAL) << "Unknown device mode.";
  }
}

template<typename Dtype>
void Tensor<Dtype>::copy_from(const Tensor<Dtype> *source, bool copy_grad, bool reset) {
  if (source->size() != _size) {
    if (!reset)
      LOG(FATAL) << "Trying to copy tensor of different sizes."
                 << _size << " vs " << source->size();
    Reset(source->shape(), source->device());
  }

  switch (state()) {
    case stensor::CPU:
      if (source->state() == stensor::GPU)
        LOG(FATAL) << "Trying to copy tensor of different sizes."
                   << "CPU" << " vs " << "GPU:" << source->device();
      if (copy_grad) {
        if (!source->const_grad() && _current_grad)
          cpu_copy(_size, source->const_grad(), grad());
        else
          LOG(WARNING) << "Copy gradient failed";
      }
      cpu_copy(_size, source->const_data(), data());

      break;
    case stensor::GPU:
      if (source->state() == stensor::CPU)
        LOG(FATAL) << "Trying to copy tensor of different sizes."
                   << "GPU:" << device() << " vs " << "CPU";
      if (copy_grad) {
        if (!source->const_grad() && _current_grad)
          gpu_copy(_size, source->const_grad(), grad());
        else
          LOG(WARNING) << "Copy gradient failed";
      }
      gpu_copy(_size, source->const_data(), data());

      break;
    default:LOG(FATAL) << "Unknown device mode.";
  }
}

template<typename Dtype>
void Tensor<Dtype>::Reset(const ShapeType &shape, int device_id) {
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
  if (_capacity == _size) return;
  _capacity = _size;
  _data.reset(new SynMem(_capacity * sizeof(Dtype), device_id));
  if (_require_grad) {
    _grad.reset(new SynMem(_capacity * sizeof(Dtype), device_id));
  }
  update_state();
  zero_data();
  if (_require_grad)
    zero_grad();
}

template<typename Dtype>
void Tensor<Dtype>::reshape(const ShapeType &shape) {
  CHECK_LE(shape.size(), kMaxTensorAxes);
  int new_size = 1;
  for (int i = 0; i < shape.size(); ++i) {
    new_size *= shape[i];
  }
  CHECK_EQ(_size, new_size) << "Size mismatch";
  _shape.resize(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    if (_size != 0) {
      CHECK_LE(shape[i], INT_MAX / _size)
        << "Tensor size exceeds INT_MAX";
    }
    _shape[i] = shape[i];
  }

}

//template<typename Dtype>
//bool Tensor<Dtype>::shape_equal(const TensorProto *other) const {
//  ShapeType shapeOther;
//  stensor::RepeatTypeToVector(other->shape(), shapeOther);
//  if (_shape.size() != shapeOther.size()) return false;
//  for (int i = 0; i < _shape.size(); ++i) {
//    if (_shape[i] != shapeOther[i]) return false;
//  }
//  return true;
//}

#define INITIAL_TO_PROTO(protoTp) \
template<typename Dtype> \
void Tensor<Dtype>::to_proto(protoTp *proto, bool write_grad) const { \
  proto->clear_shape(); \
  for (int i = 0; i < _shape.size(); ++i) {\
    proto->add_shape(_shape[i]);\
  }\
  proto->clear_data();\
  proto->clear_grad();\
  const Dtype *data_vec = const_data();\
  for (int i = 0; i < _size; ++i) {\
    proto->add_data(data_vec[i]);\
  }\
  if (write_grad) {\
    const Dtype *grad_vec = const_grad();\
    for (int i = 0; i < _size; ++i) {\
      proto->add_grad(grad_vec[i]);\
    }\
  }\
  proto->set_size(_size);\
  proto->set_name(_name);\
  proto->set_require_grad(_require_grad);\
}
INITIAL_TO_PROTO(TensorFloat);
INITIAL_TO_PROTO(TensorDouble);
INITIAL_TO_PROTO(TensorInt);
INITIAL_TO_PROTO(TensorUInt);
INITIAL_TO_PROTO(TensorBool);

#define INITIAL_FROM_PROTO(Dtype, protoTp) \
template<>\
void Tensor<Dtype>::from_proto(const protoTp &proto, bool reset) {\
  if (reset) {\
    std::vector<int> new_shape;\
    stensor::RepeatTypeToVector(proto.shape(), new_shape);\
    Reset(new_shape);\
    _size = proto.size();\
    _require_grad = proto.require_grad();\
    _name = proto.name();\
  } else {\
    CHECK_EQ(_size, proto.size());\
    CHECK_EQ(_name, proto.name());  \
    std::vector<int> shapeOther;\
    stensor::RepeatTypeToVector(proto.shape(), shapeOther);\
    CHECK(shape()==shapeOther) << "shape mismatch.";\
  }\
  switch (state()) {\
    case GPU:\
      if (proto.data_size() > 0) {\
        CHECK_EQ(_size, proto.data_size()) << "data size mismatch.";\
        stensor::gpu_copy<Dtype>(_size, proto.data().data(), data());\
      }\
      if (proto.grad_size() > 0) {\
        CHECK_EQ(_size, proto.grad_size()) << "gradiant size mismatch.";\
        stensor::gpu_copy<Dtype>(_size, proto.grad().data(), grad());\
      }\
      break;\
    case CPU:\
      if (proto.data_size() > 0) {\
        CHECK_EQ(_size, proto.data_size()) << "data size mismatch.";\
        stensor::cpu_copy<Dtype>(_size, proto.data().data(), data());\
      }\
      if (proto.grad_size() > 0) {\
        CHECK_EQ(_size, proto.grad_size()) << "gradiant size mismatch.";\
        stensor::cpu_copy<Dtype>(_size, proto.grad().data(), grad());\
      }\
      break;\
  }\
}

INITIAL_FROM_PROTO(float, TensorFloat);
INITIAL_FROM_PROTO(double,TensorDouble);
INITIAL_FROM_PROTO(int, TensorInt);
INITIAL_FROM_PROTO(unsigned int, TensorUInt);
INITIAL_FROM_PROTO(bool, TensorBool);

template<typename Dtype>
std::string memoryTostring(const Dtype *data,
                           const std::vector<int> shape,
                           bool simplified = true);

//TODO:simplified print
template<typename Dtype>
std::string memoryTostring(const Dtype *data,
                           const std::vector<int> shape,
                           bool simplified) {
  std::ostringstream out;
  int dim = shape.size();
  int count = 1;
  for (int i = 0; i < shape.size(); ++i) count *= shape[i];

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
  return out.str();
}
template<typename Dtype>
std::string Tensor<Dtype>::data_string() const {
  check_data();
  std::ostringstream out;
  const std::vector<int> shape = this->shape();
  out << memoryTostring<Tensor::Dtype>(const_data(), shape);
  out << this->shape_string();
  return out.str();
}

template<typename Dtype>
std::string Tensor<Dtype>::grad_string() const {
  CHECK(has_grad()) << "Tensor does not have gradient";
  std::ostringstream out;
  const std::vector<int> shape = this->shape();
  out << memoryTostring<Tensor::Dtype>(const_grad(), shape);
  out << shape_string() << std::endl;
  return out.str();
}

template<typename Dtype>
std::ostream &operator<<(std::ostream &out, const Tensor<Dtype> *tensor) {
  out << tensor->data_string();
  return out;
}

template std::ostream &operator<<<float> (std::ostream &out, const Tensor<float> *tensor);
template std::ostream &operator<<<double> (std::ostream &out, const Tensor<double> *tensor);
template std::ostream &operator<<<int> (std::ostream &out, const Tensor<int> *tensor);
template std::ostream &operator<<<unsigned int> (std::ostream &out, const Tensor<unsigned int> *tensor);
template std::ostream &operator<<<bool> (std::ostream &out, const Tensor<bool> *tensor);

template<typename Dtype>
inline std::ostream &operator<<(std::ostream &out, const Tensor<Dtype> &tensor) {
  return stensor::operator<<<Dtype>(out, &tensor);
}
template std::ostream &operator<<<float> (std::ostream &out, const Tensor<float> &tensor);
template std::ostream &operator<<<double> (std::ostream &out, const Tensor<double> &tensor);
template std::ostream &operator<<<int> (std::ostream &out, const Tensor<int> &tensor);
template std::ostream &operator<<<unsigned int> (std::ostream &out, const Tensor<unsigned int> &tensor);
template std::ostream &operator<<<bool> (std::ostream &out, const Tensor<bool> &tensor);

template<typename Dtype>
Dtype &Tensor<Dtype>::operator[](std::vector<int> indices) {
  CHECK_EQ(indices.size(), num_axes()) << "indices size must be equal with num axes";
  Tensor::ShapeType canonicalIndex(num_axes(), 0);
  for (int i = 0; i < num_axes(); ++i) {
    if (indices[i] < 0) {
      canonicalIndex[i] = indices[i] + shape(i);
    } else canonicalIndex[i] = indices[i];
  }
  int out = 1;
  for (int i = 0; i < indices.size(); ++i) {
    out *= static_cast<int>(canonicalIndex[i] + 1);
  }
  return data()[out - 1];
}

/* Tensor Generator end*/
/* save and load*/

template<typename Dtype>
void save(const Tensor<Dtype> *tensor, const std::string &path) {
  if (typeid(Dtype) == typeid(float)) {
    TensorFloat proto;
    tensor->to_proto(&proto);
    std::fstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);
    bool success = proto.SerializeToOstream(&output);
    CHECK(success) << "Failed to save tensor to " << path;
  } else if (typeid(Dtype) == typeid(double)) {
    TensorDouble proto;
    tensor->to_proto(&proto);
    std::fstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);
    bool success = proto.SerializeToOstream(&output);
    CHECK(success) << "Failed to save tensor to " << path;
  } else if (typeid(Dtype) == typeid(int)) {
    TensorInt proto;
    tensor->to_proto(&proto);
    std::fstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);
    bool success = proto.SerializeToOstream(&output);
    CHECK(success) << "Failed to save tensor to " << path;
  } else if (typeid(Dtype) == typeid(unsigned int)) {
    TensorUInt proto;
    tensor->to_proto(&proto);
    std::fstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);
    bool success = proto.SerializeToOstream(&output);
    CHECK(success) << "Failed to save tensor to " << path;
  } else if (typeid(Dtype) == typeid(bool)) {
    TensorBool proto;
    tensor->to_proto(&proto);
    std::fstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);
    bool success = proto.SerializeToOstream(&output);
    CHECK(success) << "Failed to save tensor to " << path;
  } else
    LOG(FATAL) << "Unsupported type:" << typeid(Dtype).name();
}

template<typename Dtype>
Tensor<Dtype> *load(const std::string &path) {
  if (typeid(Dtype) == typeid(float)) {
    TensorFloat proto;
    std::fstream input(path, std::ios::in | std::ios::binary);
    bool success = proto.ParseFromIstream(&input);
    CHECK(success) << "Failed to load tensor from " << path;
    Tensor<Dtype> *new_tensor = new Tensor<Dtype>();
    new_tensor->from_proto(proto);
    return new_tensor;
  } else if (typeid(Dtype) == typeid(double)) {
    TensorDouble proto;
    std::fstream input(path, std::ios::in | std::ios::binary);
    bool success = proto.ParseFromIstream(&input);
    CHECK(success) << "Failed to load tensor from " << path;
    Tensor<Dtype> *new_tensor = new Tensor<Dtype>();
    new_tensor->from_proto(proto);
    return new_tensor;
  } else if (typeid(Dtype) == typeid(int)) {
    TensorInt proto;
    std::fstream input(path, std::ios::in | std::ios::binary);
    bool success = proto.ParseFromIstream(&input);
    CHECK(success) << "Failed to load tensor from " << path;
    Tensor<Dtype> *new_tensor = new Tensor<Dtype>();
    new_tensor->from_proto(proto);
    return new_tensor;
  } else if (typeid(Dtype) == typeid(unsigned int)) {
    TensorUInt proto;
    std::fstream input(path, std::ios::in | std::ios::binary);
    bool success = proto.ParseFromIstream(&input);
    CHECK(success) << "Failed to load tensor from " << path;
    Tensor<Dtype> *new_tensor = new Tensor<Dtype>();
    new_tensor->from_proto(proto);
    return new_tensor;
  } else if (typeid(Dtype) == typeid(bool)) {
    TensorBool proto;
    std::fstream input(path, std::ios::in | std::ios::binary);
    bool success = proto.ParseFromIstream(&input);
    CHECK(success) << "Failed to load tensor from " << path;
    Tensor<Dtype> *new_tensor = new Tensor<Dtype>();
    new_tensor->from_proto(proto);
    return new_tensor;
  } else
    LOG(FATAL) << "Unsupported type:" << typeid(Dtype).name();
}

template<typename Dtype>
void Tensor<Dtype>::zero_data() {
  switch (state()) {
    case CPU:stensor::cpu_set<Tensor::Dtype>(_size, 0, data());
      break;
    case GPU:stensor::gpu_set<Tensor::Dtype>(_size, 0, data());
      break;
  }
}

template<typename Dtype>
void Tensor<Dtype>::zero_grad() {
  switch (state()) {
    case CPU:stensor::cpu_set<Tensor::Dtype>(_size, 0, grad());
      break;
    case GPU:stensor::gpu_set<Tensor::Dtype>(_size, 0, grad());
      break;
  }
}
//TODO: Slice
//Tensor* Tensor::operator[](std::vector<std::pair<int, int>> start_end_indices) const {
//
//}

/* save and load end*/
}//namespace stensor
