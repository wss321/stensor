#include <public/synmem.hpp>
#include "tensor.hpp"
#include "proto/tensor.pb.h"
#include "math_base_cpu.hpp"

namespace stensor {
void Tensor::to_cpu() {
  if (_capacity != 0) {
    _data->to_cpu();
    if (require_grad())
      _grad->to_cpu();
  }
}

void Tensor::to_gpu() {
  if (_capacity != 0) {
    _data->to_gpu();
    if (require_grad())
      _grad->to_gpu();
  }
}
void Tensor::CopyFrom(const Tensor &source, bool copy_grad, bool reset) {
  CopyFrom(&source, copy_grad, reset);
}

void Tensor::CopyFrom(const Tensor *source, bool copy_grad, bool reset) {
  if (source->size() != _size) {
    if (!reset)
      LOG(FATAL) << "Trying to copy tensor of different sizes."
                 << _size << " vs " << source->size();
    Reset(source->shape());
  }

  //TODO:CopyFrom GPU
  switch (state()) {
    case stensor::CPU:
      if (source->state() == stensor::GPU)
        LOG(FATAL) << "Trying to copy tensor of different sizes."
                   << "CPU" << " vs " << "GPU:" << source->device();
      if (copy_grad) {
        cpu_copy(_size, source->cpu_grad(), static_cast<Dtype *>(_grad->mutable_cpu_data()));
      } else {
        cpu_copy(_size, source->cpu_data(), static_cast<Dtype *>(_data->mutable_cpu_data()));
      }
      break;
    case stensor::GPU:
      if (source->state() == stensor::CPU)
        LOG(FATAL) << "Trying to copy tensor of different sizes."
                   << "GPU:" << device() << " vs " << "CPU";
      if (copy_grad) {
        gpu_copy(_size, source->gpu_grad(), static_cast<Dtype *>(_grad->mutable_gpu_data()));
      } else {
        gpu_copy(_size, source->gpu_data(), static_cast<Dtype *>(_data->mutable_gpu_data()));
      }
      break;
    default:LOG(FATAL) << "Unknown device mode.";
  }
}

void Tensor::Reset(const ShapeType &shape, int device_id) {
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
  if (device_id > -1) gpu_data(); else cpu_data();

  if (_require_grad) {
    _grad.reset(new SynMem(_capacity * sizeof(Dtype), device_id));
    if (device_id > -1) gpu_grad(); else cpu_grad();
  }

}

void Tensor::Reshape(const ShapeType &shape) {
  CHECK_LE(shape.size(), kMaxTensorAxes);
  uint32_t new_size = 1;
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

void Tensor::ReshapeLike(const Tensor &other) {
  Reshape(other.shape());
}
void Tensor::ReshapeLike(const Tensor *other) {
  Reshape(other->shape());
}

void Tensor::flatten() {
  Reshape(ShapeType{_size});
}

Tensor::Tensor(const ShapeType &shape, bool require_grad, int device_id) :
    _capacity(0ul), _require_grad(require_grad) {
  Reset(shape, device_id);
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

void Tensor::FromProto(const TensorProto *proto, bool reset) {
  // 1. reshape
  _size = proto->size();
  _require_grad = proto->require_grad();
  _name = proto->name();
  if (reset) {
    Tensor::ShapeType new_shape;
    stensor::RepeatTypeToVector(proto->shape(), new_shape);
    Reset(new_shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch.";
  }
  // 2. copy data
  switch (state()) {
    case GPU:
      if (proto->data_size() > 0) {
        Dtype *data_vec = mutable_gpu_data();
        CHECK_EQ(_size, proto->data_size()) << "data size mismatch.";
        stensor::cpu_copy(_size, proto->data().data(), data_vec);
      }
      // 3. copy grad
      if (proto->grad_size() > 0) {
        CHECK_EQ(_size, proto->grad_size()) << "gradiant size mismatch.";
        Dtype *grad_vec = mutable_gpu_grad();
//    for (int i = 0; i < _size; ++i) {
//      grad_vec[i] = proto->grad(i);
//    }
        stensor::gpu_copy(_size, proto->grad().data(), grad_vec);
      }
      break;
    case CPU:
      if (proto->data_size() > 0) {
        Dtype *data_vec = mutable_cpu_data();
        CHECK_EQ(_size, proto->data_size()) << "data size mismatch.";
        stensor::gpu_copy(_size, proto->data().data(), data_vec);
      }
      if (proto->grad_size() > 0) {
        CHECK_EQ(_size, proto->grad_size()) << "gradiant size mismatch.";
        Dtype *grad_vec = mutable_cpu_grad();
        stensor::cpu_copy(_size, proto->grad().data(), grad_vec);
      }
      break;
  }
  // 4. neighbors & operations
  stensor::RepeatTypeToVector(proto->neighbors(), _neighbors);
  stensor::RepeatTypeToVector(proto->operations(), _operations);
}

void Tensor::FromProto(const TensorProto &proto, bool reshape) {
  FromProto(&proto, reshape);
}

//template<typename Dtype>
//std::string memoryTostringSimplified(const Dtype *data,
//                           const Tensor::ShapeType shape){
//  std::ostringstream out;
//  int dim = shape.size();
//  int count = 1;
//  for (int i = 0; i < shape.size(); ++i) count *= shape[i];
//
//  for (int i = 0; i < dim; ++i) {
//    out << "[";
//  }
//
//}
template<typename Dtype>
std::string memoryTostring(const Dtype *data,
                           const Tensor::ShapeType shape,
                           bool simplified = true);

//TODO:simplified print
template<typename Dtype>
std::string memoryTostring(const Dtype *data,
                           const Tensor::ShapeType shape,
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

std::string Tensor::data_string() const {
  check_data();
  std::ostringstream out;
  const Tensor::ShapeType shape = this->shape();
  const Tensor::Dtype *data = state() == CPU ? this->cpu_data() : this->gpu_data();
  out << memoryTostring<Tensor::Dtype>(data, shape);
  out << this->shape_string();
  return out.str();
}

std::string Tensor::grad_string() const {
  CHECK(has_grad()) << "Tensor does not have gradient";
  std::ostringstream out;
  const Tensor::ShapeType shape = this->shape();
  const Tensor::Dtype *data = state() == CPU ? this->cpu_grad() : this->gpu_grad();
  out << memoryTostring<Tensor::Dtype>(data, shape);
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
    _data(), _grad(),
    _size(0ul), _capacity(0ul), _name(),
    _require_grad(require_grad) {
  CopyFrom(other, false, true);
}

Tensor::Tensor(const Tensor *other, bool require_grad) :
    _data(), _grad(),
    _size(0ul), _capacity(0ul), _name(),
    _require_grad(require_grad) {

  CopyFrom(other, false, true);
}

//TODO:GPU Mode
Tensor::Tensor(const std::vector<Tensor::Dtype> &other,
               const ShapeType &shape, bool require_grad,
               const Mode mode) :
    _data(), _grad(),
    _size(0ul), _capacity(0ul), _name(),
    _require_grad(require_grad) {
  CHECK(!other.empty()) << "Attempting assign an empty vector to tensor";
  uint32_t num = stensor::count(shape);
  CHECK_EQ(other.size(), num) << "Shape mismatch";
  Reset(shape);
  Tensor::Dtype *data = mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
    data[i] = other[i];
  }
}
//TODO:operater overload
Tensor &Tensor::operator=(const Tensor &other) {
  CopyFrom(other, false, true);
  return (*this);
}
Tensor &Tensor::operator=(const Tensor *other) {
  CopyFrom(other, false, true);
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

void Tensor::zero_data() {
  switch (state()) {
    case CPU:stensor::cpu_set<Tensor::Dtype>(_size, 0, mutable_cpu_data());
      break;
    case GPU:stensor::gpu_set<Tensor::Dtype>(_size, 0, mutable_gpu_data());
      break;
  }
}
void Tensor::zero_grad(){
  switch (state()) {
    case CPU:stensor::cpu_set<Tensor::Dtype>(_size, 0, mutable_cpu_grad());
      break;
    case GPU:stensor::gpu_set<Tensor::Dtype>(_size, 0, mutable_gpu_grad());
      break;
  }
}
//TODO: Slice
//Tensor* Tensor::operator[](std::vector<std::pair<int, int>> start_end_indices) const {
//
//}

/* save and load end*/
}//namespace stensor
