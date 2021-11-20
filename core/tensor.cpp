#include <public/synmem.hpp>
#include <utility>
#include "tensor.hpp"
#include "proto/tensor.pb.h"
#include "math_base_cpu.hpp"

namespace stensor {
int get_index(std::vector<int> &shape, Tensor::PairIndexType &stride, int index) {
  std::vector<int> new_shape(shape.size());
  for (int i = 0; i < shape.size(); ++i) {
    new_shape[i] = stride[i].end - stride[i].start;
  }
  int num_axis = new_shape.size();
  std::vector<int> indices(shape.size());
  int div = 1;
  indices[num_axis - 1] = index % new_shape[num_axis - 1];
  for (int i = num_axis - 2; i >= 0; --i) {
    div *= new_shape[i + 1];
    indices[i] = (index / div) % new_shape[i];
  }
//  std::cout << indices << "\n";
  for (int i = 0; i < num_axis; ++i) {
    indices[i] += stride[i].start;
  }
//  std::cout << indices;
  int offset = 0;
  for (int i = 0; i < num_axis; ++i) {
    offset *= shape[i];
    if (indices.size() > i) {
      CHECK_LT(indices[i], shape[i]);
      offset += indices[i];
    }
  }
  return offset;
}
int Tensor::get_new_index(int index) const {
  int num_axis = num_axes();
  std::vector<int> indices(_shape.size());
  int div = 1;
  indices[num_axis - 1] = index % _actual_shape[num_axis - 1];
  for (int i = num_axis - 2; i >= 0; --i) {
    div *= _actual_shape[i + 1];
    indices[i] = (index / div) % _actual_shape[i];
  }
  for (int i = 0; i < num_axis; ++i) {
    indices[i] += _axis_start_ends[i].start;
  }

  int offset = 0;
  for (int i = 0; i < num_axis; ++i) {
    offset *= _shape[i];
    if (indices.size() > i) {
      CHECK_LT(indices[i], _shape[i]);
      offset += indices[i];
    }
  }
  return offset;
}

int Tensor::get_new_index(int index, const PairIndexType &axis_start_ends) const {
  int num_axis = num_axes();
  std::vector<int> indices(_shape.size());
  int div = 1;
  indices[num_axis - 1] = index % _actual_shape[num_axis - 1];
  for (int i = num_axis - 2; i >= 0; --i) {
    div *= _actual_shape[i + 1];
    indices[i] = (index / div) % _actual_shape[i];
  }
  for (int i = 0; i < num_axis; ++i) {
    indices[i] += axis_start_ends[i].start;
  }

  int offset = 0;
  for (int i = 0; i < num_axis; ++i) {
    offset *= _shape[i];
    if (indices.size() > i) {
      CHECK_LT(indices[i], _shape[i]);
      offset += indices[i];
    }
  }
  return offset;
}

//TODO:[]
Tensor::Tensor(Tensor *other, const PairIndexType &new_axis_index, bool inplace) :
    _device(other->device()), _name(), _size(0),
    _require_grad(other->require_grad()) {
  CHECK_EQ(new_axis_index.size(), other->num_axes()) << "indices size must be equal with num axes";
  PairIndexType new_canonical_axis_index(other->num_axes());
  ShapeType new_shape(new_axis_index.size());
  int new_size = 1;
  for (int i = 0; i < other->num_axes(); ++i) {
    int start_axis_index = new_axis_index[i].start;
    int end_axis_index = new_axis_index[i].end;
    CHECK_LT(start_axis_index, end_axis_index) << "index error";
    new_canonical_axis_index[i] = PairType({start_axis_index, end_axis_index});
    new_size *= end_axis_index - start_axis_index;
    new_shape[i] = end_axis_index - start_axis_index;
  }
  if (inplace) {
    _shape = other->shape();
    _data = other->get_shared_data();
    _grad = other->get_shared_grad();
    _size = new_size;
    _capacity = other->size();
    _current_grad = other->grad();
    _current_data = other->data();
    _axis_start_ends = new_axis_index;
  } else {
    Reset(new_shape, other->device());
    copy_from(other, true, false);
  }
  update_state();
}

void Tensor::update_state() {
  if (_data) {
    if (_data->gpu_data()) {
      _current_data = static_cast<Dtype *>(_data->gpu_data());
      _device = _data->device();
    } else if (_data->cpu_data()) {
      _current_data = static_cast<Dtype *>(_data->cpu_data());
      _device = -1;
    }
  }
  if (_grad) {
    if (_grad->gpu_data()) {
      _current_grad = static_cast<Dtype *>(_grad->gpu_data());
      _device = _grad->device();
    } else if (_grad->cpu_data()) {
      _current_grad = static_cast<Dtype *>(_grad->cpu_data());
      _device = -1;
    }
  }
  if (!_axis_start_ends.empty()) {
    _actual_shape.resize(_axis_start_ends.size());
    for (int i = 0; i < _shape.size(); ++i) {
      _actual_shape[i] = _axis_start_ends[i].end - _axis_start_ends[i].start;
    }
  }
}

void Tensor::to_cpu() {
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

void Tensor::to_gpu() {
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

void Tensor::copy_from(const Tensor &source, bool copy_grad, bool reset) {
  copy_from(&source, copy_grad, reset);
}

void Tensor::copy_data_from(const Tensor *other, bool reset) {
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

void Tensor::copy_grad_from(const Tensor *other, bool reset) {
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

void Tensor::copy_from(const Tensor *source, bool copy_grad, bool reset) {
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
      if (copy_grad)
        for (uint32_t i = 0; i < size(); ++i)
          grad_at(i) = source->grad_at(i);
//        cpu_copy(_size, source->const_grad(), grad());
      for (uint32_t i = 0; i < size(); ++i)
        data_at(i) = source->data_at(i);
//      cpu_copy(_size, source->const_data(), data());

      break;
    case stensor::GPU:
      if (source->state() == stensor::CPU)
        LOG(FATAL) << "Trying to copy tensor of different sizes."
                   << "GPU:" << device() << " vs " << "CPU";
//      if (copy_grad)
//        gpu_copy(_size, source->const_grad(), grad());
//      gpu_copy(_size, source->const_data(), data());
      if (copy_grad)
        for (uint32_t i = 0; i < size(); ++i)
          grad_at(i) = source->grad_at(i);
//        cpu_copy(_size, source->const_grad(), grad());
      for (uint32_t i = 0; i < size(); ++i)
        data_at(i) = source->data_at(i);
//      cpu_copy(_size, source->const_data(), data());

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
  if (_require_grad) {
    _grad.reset(new SynMem(_capacity * sizeof(Dtype), device_id));
  }
  update_state();
}

void Tensor::reshape(const ShapeType &shape) {
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

bool Tensor::shape_equal(const Tensor &other) const {
  ShapeType shapeOther;
  stensor::RepeatTypeToVector(other.shape(), shapeOther);
  if (_shape.size() != shapeOther.size()) return false;
  for (int i = 0; i < _shape.size(); ++i) {
    if (_shape[i] != shapeOther[i]) return false;
  }
  return true;
}

bool Tensor::shape_equal(const TensorProto &other) const {
  ShapeType shapeOther;
  stensor::RepeatTypeToVector(other.shape(), shapeOther);
  if (_shape.size() != shapeOther.size()) return false;
  for (int i = 0; i < _shape.size(); ++i) {
    if (_shape[i] != shapeOther[i]) return false;
  }
  return true;
}

void Tensor::to_proto(TensorProto &proto, bool write_grad) const {
  proto.clear_shape();
  // 1. shape
  for (int i = 0; i < _shape.size(); ++i) {
    proto.add_shape(_shape[i]);
  }
  proto.clear_data();
  proto.clear_grad();
  // 2. data
  const float *data_vec = const_data();
  for (int i = 0; i < _size; ++i) {
    proto.add_data(data_vec[i]);
  }
  // 3. grad
  if (write_grad) {
    const float *grad_vec = const_grad();
    for (int i = 0; i < _size; ++i) {
      proto.add_grad(grad_vec[i]);
    }
  }
  // 4. size & name & isparam
  proto.set_size(_size);
  proto.set_name(_name);
  proto.set_require_grad(_require_grad);
  // 5. neighbor & operations
  proto.clear_neighbors();
  proto.clear_operations();
  for (int i = 0; i < _neighbors.size(); ++i) {
    proto.add_neighbors(_neighbors[i]);
    proto.add_operations(_operations[i]);
  }
}

void Tensor::from_proto(const TensorProto &proto, bool reset) {
  // 1. reshape
  _size = proto.size();
  _require_grad = proto.require_grad();
  _name = proto.name();
  if (reset) {
    Tensor::ShapeType new_shape;
    stensor::RepeatTypeToVector(proto.shape(), new_shape);
    Reset(new_shape);
  } else {
    CHECK(shape_equal(proto)) << "shape mismatch.";
  }
  // 2. copy data
  switch (state()) {
    case GPU:
      if (proto.data_size() > 0) {
        CHECK_EQ(_size, proto.data_size()) << "data size mismatch.";
        stensor::gpu_copy(_size, proto.data().data(), data());
      }
      // 3. copy grad
      if (proto.grad_size() > 0) {
        CHECK_EQ(_size, proto.grad_size()) << "gradiant size mismatch.";
        stensor::gpu_copy(_size, proto.grad().data(), grad());
      }
      break;
    case CPU:
      if (proto.data_size() > 0) {
        CHECK_EQ(_size, proto.data_size()) << "data size mismatch.";
        stensor::cpu_copy(_size, proto.data().data(), data());
      }
      if (proto.grad_size() > 0) {
        CHECK_EQ(_size, proto.grad_size()) << "gradiant size mismatch.";
        stensor::cpu_copy(_size, proto.grad().data(), grad());
      }
      break;
  }
  // 4. neighbors & operations
  stensor::RepeatTypeToVector(proto.neighbors(), _neighbors);
  stensor::RepeatTypeToVector(proto.operations(), _operations);
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
  out << memoryTostring<Tensor::Dtype>(const_data(), shape);
  out << this->shape_string();
  return out.str();
}

std::string Tensor::grad_string() const {
  CHECK(has_grad()) << "Tensor does not have gradient";
  std::ostringstream out;
  const Tensor::ShapeType shape = this->shape();
  out << memoryTostring<Tensor::Dtype>(const_grad(), shape);
  out << "{shape:" << shape_string() << ", dtype:"
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

Tensor &Tensor::operator=(const Tensor &other) {
  copy_from(other, false, true);
  return (*this);
}
Tensor &Tensor::operator=(const Tensor *other) {
  copy_from(other, false, true);
  return (*this);
}

Tensor::Dtype &Tensor::operator[](std::vector<int> indices) {
  CHECK_EQ(indices.size(), num_axes()) << "indices size must be equal with num axes";
  Tensor::ShapeType canonicalIndex(num_axes(), 0);
  for (int i = 0; i < num_axes(); ++i) {
    if (indices[i] < 0) {
//      CHECK_GE(indices[i], shape(i));
      canonicalIndex[i] = indices[i] + shape(i);
    } else canonicalIndex[i] = indices[i];
  }

  int out = 1;
  for (int i = 0; i < indices.size(); ++i) {
    out *= static_cast<int>(canonicalIndex[i] + 1);
  }
  return data()[out - 1];
}

Tensor::PairIndexType Tensor::get_canonical_start_ends(PairIndexType start_end_indices) const {
  CHECK_EQ(start_end_indices.size(), num_axes()) << "indices size must be equal with num axes";
  PairIndexType new_axis_index(num_axes());
  for (int i = 0; i < num_axes(); ++i) {
    int start_axis_index = canonical_axis_index(start_end_indices[i].start, i);
    int end_axis_index = canonical_axis_index(start_end_indices[i].end, i);
    CHECK_LT(start_axis_index, end_axis_index) << "index error";
    new_axis_index[i] = PairType({start_axis_index, end_axis_index});
  }
  return new_axis_index;
}

Tensor *Tensor::operator[](PairIndexType start_end_indices) {
  return new Tensor(this, get_canonical_start_ends(start_end_indices), false);

}

/* Tensor Generator end*/
/* save and load*/
void save(const Tensor *tensor, const std::string &path) {
  TensorProto proto;
  tensor->to_proto(proto);
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
  new_tensor->from_proto(proto);
  return new_tensor;
}

void Tensor::zero_data() {
  switch (state()) {
    case CPU:stensor::cpu_set<Tensor::Dtype>(_size, 0, data());
      break;
    case GPU:stensor::gpu_set<Tensor::Dtype>(_size, 0, data());
      break;
  }
}
void Tensor::zero_grad() {
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
