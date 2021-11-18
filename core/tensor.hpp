#ifndef STENSOR_TENSOR_HPP
#define STENSOR_TENSOR_HPP
#include <boost/shared_ptr.hpp>
#include "proto/tensor.pb.h"
#include "public/common.hpp"
#include "public/synmem.hpp"
#include "math_base_cuda.hpp"

using namespace boost;
namespace stensor {

const uint32_t kMaxTensorAxes = MAX_AXES;

class Tensor {
 public:
  typedef std::vector<uint32_t> ShapeType;
  typedef float Dtype;
  typedef TensorProto_Operation OpType;
  typedef shared_ptr<SynMem> SharedPtr;
  typedef SynMem::SynState State;
 private:
  SharedPtr _data;
  SharedPtr _grad;
  ShapeType _shape;
  uint32_t _size;
  std::vector<std::string> _neighbors;
  std::vector<OpType> _operations;
  bool _require_grad;
  std::string _name;
  uint32_t _capacity;
  void Reset(const ShapeType &shape, int device_id = -1);//device_id = -1 represent cpu
 public:
  Tensor() : _data(),
             _grad(), _size(0ul),
             _capacity(0ul), _name(),
             _require_grad(false) {};
  ~Tensor() {
    _data.reset();
    _grad.reset();
    _size = 0;
    _capacity = 0;
    _name = "";
    _require_grad = false;
    _neighbors.clear();
    _operations.clear();
    _shape.clear();
  }
  explicit Tensor(const ShapeType &shape, bool require_grad = false, int device_id = -1);
  Tensor(const Tensor &other, bool require_grad = false);
  Tensor(const Tensor *other, bool require_grad = false);
  Tensor(const std::vector<Dtype> &other, const ShapeType &shape, bool require_grad = false, const Mode mode = CPU);
  inline Mode state() const {
    check_data();
    if (_data->state() == SynMem::AT_GPU || _data->state() == SynMem::SYNED) return GPU;
    return CPU;
  }
  inline void check_data() const { CHECK(_data)<<"Data is None";}
  inline void check_grad() const { CHECK(_require_grad &&_grad)<<"Gradient is None";}
  inline int device() const {
    check_data();
    return _data->device(); }
  inline bool has_grad() const {
    check_grad();
    return _grad->has_cpu_data() || _grad->has_gpu_data();
  }

  inline void set_require_grad(bool require_grad) { _require_grad = require_grad; }
  inline void set_name(const std::string &new_name) { _name = new_name; }

  // shape operations
  void Reshape(const ShapeType &shape);
  void ReshapeLike(const Tensor &other);
  void ReshapeLike(const Tensor *other);
  void flatten();

  inline std::string shape_string() const {
    std::ostringstream stream;
    stream << "{shape:(";
    for (uint32_t i = 0; i < _shape.size() - 1; ++i) {
      stream << _shape[i] << "x";
    }
    if (!_shape.empty())
      stream << _shape[_shape.size() - 1];
    else stream << "0";
    stream << "=" << _size << ")";
    if (state() == CPU) stream << ",device:CPU";
    else if (state() == GPU) stream << ",device:GPU" << device();
    stream << ",dtype:"
           << abi::__cxa_demangle(typeid(Dtype).name(), nullptr, nullptr, nullptr)
           << "}" << std::endl;
    return stream.str();
  }
  std::string data_string() const;
  std::string grad_string() const;

  inline const ShapeType &shape() const { return _shape; }
  inline uint32_t CanonicalAxisIndex(int32_t axis_index) const {
    int32_t num_axes_t = static_cast<int32_t>(num_axes());
    DCHECK_GE(axis_index, -num_axes_t)
      << "axis " << axis_index << " out of range for " << num_axes_t
      << "-D Tensor with shape " << shape_string();
    DCHECK_LT(axis_index, num_axes_t)
      << "axis " << axis_index << " out of range for " << num_axes_t
      << "-D Tensor with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes_t;
    }
    return axis_index;
  }

  inline uint32_t shape(int32_t index) const {
    return _shape[CanonicalAxisIndex(index)];
  }

  void to_cpu();
  void to_gpu();

  bool ShapeEquals(const Tensor *other);
  bool ShapeEquals(const Tensor &other);
  bool ShapeEquals(const TensorProto *other);
  bool ShapeEquals(const TensorProto &other);

  inline uint32_t num_axes() const { return _shape.size(); }
  inline Dtype data_at(int index) const {
    if (index < 0) {
      CHECK_GE(size() + index, 0);
      index = static_cast<int>(size()) + index;
    }
    CHECK_LE(index + 1, _size) << "index out of range";
    if (state() == GPU) return gpu_data()[index];
    return cpu_data()[index];
  };

  inline Dtype grad_at(int index) const {
    if (index < 0) {
      CHECK_GE(size() + index, 0);
      index = static_cast<int>(size()) + index;
    }
    CHECK_LT(index, _size) << "index out of range";
    if (state() == GPU) return gpu_grad()[index];
    return cpu_grad()[index];
  };

  inline uint32_t offset(const ShapeType &indices) const {
    DCHECK_LE(indices.size(), num_axes());
    uint32_t offset = 0;
    for (int32_t i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        DCHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }

  inline uint32_t size() const {
    return _size;
  };

  inline uint32_t count(int32_t start_axis, int32_t end_axis) const {
    DCHECK_LE(start_axis, end_axis);
    DCHECK_GE(start_axis, 0);
    DCHECK_GE(end_axis, 0);
    int32_t num_axes_t = static_cast<int32_t>(num_axes());
    DCHECK_LE(start_axis, num_axes_t);
    DCHECK_LE(end_axis, num_axes_t);
    uint32_t count = 1ul;
    for (int32_t i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }

  void CopyFrom(const Tensor &source, bool copy_grad = false,
                bool reset = false);
  void CopyFrom(const Tensor *source, bool copy_grad = false,
                bool reset = false);

  inline const Dtype *cpu_data() const {
    check_data();
    return (const Dtype *) _data->cpu_data();
  };

  inline const Dtype *cpu_grad() const {
    check_grad();
    return (const Dtype *) _grad->cpu_data();
  };

  inline Dtype *mutable_cpu_data() {
    check_data();
    return static_cast<Dtype * >(_data->mutable_cpu_data());
  };

  inline Dtype *mutable_cpu_grad() {
    check_grad();
    return static_cast<Dtype * >(_grad->mutable_cpu_data());
  };

  inline const Dtype *gpu_data() const {
    check_data();
    return (const Dtype *) _data->gpu_data();
  };

  inline const Dtype *gpu_grad() const {
    check_grad();
    return (const Dtype *) _grad->gpu_data();
  };

  inline Dtype *mutable_gpu_data() {
    check_data();
    return static_cast<Dtype * >(_data->mutable_gpu_data());
  };

  inline Dtype *mutable_gpu_grad() {
    check_grad();
    return static_cast<Dtype * >(_grad->mutable_gpu_data());
  };

  inline bool require_grad() const {
    return _require_grad;
  }

//  const std::pair<Tensor *, std::string> neighbors() const;
  void FromProto(const TensorProto &proto, bool reshape = true);
  void FromProto(const TensorProto *proto, bool reset = true);
  void ToProto(TensorProto &proto, bool write_grad = false) const;
  void ToProto(TensorProto *proto, bool write_grad = false) const;

  Tensor &operator=(const Tensor &other);
  Tensor &operator=(const Tensor *other);

  Dtype operator[](std::vector<int> indices) const; // get data
  void zero_data();
  void zero_grad();
  Tensor *operator[](std::vector<std::pair<int, int>> start_end_indices) const; // slice
// DISABLE_COPY_AND_ASSIGN(Tensor);
 private:
  void register_op(OpType);
};

std::ostream &operator<<(std::ostream &out, const Tensor &tensor);
std::ostream &operator<<(std::ostream &out, const Tensor *tensor);

/* save and load*/
void save(const Tensor *tensor, const std::string &path);
inline void save(const Tensor &tensor, const std::string &path) { save(&tensor, path); }
Tensor *load(const std::string &path);
/* save and load end*/

}//namespace stensor

#endif //STENSOR_TENSOR_HPP
