#ifndef STENSOR_TENSOR_HPP
#define STENSOR_TENSOR_HPP
#include <boost/shared_ptr.hpp>
#include "proto/tensor.pb.h"
#include "public/common.hpp"
#include "public/synmem.hpp"

using namespace boost;
namespace stensor {

const uint32_t kMaxTensorAxes = 32;

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
 public:
  Tensor() : _data(),
             _grad(), _size(0ul),
             _capacity(0ul), _name(),
             _require_grad(false) {};
  explicit Tensor(const ShapeType &shape,bool require_grad=false);
  Tensor(const Tensor &other,bool require_grad=false);
  Tensor(const Tensor *other,bool require_grad=false);
  Tensor(const std::vector<Dtype> &other, const ShapeType &shape,bool require_grad=false, const Mode mode = CPU);

  Mode state() const {
    if (_data->state() == SynMem::AT_GPU || _data->state() == SynMem::SYNCED) return GPU;
    return CPU;
  }
  // shape operations
  void Reshape(const ShapeType &shape);
  void ReshapeLike(const Tensor &other);
  void ReshapeLike(const Tensor *other);

  void flatten();
  inline std::string shape_string() const {
    std::ostringstream stream;
    stream << "(";
    for (uint32_t i = 0; i < _shape.size() - 1; ++i) {
      stream << _shape[i] << "x";
    }
    if (!_shape.empty())
      stream << _shape[_shape.size() - 1];
    stream << ")";
    stream << "(total:" << _size << ")";
    return stream.str();
  }
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
  bool ShapeEquals(const Tensor *other);
  bool ShapeEquals(const Tensor &other);
  bool ShapeEquals(const TensorProto *other);
  bool ShapeEquals(const TensorProto &other);

  inline uint32_t num_axes() const { return _shape.size(); }
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
                bool reshape = false, Mode mode = stensor::CPU);
  void CopyFrom(const Tensor *source, bool copy_grad = false,
                bool reshape = false, Mode mode = stensor::CPU);

  inline const Dtype *cpu_data() const {
    CHECK(_data) << "Data is none.";
    return (const Dtype *) _data->cpu_data();
  };

  inline const Dtype *cpu_grad() const {
    CHECK(_grad) << "Grad is none.";
    return (const Dtype *) _grad->cpu_data();
  };

  inline Dtype *mutable_cpu_data() {
    CHECK(_data) << "Data is none.";
    return static_cast<Dtype * >(_data->mutable_cpu_data());
  };

  inline Dtype *mutable_cpu_grad() {
    CHECK(_grad) << "Grad is none.";
    return static_cast<Dtype * >(_grad->mutable_cpu_data());
  };

  inline bool require_grad() const {
    return _require_grad;
  }

  std::string data_string() const;

//  const std::pair<Tensor *, std::string> neighbors() const;
  void FromProto(const TensorProto &proto, bool reshape = true);
  void FromProto(const TensorProto *proto, bool reshape = true);
  void ToProto(TensorProto &proto, bool write_grad = false) const;
  void ToProto(TensorProto *proto, bool write_grad = false) const;

  void operator+(Tensor &other);
  void operator-(Tensor &other);
  void operator*(Tensor &other);
  void operator/(Tensor &other);
  Tensor &operator=(const Tensor &other);
  Tensor &operator=(const Tensor *other);

  // scatter
  void operator+(Dtype scalar);
  void operator-(Dtype scalar);
  void operator*(Dtype scalar);
  void operator/(Dtype scalar);


// DISABLE_COPY_AND_ASSIGN(Tensor);
 private:
  void register_op(OpType);
};

std::ostream &operator<<(std::ostream &out, const Tensor &tensor);
std::ostream &operator<<(std::ostream &out, const Tensor *tensor);

/* math of Tensor */
void set(Tensor &tensor, const float val);
void set(Tensor *tensor, const float val);

Tensor *add(Tensor &tensor, const float val, bool inplace = false);
Tensor *sub(Tensor &tensor, const float val, bool inplace = false);
Tensor *scale(Tensor &tensor, const float val, bool inplace = false);
Tensor *pow(Tensor &tensor, const float val, bool inplace = false);

Tensor *add(const Tensor &a, const Tensor &b);
Tensor *sub(const Tensor &a, const Tensor &b);
Tensor *scale(const Tensor &a, const Tensor &b);
Tensor *div(const Tensor &a, const Tensor &b);
Tensor *exp(const Tensor &a, const Tensor &b);
/* math of Tensor end */

/* Tensor Generator*/
Tensor *random(const Tensor::ShapeType& shape, bool require_grad, float a, float b);
Tensor *random(const Tensor::ShapeType& shape, bool require_grad= false);
Tensor *random(const Tensor::ShapeType& shape, float a, float b);

Tensor *random_gaussian(const Tensor::ShapeType& shape, bool require_grad , float mu, float sigma);
Tensor *random_gaussian(const Tensor::ShapeType& shape, bool require_grad= false);
Tensor *random_gaussian(const Tensor::ShapeType& shape, float mu, float sigma);

Tensor *constants(const Tensor::ShapeType& shape, Tensor::Dtype val, bool require_grad = false);
Tensor *zeros(const Tensor::ShapeType& shape, bool require_grad = false);
Tensor *ones(const Tensor::ShapeType& shape, bool require_grad = false);
Tensor *zeros_like(Tensor *other, bool require_grad = false);
Tensor *ones_like(Tensor *other, bool require_grad = false);
Tensor *constants_like(Tensor *other, Tensor::Dtype val, bool require_grad = false);
/* Tensor Generator end*/

}//namespace stensor

#endif //STENSOR_TENSOR_HPP
