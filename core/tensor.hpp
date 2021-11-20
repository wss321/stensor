#ifndef STENSOR_TENSOR_HPP
#define STENSOR_TENSOR_HPP
#include <boost/shared_ptr.hpp>
#include "proto/tensor.pb.h"
#include "public/common.hpp"
#include "public/synmem.hpp"
#include "math_base_cuda.hpp"

using namespace boost;
namespace stensor {

const int kMaxTensorAxes = MAX_AXES;

class Tensor {
 public:
  typedef std::vector<int> ShapeType;
  typedef float Dtype;
  typedef TensorProto_Operation OpType;
  typedef shared_ptr<SynMem> SharedPtr;
  typedef struct { int start;int end; } PairType;
  typedef std::vector<PairType> PairIndexType;
  typedef std::vector<std::string> NbType;
 private:
  SharedPtr _data;
  SharedPtr _grad;
  ShapeType _shape;
  int _size;
  bool _require_grad;
  std::string _name;
  int _capacity;
  int _device;
  PairIndexType _axis_start_ends;
  ShapeType _actual_shape;
  NbType _neighbors;
  std::vector<OpType> _operations;
  Dtype *_current_data;
  Dtype *_current_grad;

 private:
  void Reset(const ShapeType &shape, int device_id = -1);//device_id = -1 represent cpu
  void update_state(); // update current data, grad, device, actual_shape
  int get_new_index(int index) const;
  int get_new_index(int index, const PairIndexType& axis_start_ends) const;
  PairIndexType get_canonical_start_ends(PairIndexType start_end_indices) const;

 public:
  Tensor() :
      _data(), _grad(), _size(0),
      _capacity(0), _name(), _device(-1),
      _require_grad(false), _axis_start_ends({}),
      _current_data(nullptr), _current_grad(nullptr) {};
  ~Tensor() {
    _data.reset();
    _grad.reset();
  }
  explicit Tensor(const ShapeType &shape, int device_id = -1, bool require_grad = false) :
      _capacity(0), _require_grad(require_grad), _axis_start_ends({}),
      _current_data(nullptr), _current_grad(nullptr) {
    Reset(shape, device_id);
  }
  Tensor(const Tensor &other, bool require_grad = false) :
      _data(), _grad(), _device(other.device()),
      _size(0), _capacity(0), _name(),
      _require_grad(require_grad), _axis_start_ends({}),
      _current_data(nullptr), _current_grad(nullptr) {
    copy_from(other, false, true);
  }

  Tensor(const Tensor *other, bool require_grad = false) :
      _data(), _grad(), _device(other->device()),
      _size(0), _capacity(0), _name(),
      _require_grad(require_grad), _axis_start_ends({}),
      _current_data(nullptr), _current_grad(nullptr) {
    copy_from(other, false, true);
  }

  Tensor(Tensor *other, const PairIndexType &new_axis_index, bool inplace = true);
  inline void set_stride(PairIndexType new_strid){
    _axis_start_ends = new_strid;
    update_state();
  };
  inline Dtype *data() {
    check_data();
    return _current_data;
  }
  inline Dtype *grad() {
    check_grad();
    return _current_data;
  }

  inline SharedPtr get_shared_data() {
    check_data();
    return _data;
  }
  inline SharedPtr get_shared_grad() {
    check_data();
    return _grad;
  }

  inline const Dtype *const_data() const {
    check_data();
    return _current_data;
  }
  inline const Dtype *const_grad() const {
    check_grad();
    return _current_data;
  }

  inline Mode state() const {
    check_data();
    if (_device > -1) return GPU;
    return CPU;
  }
  inline void check_data() const { CHECK(_current_data) << "Data is None"; }
  inline void check_grad() const { CHECK(_require_grad && _current_grad) << "Gradient is None"; }
  inline int device() const {
    return _device;
  }
  inline bool has_grad() const {
    return _current_grad != nullptr;
  }

  inline void set_require_grad(bool require_grad) { _require_grad = require_grad; }
  inline void set_name(const std::string &new_name) { _name = new_name; }

  // shape operations
  void reshape(const ShapeType &shape);
  inline void reshape_like(const Tensor &other) { reshape(other.shape()); }

  inline void flatten() { reshape(ShapeType{_size}); }

  inline std::string shape_string() const {
    std::ostringstream stream;
    stream << "{shape:(";
    for (int i = 0; i < _shape.size() - 1; ++i) {
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
  inline int canonical_axis(int axis_index) const {
    int num_axes_t = num_axes();
    CHECK_GE(axis_index, -num_axes_t)
      << "axis " << axis_index << " out of range for " << num_axes_t
      << "-D Tensor with shape " << shape_string();
    CHECK_LT(axis_index, num_axes_t)
      << "axis " << axis_index << " out of range for " << num_axes_t
      << "-D Tensor with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes_t;
    }
    return axis_index;
  }
  inline int canonical_axis_index(int index_at_axis, int axis) const {
    axis = canonical_axis(axis);
    int max_index_at_axis = shape(axis);
    if (index_at_axis < 0) {
      CHECK_GE(max_index_at_axis + index_at_axis, 0) << "index out of range";
      return max_index_at_axis + index_at_axis;
    } else {
      CHECK_LE(index_at_axis, max_index_at_axis) << "index out of range";
      return index_at_axis;
    }
  }

  inline int shape(int index) const {
    return _shape[canonical_axis(index)];
  }

  void to_cpu();
  void to_gpu();

  bool shape_equal(const Tensor &other) const;
  bool shape_equal(const TensorProto &other) const;

  inline int num_axes() const { return _shape.size(); }
  inline Dtype& data_at(int index) {
    if (index < 0) {
      DCHECK_GE(size() + index, 0);
      index = static_cast<int>(size()) + index;
    }
    DCHECK_LE(index + 1, _size) << "index out of range";
    if (_axis_start_ends.empty())
      return _current_data[index];
    else return _current_data[get_new_index(index)];
  };
  inline Dtype data_at(int index) const {
    if (index < 0) {
      DCHECK_GE(size() + index, 0);
      index = static_cast<int>(size()) + index;
    }
    DCHECK_LE(index + 1, _size) << "index out of range";
    if (_axis_start_ends.empty())
      return _current_data[index];
    else return _current_data[get_new_index(index)];
  };

  inline Dtype& data_at(uint32_t index) {
    DCHECK_LE(index + 1, _size) << "index out of range";
    if (_axis_start_ends.empty())
      return _current_data[index];
    else return _current_data[get_new_index(static_cast<int>(index))];
  };

  inline Dtype& grad_at(int index) {
    if (index < 0) {
      DCHECK_GE(size() + index, 0);
      index = static_cast<int>(size()) + index;
    }
    DCHECK_LE(index + 1, _size) << "index out of range";
    if (_axis_start_ends.empty())
      return _current_grad[index];
    else return _current_grad[get_new_index(index)];
  };

  inline Dtype& grad_at(uint32_t index) {
    DCHECK_LE(index + 1, _size) << "index out of range";
    if (_axis_start_ends.empty())
      return _current_grad[index];
    else return _current_grad[get_new_index(static_cast<int>(index))];
  };
  inline Dtype grad_at(uint32_t index) const {
    DCHECK_LE(index + 1, _size) << "index out of range";
    if (_axis_start_ends.empty())
      return _current_grad[index];
    else return _current_grad[get_new_index(static_cast<int>(index))];
  };

  inline int offset(const ShapeType &indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }

  inline int size() const {
    return _size;
  };

  inline int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    int num_axes_t = static_cast<int>(num_axes());
    CHECK_LE(start_axis, num_axes_t);
    CHECK_LE(end_axis, num_axes_t);
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }

  void copy_from(const Tensor &source, bool copy_grad = false,
                 bool reset = false);
  void copy_from(const Tensor *source, bool copy_grad = false,
                 bool reset = false);

  inline bool require_grad() const {
    return _require_grad;
  }

//  const std::pair<Tensor *, std::string> neighbors() const;
  void from_proto(const TensorProto &proto, bool reset = true);
  void to_proto(TensorProto &proto, bool write_grad = false) const;

  Tensor &operator=(const Tensor &other);
  Tensor &operator=(const Tensor *other);

  Dtype &operator[](std::vector<int> indices); // get data
//  Dtype &operator[](std::vector<uint32_t> indices);
  Tensor *operator[](PairIndexType start_end_indices); // slice
  Dtype &operator[](int index) {
    if (index < 0) {
      DCHECK_GE(size() + index, 0);
      index = static_cast<int>(size()) + index;
    }
    DCHECK_LE(index + 1, _size) << "index out of range";
    if (_axis_start_ends.empty())
      return _current_data[index];
    else return _current_data[get_new_index(index)];
  }

  Dtype &operator[](uint32_t index) {
    DCHECK_LE(index + 1, _size) << "index out of range";
    if (_axis_start_ends.empty())
      return _current_data[index];
    else return _current_data[get_new_index(static_cast<int>(index))];
  }
  void zero_data();
  void zero_grad();

  void copy_data_from(const Tensor *other, bool reset = false);
  void copy_grad_from(const Tensor *other, bool reset = false);
// DISABLE_COPY_AND_ASSIGN(Tensor);
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
