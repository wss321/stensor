/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#ifndef STENSOR_NN_MODULE_HPP_
#define STENSOR_NN_MODULE_HPP_
#include "tensor.hpp"
#include "common.hpp"
#include "math_tesnsor.hpp"
#include "proto/module.pb.h"

namespace stensor {

namespace nn {
typedef shared_ptr<stensor::Tensor> SharedTensor;
typedef std::vector<shared_ptr<stensor::Tensor>> TensorVec;
class Module {
 public:
  Module() {};
  explicit Module(const ModuleParameter &param) {
    name_ = param.name();
    type_ = param.type();
    for (int i = 0; i < param.param_size(); ++i)
      parameters_[i]->from_proto(param.param(i));

    for (int i = 0; i < param.submodule_size(); ++i)
      submodules_[i]->from_proto(param.submodule(i));
  };
  virtual ~Module() {};
  virtual std::vector<Tensor*> forward(std::vector<Tensor*> &inputs)=0;
  virtual void backward() = 0;
  virtual void to_proto(ModuleParameter *param, bool save_grad = false) const {
    param->Clear();
    param->set_name(name_);
    param->set_type(type_);
    for (int i = 0; i < parameters_.size(); ++i)
      parameters_[i]->to_proto(param->add_param(), save_grad);

    for (int i = 0; i < submodules_.size(); ++i)
      submodules_[i]->to_proto(param->add_submodule(), save_grad);

  }
  virtual void from_proto(const ModuleParameter &param) {
    CHECK_EQ(name_, param.name()) << "name mismatch";
    CHECK_EQ(type_, param.type());
    for (int i = 0; i < param.param_size(); ++i)
      parameters_[i]->from_proto(param.param(i));

    for (int i = 0; i < param.submodule_size(); ++i)
      submodules_[i]->from_proto(param.submodule(i));
  }

  virtual TensorVec get_learnable_params() {
    TensorVec learnable_params;
    for (int i = 0; i < parameters_.size(); ++i)
      if (parameters_[i]->require_grad())
        learnable_params.push_back(parameters_[i]);

    for (int i = 0; i < submodules_.size(); ++i) {
      TensorVec temp = submodules_[i]->get_learnable_params();
      for (int j = 0; j < temp.size(); ++j) {
        learnable_params.push_back(temp[j]);
      }
    }
    return learnable_params;
  }
  virtual inline Mode state(){return state_;}
  virtual inline std::string name(){return name_;}
  virtual inline std::string type(){return type_;}
 protected:
  std::string name_;
  std::string type_;
  Mode state_;
  TensorVec parameters_;
  std::vector<Tensor*> inputs_;
  std::vector<Tensor*> outputs_;
  std::vector<std::shared_ptr<Module>> submodules_;
 DISABLE_COPY_AND_ASSIGN(Module);
};
}//namespace nn
/* save and load*/
void save(const nn::Module *module, const std::string &path);
void load(nn::Module *module, const std::string &path);

}//namespace stensor
#endif //STENSOR_NN_MODULE_HPP_
