/**
* Copyright 2021 wss
* Created by wss on 11月,21, 2021
*/
#include "module.hpp"

namespace stensor {
void save(const nn::Module *module, const std::string &path){
  ModuleParameter module_parameter;
  module->to_proto(&module_parameter);
  std::fstream out_file(path, std::ios::out | std::ios::trunc | std::ios::binary);
  bool success = module_parameter.SerializeToOstream(&out_file);
  CHECK(success) << "Failed to save module to " << path;
}
void load(nn::Module *module, const std::string &path){
  ModuleParameter module_parameter;
  std::fstream in_file(path, std::ios::in | std::ios::binary);
  bool success = module_parameter.ParseFromIstream(&in_file);
  module->from_proto(module_parameter);
  CHECK(success) << "Failed to load module from " << path;
}
namespace nn {

}//namespace nn

}//namespace stensor