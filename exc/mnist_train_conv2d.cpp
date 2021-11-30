/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#include "public/common.hpp"
#include "core/tensor.hpp"
#include "core/math_tesnsor.hpp"
#include "nn/layers.hpp"
#include "optimizer/optimizers.hpp"
#include "read_mnist.hpp"

using namespace std;
using namespace stensor;

class SimpleNet : public nn::Module {
 public:
  SimpleNet(int dim_in, int num_classes, int device_id = 0) {
    int axis = -1;
    state_ = device_id > -1 ? GPU : CPU;
    type_ = "Custom";
    name_ = "SimpleNet";
    nn::Conv2d *conv1 = new nn::Conv2d("conv1", 1, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, device_id, false);
    nn::ReLU *act1 = new nn::ReLU("act1", device_id, true);
    nn::Pooling2d *pool1 = new nn::Pooling2d("pool1", nn::MAXPOOL, 2, 2, device_id);
    nn::Conv2d *conv2 = new nn::Conv2d("conv2", 4, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1, device_id, false);
    nn::ReLU *act2 = new nn::ReLU("act2", device_id, true);
    nn::Pooling2d *pool2 = new nn::Pooling2d("pool2", nn::MAXPOOL, 2, 2, device_id);
    nn::Reshape *reshape = new nn::Reshape("reshape", {-1, 8 * 7 * 7});
    nn::Linear *l1 = new nn::Linear("l1", 8 * dim_in / 16, 64, axis, device_id, true);
    nn::ReLU *act3 = new nn::ReLU("act3", device_id, true);
    nn::Linear *l2 = new nn::Linear("l2", 64, num_classes, axis, device_id, true);

    submodules_.clear();
    submodules_.push_back(std::shared_ptr<nn::Module>(conv1));
    submodules_.push_back(std::shared_ptr<nn::Module>(pool1));
    submodules_.push_back(std::shared_ptr<nn::Module>(conv2));
    submodules_.push_back(std::shared_ptr<nn::Module>(pool2));
    submodules_.push_back(std::shared_ptr<nn::Module>(l1));
    submodules_.push_back(std::shared_ptr<nn::Module>(l2));
    submodules_.push_back(std::shared_ptr<nn::Module>(act1));
    submodules_.push_back(std::shared_ptr<nn::Module>(act2));
    submodules_.push_back(std::shared_ptr<nn::Module>(act3));
    submodules_.push_back(std::shared_ptr<nn::Module>(reshape));

    for (auto &sm: submodules_) {
      modules[sm->name()] = sm;
    }
  };
  ~SimpleNet() {};
  inline nn::TensorVec forward(nn::TensorVec &inputs) override {//image and ground-truth
    inputs_.clear();
    inputs_.push_back(inputs[0]);
    nn::TensorVec x;
    nn::TensorVec x_reshape;
    std::vector<int> orig_shape;
    x = modules["conv1"]->forward(inputs_);
    x = modules["act1"]->forward(x);
    x = modules["pool1"]->forward(x);
    x = modules["conv2"]->forward(x);
    x = modules["act2"]->forward(x);
    x = modules["pool2"]->forward(x);
    x = modules["reshape"]->forward(x);
    x = modules["l1"]->forward(x);
    x = modules["act3"]->forward(x);
    x = modules["l2"]->forward(x);
    return x;
  }

  inline void backward() override {
    modules["l2"]->backward();
    modules["act3"]->backward();
    modules["l1"]->backward();
    modules["reshape"]->backward();
    modules["pool2"]->backward();
    modules["act2"]->backward();
    modules["conv2"]->backward();
    modules["pool1"]->backward();
    modules["act1"]->backward();
    modules["conv1"]->backward();
  }
 private:
  std::map<std::string, std::shared_ptr<nn::Module>> modules;
};
int calc_acc(const Tensor *logit, const Tensor *gt) {
  Tensor *pred = stensor::argmax(logit, -1);
  CHECK_EQ(pred->size(), gt->size());
  int count = 0;
  const float *gt_data = gt->const_data();
  const float *pred_data = pred->const_data();
  for (int i = 0; i < pred->size(); ++i) {
    if (*gt_data == *pred_data) count++;
    gt_data++;
    pred_data++;
  }
  delete pred;
  return count;
}

int main() {
  stensor::Config::set_random_seed(1234);
  int batch_size = 64;
  int device_id = 0; // using GPU0
  SimpleNet net(28 * 28, 10, device_id);
  nn::CrossEntropyLoss loss("loss", -1, device_id);

//  stensor::optim::SGD optimizer(&net, 0.001, 1e-4, 0.9);
//  stensor::optim::AdaGrad optimizer(&net, 0.001, 1e-4, 1e-7);
  stensor::optim::Adam optimizer(&net, 0.001, 1e-4, 0.9, 0.999);
//  stensor::optim::RMSprop optimizer(&net, 0.001, 1e-4, 0.9);

  string mnist_root("/home/wss/CLionProjects/stensor/data/mnist/");
  nn::TensorVec mnist_data(
      read_Mnist_Images_to_Tensor(mnist_root + "train-images-idx3-ubyte", batch_size));
  nn::TensorVec mnist_label(
      read_Mnist_Label2Tensor(mnist_root + "train-labels-idx1-ubyte", batch_size));
  long long start_t = systemtime_ms();
  for (int e = 0; e < 30; ++e) {
    float correct_count = 0;
    for (int i = 0; i < mnist_data.size(); ++i) {
      optimizer.zero_grad();
      nn::TensorVec in;
      nn::SharedTensor img = mnist_data[i];
      if (e == 0)
        stensor::scale(img.get(), 1 / 255.0, img.get());
      img->reshape({batch_size, 1, 28, 28});
      nn::SharedTensor gt = mnist_label[i];
      gt->reshape({batch_size});
      if (device_id > -1) {
        if (img->state() == stensor::CPU)
          img->to_gpu();
        if (gt->state() == stensor::CPU)
          gt->to_gpu();
      }

      in.push_back(img);

      nn::TensorVec logit = net.forward(in);
      nn::TensorVec pair{logit[0], gt};
      loss.forward(pair);
      loss.backward();
      net.backward();
      optimizer.step();
      correct_count += calc_acc(logit[0].get(), gt.get());
    }
    std::cout << "epoch:" << e << ", loss:" << loss.get_loss()
              << ", training acc:" << correct_count / 60000.0 << std::endl;
  }
  LOG(INFO) << "Time cost:" << (systemtime_ms() - start_t) / 1000 << " s\n";
  return 0;
}
