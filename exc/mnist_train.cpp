/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#include "public/common.hpp"
#include "core/tensor.hpp"
#include "core/math_tesnsor.hpp"
#include "nn/layers.hpp"
#include "optimizer/optimizers.hpp"

using namespace std;
using namespace stensor;

int ReverseInt(int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

nn::TensorVec read_Mnist_Label2Tensor(const string &filename, int batch_size = 64) {
  ifstream file(filename, ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    file.read((char *) &magic_number, sizeof(magic_number));
    file.read((char *) &number_of_images, sizeof(number_of_images));
    magic_number = ReverseInt(magic_number);
    number_of_images = ReverseInt(number_of_images);
    cout << "magic number = " << magic_number << endl;
    cout << "number of images = " << number_of_images << endl;
    nn::TensorVec out(number_of_images / batch_size);
    for (int i = 0; i < out.size(); i++) {
      Tensor *out_tensor = new stensor::Tensor({batch_size, 1});
      out[i].reset(out_tensor);
      float *data = out_tensor->data();
      for (int j = 0; j < batch_size; ++j) {
        unsigned char label = 0;
        file.read((char *) &label, sizeof(label));
        *data = (float) label;
        data++;
      }

    }
    return out;
  }
  return {};
}

nn::TensorVec read_Mnist_Images_to_Tensor(const string &filename, int batch_size = 64) {
  ifstream file(filename, ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    unsigned char label;
    file.read((char *) &magic_number, sizeof(magic_number));
    file.read((char *) &number_of_images, sizeof(number_of_images));
    file.read((char *) &n_rows, sizeof(n_rows));
    file.read((char *) &n_cols, sizeof(n_cols));
    magic_number = ReverseInt(magic_number);
    number_of_images = ReverseInt(number_of_images);
    n_rows = ReverseInt(n_rows);
    n_cols = ReverseInt(n_cols);

    cout << "magic number = " << magic_number << endl;
    cout << "number of images = " << number_of_images << endl;
    cout << "rows = " << n_rows << endl;
    cout << "cols = " << n_cols << endl;
    nn::TensorVec out(number_of_images / batch_size);

    for (int i = 0; i < out.size(); i++) {
      Tensor *out_tensor = new stensor::Tensor({batch_size, n_rows * n_cols});
      out[i].reset(out_tensor);
      float *data = out_tensor->data();
      for (int j = 0; j < batch_size; ++j) {
        for (int r = 0; r < n_rows; r++) {
          for (int c = 0; c < n_cols; c++) {
            unsigned char pixel = 0;
            file.read((char *) &pixel, sizeof(pixel));
            *data = (float) pixel;
            data++;
          }
        }
      }
    }
    return out;
  }
  return {};
}

class SimpleNet : public nn::Module {
 public:
  SimpleNet(int dim_in, int num_classes, int device_id = 0) {
    int axis = -1;
    state_ = device_id > -1 ? GPU : CPU;
    type_ = "Custom";
    name_ = "SimpleNet";
    nn::Linear *l1 = new nn::Linear("l1", dim_in, 64, axis, device_id, true);
    nn::ReLU *act = new nn::ReLU("act");
    nn::Linear *l2 = new nn::Linear("l2", 64, num_classes, axis, device_id, true);

    submodules_.clear();
    submodules_.push_back(std::shared_ptr<nn::Module>(l1));
    submodules_.push_back(std::shared_ptr<nn::Module>(l2));
    submodules_.push_back(std::shared_ptr<nn::Module>(act));

    for (auto &sm: submodules_) {
      modules[sm->name()] = sm;
    }
  };
  ~SimpleNet() {};
  inline nn::TensorVec forward(nn::TensorVec &inputs) override {//image and ground-truth
    inputs_.clear();
    inputs_.push_back(inputs[0]);
    nn::TensorVec x;
    x = modules["l1"]->forward(inputs_);
    x = modules["act"]->forward(x);
    x = modules["l2"]->forward(x);
    return x;
  }
  inline void backward() override {
    modules["l2"]->backward();
    modules["act"]->backward();
    modules["l1"]->backward();
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
//  stensor::optim::Adam optimizer(&net, 0.001, 1e-4, 0.9, 0.999);
  stensor::optim::RMSprop optimizer(&net, 0.001, 1e-4, 0.9);

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
      img->reshape({batch_size, 28 * 28});
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
