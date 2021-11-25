/**
* Copyright 2021 wss
* Created by wss on 11月,25, 2021
*/
#include "public/common.hpp"
#include "core/tensor.hpp"
#include "optimizer/sgd.hpp"
#include "linear_layer.hpp"
#include "softmax_layer.hpp"
#include "cross_entropy_loss_layer.hpp"

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

void read_Mnist_Label(const string &filename, vector<double> &labels) {
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

    for (int i = 0; i < number_of_images; i++) {
      unsigned char label = 0;
      file.read((char *) &label, sizeof(label));
      labels.push_back((double) label);
    }

  }
}
void read_Mnist_Images_to_vector(const string &filename, vector<vector<double>> &images) {
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

    for (int i = 0; i < number_of_images; i++) {
      vector<double> tp;
      for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++) {
          unsigned char image = 0;
          file.read((char *) &image, sizeof(image));
          tp.push_back(image);
        }
      }
      images.push_back(tp);
    }
  }
}

nn::TensorVec read_Mnist_Label2Tensor(const string &filename) {
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
    nn::TensorVec out(number_of_images);
    for (int i = 0; i < number_of_images; i++) {
      Tensor *out_tensor = new stensor::Tensor({1, 1});
      out[i].reset(out_tensor);
      unsigned char label = 0;
      file.read((char *) &label, sizeof(label));
      *out_tensor->data() = (float) label;
    }
    return out;
  }
  return {};
}

nn::TensorVec read_Mnist_Images_to_Tensor(const string &filename) {
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
    nn::TensorVec out(number_of_images);

    for (int i = 0; i < number_of_images; i++) {
      Tensor *out_tensor = new stensor::Tensor({n_rows * n_cols});
      out[i].reset(out_tensor);
      float *data = out_tensor->data();
      for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++) {
          unsigned char pixel = 0;
          file.read((char *) &pixel, sizeof(pixel));
          *data = (float) pixel;
          data++;
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
    nn::LinearLayer *l1 = new nn::LinearLayer("l1", dim_in, 10, axis, device_id, true);
//    nn::LinearLayer *l2 = new nn::LinearLayer("l2", 64, num_classes, axis, device_id, true);
//    nn::CrossEntropyLossLayer loss("loss", axis, device_id);
    submodules_.clear();
    submodules_.push_back(std::shared_ptr<nn::Module>(l1));
//    submodules_.push_back(std::shared_ptr<nn::Module>(l2));
//    submodules_.push_back(std::shared_ptr<nn::Module>(&loss));
    for (auto &sm: submodules_) {
      modules[sm->name()] = sm;
    }
  };
  ~SimpleNet() {};
  nn::TensorVec forward(nn::TensorVec &inputs) override {//image and ground-truth
    inputs_.clear();
    inputs_.push_back(inputs[0]);
    modules["l1"];
    nn::TensorVec x1 = modules["l1"]->forward(inputs_);
//    nn::TensorVec x2 = modules["l2"]->forward(x1);
//    nn::TensorVec x3 = modules["loss"]->forward(x2);
    return x1;
  }
  void backward() override {
//    modules["l2"]->backward();
    modules["l1"]->backward();
  }
 private:
  std::map<std::string, std::shared_ptr<nn::Module>> modules;
};

int main() {
  stensor::Config::set_random_seed(1234);
  int device_id = -1;
  SimpleNet net(28 * 28, 10, device_id);
  nn::CrossEntropyLossLayer loss("loss", -1, device_id);

  stensor::optim::SGD sgd(&net, 0.001, 0.0);

  string mnist_root("/home/wss/CLionProjects/stensor/data/mnist/");
  nn::TensorVec mnist_data(
      read_Mnist_Images_to_Tensor(mnist_root + "t10k-images-idx3-ubyte"));
  nn::TensorVec mnist_label(
      read_Mnist_Label2Tensor(mnist_root + "t10k-labels-idx1-ubyte"));
  long long start_t = systemtime_ms();
  for (int e = 0; e < 100; ++e) {
    for (int i = 0; i < mnist_data.size(); ++i) {

      sgd.zero_grad();
      nn::TensorVec in;
      nn::SharedTensor img = mnist_data[i];
      if (e == 0)
        stensor::scale(img.get(), 1 / 255.0, img.get());
      img->reshape({1, 28 * 28});
      nn::SharedTensor gt = mnist_label[i];
      gt->reshape({1});
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
      sgd.step();
    }
    std::cout << "epoch:" << e << ", loss:" << loss.get_loss() << std::endl;
  }
  LOG(INFO) << "Time cost:" << (systemtime_ms() - start_t)/1000 << " s\n";
  return 0;
}