#include "gtest/gtest.h"
extern "C" {
#include "src/img/resize2d_bilinear.h"
// #include "src/tool/tool.h"
}

void resize2d_bilinear_assign_float(Tensor t) {
  int64_t size = aitisa_tensor_size(t);
  float* data = (float*)aitisa_tensor_data(t);
  float value = 0;
  for (int i = 0; i < size; ++i) {
    value = i * 0.1;
    data[i] = value;
  }
}

namespace aitisa_api {
namespace {

TEST(Resize, Float2d_case1) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[2] = {3, 3};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  resize2d_bilinear_assign_float(input);

  Tensor output;
  aitisa_resize2d_bilinear(input, 5, 5, &output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.0,  0.18, 0.16, 0.34, 0.2,  0.06, 0.24, 0.22, 0.4,
                       0.2,  0.32, 0.5,  0.48, 0.66, 0.5,  0.38, 0.56, 0.54,
                       0.72, 0.5,  0.6,  0.6,  0.7,  0.7,  0.8};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}

TEST(Resize, Float2d_case2) {
  Tensor input;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t dims[2] = {5, 5};
  aitisa_create(dtype, device, dims, 2, NULL, 0, &input);
  resize2d_bilinear_assign_float(input);

  Tensor output;
  aitisa_resize2d_bilinear(input, 3, 3, &output);

  float* out_data = (float*)aitisa_tensor_data(output);
  float test_data[] = {0.,         0.43333333, 0.46666667, 0.56666667, 1.,
                       1.03333333, 1.53333333, 1.96666667, 2.};
  int64_t size = aitisa_tensor_size(output);
  for (int64_t i = 0; i < size; i++) {
    /* Due to the problem of precision, consider the two numbers
       are equal when their difference is less than 0.000001*/
    EXPECT_TRUE(abs(out_data[i] - test_data[i]) < 0.000001);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api