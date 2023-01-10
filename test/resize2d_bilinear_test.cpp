#include "gtest/gtest.h"
#include "test/test_data/read_data.h"
extern "C" {
#include "src/nn/resize2d_bilinear.h"
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

  auto out_data = (float*)aitisa_tensor_data(output);
  int64_t size = aitisa_tensor_size(output);
  char path[] = "../../test/test_data/Resize_float2d_case1.dat";
  float test_data[25];
  read_date(path, test_data);
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

  auto out_data = (float*)aitisa_tensor_data(output);
  char path[] = "../../test/test_data/Resize_float2d_case2.dat";
  float test_data[9];
  read_date(path, test_data);
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