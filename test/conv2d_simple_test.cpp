#include <vector>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

extern "C" {
#include "src/basic/factories.h"
#include "src/nn/conv2d_simple.h"
}

namespace aitisa_api {
namespace {

TEST(Conv2DSimple, Float_case1) {
  Tensor input, filter, output;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t input_dims[4] = {2, 2, 5, 5};
  int64_t filter_dims[4] = {3, 2, 3, 3};
  aitisa_full(dtype, device, input_dims, 4, 1.0, &input);
  aitisa_full(dtype, device, filter_dims, 4, 1.0, &filter);

  int ndim = aitisa_tensor_ndim(input);

  aitisa_conv2d_simple(input, filter, &output);
  float* output_data = (float*)aitisa_tensor_data(output);
  int64_t output_size = aitisa_tensor_size(output);
  for (int i = 0; i < output_size; ++i) {
    EXPECT_EQ(18.0, output_data[i]);
  }

  aitisa_destroy(&input);
  aitisa_destroy(&filter);
  aitisa_destroy(&output);
}
}  // namespace
}  // namespace aitisa_api
