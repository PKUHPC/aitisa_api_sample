#include <iostream>
//#include "test/test_data/read_data.h"
//extern "C" {
//#include "src/basic/factories.h"
//#include "src/math/matmul_simple.h"
//}
//
//void matmul_simple_assign_float(Tensor t) {
//  int64_t ndim = aitisa_tensor_ndim(t);
//  int64_t* dims = aitisa_tensor_dims(t);
//  int64_t size = aitisa_tensor_size(t);
//  float* data = (float*)aitisa_tensor_data(t);
//  float value = 0.0;
//  for (int i = 0; i < size; ++i) {
//    value = i * 0.1;
//    data[i] = value;
//  }
//}
//
//void matmul_simple_assign_double(Tensor t) {
//  int64_t ndim = aitisa_tensor_ndim(t);
//  int64_t* dims = aitisa_tensor_dims(t);
//  int64_t size = aitisa_tensor_size(t);
//  double* data = (double*)aitisa_tensor_data(t);
//  double value = 0.0;
//  for (int i = 0; i < size; ++i) {
//    value = i * 0.1;
//    data[i] = value;
//  }
//}
//
//void matmul_simple_assign_int(Tensor t) {
//  int64_t ndim = aitisa_tensor_ndim(t);
//  int64_t* dims = aitisa_tensor_dims(t);
//  int64_t size = aitisa_tensor_size(t);
//  int* data = (int*)aitisa_tensor_data(t);
//  int value = 0;
//  for (int i = 0; i < size; ++i) {
//    value = i;
//    data[i] = value;
//  }
//}
//
//namespace aitisa_api {
//namespace {
//TEST(Matmul_float_case1, matrix_matrix) {
//  Tensor tensor1;
//  Tensor tensor2;
//  DataType dtype = kFloat;
//  Device device = {DEVICE_CPU, 0};
//  int64_t dims1[2] = {5, 4};
//  int64_t dims2[2] = {4, 3};
//  aitisa_full(dtype, device, dims1, 2, 0.0, &tensor1);
//  aitisa_full(dtype, device, dims2, 2, 0.0, &tensor2);
//  matmul_simple_assign_float(tensor1);
//  matmul_simple_assign_float(tensor2);
//  Tensor output;
//  aitisa_matmul_simple(tensor1, tensor2, &output);
//
//  int64_t expected_ndim = 2;
//  int64_t expected_dims[2] = {5, 3};
//  EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
//  for (int i = 0; i < expected_ndim; ++i) {
//    EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
//  }
//  auto* data = (float*)aitisa_tensor_data(output);
//  char path[] = "../../test/test_data/Matmul_float_case1.dat";
//  float result[15];
//  read_date(path, result);
//
//  for (int i = 0; i < aitisa_tensor_size(output); ++i) {
//    EXPECT_FLOAT_EQ(result[i], data[i]);
//  }
//  aitisa_destroy(&tensor1);
//  aitisa_destroy(&tensor2);
//  aitisa_destroy(&output);
//}
//
//TEST(Matmul_double_case1, matrix_matrix) {
//  Tensor tensor1;
//  Tensor tensor2;
//  DataType dtype = kDouble;
//  Device device = {DEVICE_CPU, 0};
//  int64_t dims1[2] = {5, 4};
//  int64_t dims2[2] = {4, 3};
//  aitisa_full(dtype, device, dims1, 2, 0.0, &tensor1);
//  aitisa_full(dtype, device, dims2, 2, 0.0, &tensor2);
//  matmul_simple_assign_double(tensor1);
//  matmul_simple_assign_double(tensor2);
//  Tensor output;
//  aitisa_matmul_simple(tensor1, tensor2, &output);
//
//  int64_t expected_ndim = 2;
//  int64_t expected_dims[2] = {5, 3};
//  EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
//  for (int i = 0; i < expected_ndim; ++i) {
//    EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
//  }
//  auto* data = (double*)aitisa_tensor_data(output);
//  char path[] = "../../test/test_data/Matmul_double_case1.dat";
//  double result[15];
//  read_date(path, result);
//  for (int i = 0; i < aitisa_tensor_size(output); ++i) {
//    EXPECT_DOUBLE_EQ(result[i], data[i]);
//  }
//  aitisa_destroy(&tensor1);
//  aitisa_destroy(&tensor2);
//  aitisa_destroy(&output);
//}
//
//TEST(Matmul_int_case1, matrix_matrix) {
//  Tensor tensor1;
//  Tensor tensor2;
//  DataType dtype = kInt32;
//  Device device = {DEVICE_CPU, 0};
//  int64_t dims1[2] = {5, 4};
//  int64_t dims2[2] = {4, 3};
//  aitisa_full(dtype, device, dims1, 2, 0, &tensor1);
//  aitisa_full(dtype, device, dims2, 2, 0, &tensor2);
//  matmul_simple_assign_int(tensor1);
//  matmul_simple_assign_int(tensor2);
//  Tensor output;
//  aitisa_matmul_simple(tensor1, tensor2, &output);
//
//  int64_t expected_ndim = 2;
//  int64_t expected_dims[2] = {5, 3};
//  EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
//  for (int i = 0; i < expected_ndim; ++i) {
//    EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
//  }
//  auto data = (int*)aitisa_tensor_data(output);
//  char path[] = "../../test/test_data/Matmul_int_case1.dat";
//  int result[15];
//  read_date(path, result);
//  for (int i = 0; i < aitisa_tensor_size(output); ++i) {
//    EXPECT_TRUE(abs(result[i] - data[i]) < 0.000001);
//  }
//  aitisa_destroy(&tensor1);
//  aitisa_destroy(&tensor2);
//  aitisa_destroy(&output);
//}
//
//}  // namespace
//}  // namespace aitisa_api
int main(int argc, char **argv) {
  std::cout << "matmul" << std::endl;
  return 0;
}
