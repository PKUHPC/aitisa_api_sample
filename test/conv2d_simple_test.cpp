#include <vector>
#include "test/test_data/read_and_write_data.h"

extern "C" {
#include "src/basic/factories.h"
#include "src/nn/conv2d_simple.h"
}
template <class datatype>
void assign(Tensor t, std::vector<datatype> input_data) {
  int64_t size = aitisa_tensor_size(t);
  auto* tensor_data = (float*)aitisa_tensor_data(t);
  for (int i = 0; i < size; ++i) {
    tensor_data[i] = input_data[i];
  }
}
int main(int argc, char** argv) {
  Tensor input, filter, output;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t input_ndim, filter_ndim, result_ndim, input_dtype, filter_dtype,
      result_dtype, input_num, filter_num, result_num;
  std::vector<int64_t> input_dims_vector, filter_dims_vector,
      result_dims_vector;
  std::vector<float> input_data_vector, filter_data_vector, result_data_vector;
  read_data("../../test/test_data/input/Conv_float_case1_input.dat",
            &input_ndim, &input_dims_vector, &input_dtype, &input_num,
            &input_data_vector);
  read_data("../../test/test_data/input/Conv_float_case1_filter.dat",
            &filter_ndim, &filter_dims_vector, &filter_dtype, &filter_num,
            &filter_data_vector);

  int64_t* input_dims;
  input_dims = (int64_t*)malloc(sizeof(int64_t) * input_ndim);
  memset(input_dims, 0, sizeof(float) * input_ndim);
  for (int i = 0; i < input_ndim; i++) {
    input_dims[i] = input_dims_vector[i];
  }

  int64_t* filter_dims;
  filter_dims = (int64_t*)malloc(sizeof(int64_t) * filter_ndim);
  memset(filter_dims, 0, sizeof(float) * filter_ndim);
  for (int i = 0; i < filter_ndim; i++) {
    filter_dims[i] = filter_dims_vector[i];
  }

  aitisa_create(dtype, device, input_dims, 4, NULL, 0, &input);
  aitisa_create(dtype, device, filter_dims, 4, NULL, 0, &filter);
  assign(input, input_data_vector);
  assign(filter, filter_data_vector);
  aitisa_conv2d_simple(input, filter, &output);

  auto output_data = (float*)aitisa_tensor_data(output);
  read_data("../../test/test_data/ans/Conv_float_case1.dat", &result_ndim,
            &result_dims_vector, &result_dtype, &result_num,
            &result_data_vector);

  int64_t output_size = aitisa_tensor_size(output);
  for (int i = 0; i < output_size; ++i) {
    if (output_data[i] != result_data_vector[i]) {
      std::cout << " mismatch" << std::endl;
      exit(1);
    }
  }
  free(input_dims);
  free(filter_dims);

  aitisa_destroy(&input);
  aitisa_destroy(&filter);
  aitisa_destroy(&output);
}
