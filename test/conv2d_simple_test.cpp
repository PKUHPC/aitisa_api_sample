#include <vector>
#include "test/test_data/read_and_write_data.h"

extern "C" {
#include "src/basic/factories.h"
#include "src/nn/conv2d_simple.h"
}
template<class datetype>
void assign(Tensor t,datetype* input_data) {
  int64_t size = aitisa_tensor_size(t);
  auto* tensor_data = (float*)aitisa_tensor_data(t);
  for (int i = 0; i < size; ++i) {
    tensor_data[i] = input_data[i];
  }
}
int main(int argc, char **argv) {
  Tensor input, filter, output;
  DataType dtype = kFloat;
  Device device = {DEVICE_CPU, 0};
  int64_t input_dims[4] = {2, 2, 5, 5};
  int64_t filter_dims[4] = {3, 2, 3, 3};
  float input_date[100];
  float filter_date[54];

  read_date("../../test/test_data/input/Conv_float_case1_input.dat",input_date);
  read_date("../../test/test_data/input/Conv_float_case1_filter.dat",filter_date);

  aitisa_create(dtype, device, input_dims, 4, NULL, 0,&input);
  aitisa_create(dtype, device, filter_dims, 4, NULL, 0,&filter);
  assign(input,input_date);
  assign(filter,filter_date);
  aitisa_conv2d_simple(input, filter, &output);

  auto output_data = (float*)aitisa_tensor_data(output);
  float result[54];
  read_date("../../test/test_data/ans/Conv_float_case1.dat", result);
  int64_t output_size = aitisa_tensor_size(output);
  for (int i = 0; i < output_size; ++i) {
    if(output_data[i] != result[i]){
      std::cout << " mismatch" << std::endl;
      exit(1);
    }
  }
  aitisa_destroy(&input);
  aitisa_destroy(&filter);
  aitisa_destroy(&output);
}
