#include "gtest/gtest.h"
extern "C"
{
#include "src/basic/factories.h"
#include "src/math/matmul_simple.h"
    // #include "src/tool/tool.h"
}

void natural_assign(Tensor t)
{
    int64_t ndim = aitisa_tensor_ndim(t);
    int64_t *dims = aitisa_tensor_dims(t);
    int64_t size = aitisa_tensor_size(t);
    float *data = (float *)aitisa_tensor_data(t);
    float value = 0;
    for (int i = 0; i < size; ++i)
    {
        value = i * 0.1;
        data[i] = value;
    }
}

namespace aitisa_api
{
    namespace
    {
        TEST(Matmul, matrix_matrix)
        {
            Tensor tensor1;
            Tensor tensor2;
            DataType dtype = kFloat;
            Device device = {DEVICE_CPU, 0};
            int64_t dims1[2] = {5, 4};
            int64_t dims2[2] = {4, 3};
            aitisa_full(dtype, device, dims1, 2, 2.1, &tensor1);
            aitisa_full(dtype, device, dims2, 2, 2.0, &tensor2);
            natural_assign(tensor1);
            natural_assign(tensor2);
            Tensor output;
            aitisa_matmul_simple(tensor1, tensor2, &output);
            /*
            tensor_printer2d(tensor1);
            tensor_printer2d(tensor2);
            tensor_printer2d(output);
            */
            int64_t expected_ndim = 2;
            int64_t expected_dims[2] = {5, 3};
            EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
            for (int i = 0; i < expected_ndim; ++i)
            {
                EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
            }
            float *data = (float *)aitisa_tensor_data(output);
            float result[15] = {0.42, 0.48, 0.54, 1.14, 1.36, 1.58, 1.86, 2.24,
                                2.62, 2.58, 3.12, 3.66, 3.3, 4, 4.7};
            for (int i = 0; i < aitisa_tensor_size(output); ++i)
            {
                EXPECT_FLOAT_EQ(result[i], data[i]);
            }
            aitisa_destroy(&tensor1);
            aitisa_destroy(&tensor2);
            aitisa_destroy(&output);
        }
    } // namespace
} // namespace aitisa_api