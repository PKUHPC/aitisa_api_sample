#include "gtest/gtest.h"
extern "C"
{
#include "src/basic/factories.h"
#include "src/math/matmul_simple.h"
    // #include "src/tool/tool.h"
}

void matmul_simple_assign_float(Tensor t)
{
    int64_t ndim = aitisa_tensor_ndim(t);
    int64_t *dims = aitisa_tensor_dims(t);
    int64_t size = aitisa_tensor_size(t);
    float *data = (float *)aitisa_tensor_data(t);
    float value = 0.0;
    for (int i = 0; i < size; ++i)
    {
        value = i * 0.1;
        data[i] = value;
    }
}

void matmul_simple_assign_double(Tensor t)
{
    int64_t ndim = aitisa_tensor_ndim(t);
    int64_t *dims = aitisa_tensor_dims(t);
    int64_t size = aitisa_tensor_size(t);
    double *data = (double *)aitisa_tensor_data(t);
    double value = 0.0;
    for (int i = 0; i < size; ++i)
    {
        value = i * 0.1;
        data[i] = value;
    }
}

void matmul_simple_assign_int(Tensor t)
{
    int64_t ndim = aitisa_tensor_ndim(t);
    int64_t *dims = aitisa_tensor_dims(t);
    int64_t size = aitisa_tensor_size(t);
    int *data = (int *)aitisa_tensor_data(t);
    int value = 0;
    for (int i = 0; i < size; ++i)
    {
        value = i;
        data[i] = value;
    }
}

namespace aitisa_api
{
    namespace
    {
        TEST(Matmul_float_case1, matrix_matrix)
        {
            Tensor tensor1;
            Tensor tensor2;
            DataType dtype = kFloat;
            Device device = {DEVICE_CPU, 0};
            int64_t dims1[2] = {5, 4};
            int64_t dims2[2] = {4, 3};
            aitisa_full(dtype, device, dims1, 2, 0.0, &tensor1);
            aitisa_full(dtype, device, dims2, 2, 0.0, &tensor2);
            matmul_simple_assign_float(tensor1);
            matmul_simple_assign_float(tensor2);
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

	TEST(Matmul_double_case1, matrix_matrix)
        {
            Tensor tensor1;
            Tensor tensor2;
            DataType dtype = kDouble;
            Device device = {DEVICE_CPU, 0};
            int64_t dims1[2] = {5, 4};
            int64_t dims2[2] = {4, 3};
            aitisa_full(dtype, device, dims1, 2, 0.0, &tensor1);
            aitisa_full(dtype, device, dims2, 2, 0.0, &tensor2);
            matmul_simple_assign_double(tensor1);
            matmul_simple_assign_double(tensor2);
            Tensor output;
            aitisa_matmul_simple(tensor1, tensor2, &output);
            
            int64_t expected_ndim = 2;
            int64_t expected_dims[2] = {5, 3};
            EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
            for (int i = 0; i < expected_ndim; ++i)
            {
                EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
            }
            double *data = (double *)aitisa_tensor_data(output);
            double result[15] = {0.42, 0.48, 0.54, 1.14, 1.36, 1.58, 1.86, 2.24,
                                2.62, 2.58, 3.12, 3.66, 3.3, 4, 4.7};
            for (int i = 0; i < aitisa_tensor_size(output); ++i)
            {
                EXPECT_DOUBLE_EQ(result[i], data[i]);
            }
            aitisa_destroy(&tensor1);
            aitisa_destroy(&tensor2);
            aitisa_destroy(&output);
        }

        TEST(Matmul_int_case1, matrix_matrix)
        {
            Tensor tensor1;
            Tensor tensor2;
            DataType dtype = kInt32;
            Device device = {DEVICE_CPU, 0};
            int64_t dims1[2] = {5, 4};
            int64_t dims2[2] = {4, 3};
            aitisa_full(dtype, device, dims1, 2, 0, &tensor1);
            aitisa_full(dtype, device, dims2, 2, 0, &tensor2);
            matmul_simple_assign_int(tensor1);
            matmul_simple_assign_int(tensor2);
            Tensor output;
            aitisa_matmul_simple(tensor1, tensor2, &output);

            int64_t expected_ndim = 2;
            int64_t expected_dims[2] = {5, 3};
            EXPECT_EQ(expected_ndim, aitisa_tensor_ndim(output));
            for (int i = 0; i < expected_ndim; ++i)
            {
                EXPECT_EQ(expected_dims[i], aitisa_tensor_dim(output, i));
            }
            int *data = (int *)aitisa_tensor_data(output);
            int result[15] = {42, 48, 54, 114, 136, 158, 186, 224,
                                262, 258, 312, 366, 330, 400, 470};
            for (int i = 0; i < aitisa_tensor_size(output); ++i)
            {
                //EXPECT_DOUBLE_EQ(result[i], data[i]);
                EXPECT_TRUE(abs(result[i] - data[i]) < 0.000001);
            }
            aitisa_destroy(&tensor1);
            aitisa_destroy(&tensor2);
            aitisa_destroy(&output);
        }


    } // namespace
} // namespace aitisa_api
