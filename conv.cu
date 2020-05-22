#include <cudnn.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

typedef float DataType;

int main(int argc, const char* argv[]) {
  cudaSetDevice(0);

  int in = 1, ic = 32, irow = 256, icol = 2944;
  int on = 1, oc = 32, orow = 128, ocol = 1472;
  cudnnDataType_t cudnn_type = sizeof(DataType) == 4 ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF;

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/cudnn_type,
                                        /*batch_size=*/in,
                                        /*channels=*/ic,
                                        /*image_height=*/irow,
                                        /*image_width=*/icol));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/cudnn_type,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/oc,
                                        /*in_channels=*/ic,
                                        /*kernel_height=*/3,
                                        /*kernel_width=*/3));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/1,
                                             /*pad_width=*/1,
                                             /*vertical_stride=*/2,
                                             /*horizontal_stride=*/2,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/CUDNN_CROSS_CORRELATION,
                                             /*computeType=*/cudnn_type));
  checkCUDNN(cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

  int batch_size{0}, channels{0}, height{0}, width{0};
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   &batch_size,
                                                   &channels,
                                                   &height,
                                                   &width));

  std::cerr << "Output Image: " << channels << " x " << height << " x " << width
            << std::endl;

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/cudnn_type,
                                        /*batch_size=*/on,
                                        /*channels=*/oc,
                                        /*image_height=*/orow,
                                        /*image_width=*/ocol));

  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  checkCUDNN(
      cudnnGetConvolutionForwardAlgorithm(cudnn,
                                          input_descriptor,
                                          kernel_descriptor,
                                          convolution_descriptor,
                                          output_descriptor,
                                          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                          /*memoryLimitInBytes=*/2 * 1024 * 1024 * 1024,
                                          &convolution_algorithm));
  printf("%d\n", convolution_algorithm);
  //convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
  //printf("%d\n", convolution_algorithm);

  size_t workspace_bytes{0};
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     &workspace_bytes));
  std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
            << std::endl;
  assert(workspace_bytes > 0);

  void* d_workspace{nullptr};
  cudaMalloc(&d_workspace, workspace_bytes);

  int image_bytes = in * ic * irow * icol * sizeof(DataType);
  int output_bytes = on * oc * orow * ocol * sizeof(DataType);

  short* d_input{nullptr};
  cudaMalloc(&d_input, image_bytes);
  cudaMemset(d_input, 1, image_bytes);

  short* d_output{nullptr};
  cudaMalloc(&d_output, output_bytes);
  cudaMemset(d_output, 0, output_bytes);

  // clang-format off
  const short kernel_template[3][3] = {
    {1, 1, 1},
    {1, -8, 1},
    {1, 1, 1}
  };
  // clang-format on

  short h_kernel[256][256][3][3];
  for (int kernel = 0; kernel < ic; ++kernel) {
    for (int channel = 0; channel < oc; ++channel) {
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          h_kernel[kernel][channel][row][column] = kernel_template[row][column];
        }
      }
    }
  }

  short* d_kernel{nullptr};
  cudaMalloc(&d_kernel, sizeof(h_kernel));
  cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

  const float alpha = 1.0f, beta = 0.0f;

  cudaEvent_t start_event, stop_event;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEventRecord(start_event, 0);
  checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     d_input,
                                     kernel_descriptor,
                                     d_kernel,
                                     convolution_descriptor,
                                     convolution_algorithm,
                                     d_workspace,
                                     workspace_bytes,
                                     &beta,
                                     output_descriptor,
                                     d_output));
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  float time;
  cudaEventElapsedTime(&time, start_event, stop_event);
  printf("conv took %f ms\n", time);
/*
  short* h_output = new short[output_bytes];
  cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);

  delete[] h_output;
  */
  cudaFree(d_kernel);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_workspace);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);

  cudnnDestroy(cudnn);
}
