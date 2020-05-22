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

int main(int argc, const char* argv[]) {
  int gpu_id = 0;
  std::cerr << "GPU: " << gpu_id << std::endl;

  std::vector<short> image(32 * 300 * 2944);
  int rows = 300;
  int cols = 2944;

  cudaSetDevice(gpu_id);

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_HALF,
                                        /*batch_size=*/1,
                                        /*channels=*/32,
                                        /*image_height=*/rows,
                                        /*image_width=*/cols));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_HALF,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/32,
                                        /*in_channels=*/32,
                                        /*kernel_height=*/3,
                                        /*kernel_width=*/3));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/1,
                                             /*pad_width=*/1,
                                             /*vertical_stride=*/1,
                                             /*horizontal_stride=*/1,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/CUDNN_CROSS_CORRELATION,
                                             /*computeType=*/CUDNN_DATA_HALF));
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
                                        /*dataType=*/CUDNN_DATA_HALF,
                                        /*batch_size=*/1,
                                        /*channels=*/32,
                                        /*image_height=*/rows,
                                        /*image_width=*/cols));

  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  checkCUDNN(
      cudnnGetConvolutionForwardAlgorithm(cudnn,
                                          input_descriptor,
                                          kernel_descriptor,
                                          convolution_descriptor,
                                          output_descriptor,
                                          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                          /*memoryLimitInBytes=*/0,
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

  int image_bytes = batch_size * 32 * height * width * sizeof(short);
  int output_bytes = batch_size * 32 * height * width * sizeof(short);

  short* d_input{nullptr};
  cudaMalloc(&d_input, image_bytes);
  cudaMemcpy(d_input, image.data(), image_bytes, cudaMemcpyHostToDevice);

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

  short h_kernel[32][3][3][3];
  for (int kernel = 0; kernel < 32; ++kernel) {
    for (int channel = 0; channel < 32; ++channel) {
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
