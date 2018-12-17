#include <cstdlib>
#include <cub/cub.cuh>
#include <iostream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <sys/time.h>
#include <time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start) {
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec * USECPSEC) + tv.tv_usec) - start;
}

const size_t ksize = 10000;
const size_t vsize = 10000000;
const int nTPB = 256;

struct my_max_func {
  template <typename T1, typename T2>
  __host__ __device__ T1 operator()(const T1 t1, const T2 t2) {
    T1 res;
    if (thrust::get<0>(t1) > thrust::get<0>(t2)) {
      thrust::get<0>(res) = thrust::get<0>(t1);
      thrust::get<1>(res) = thrust::get<1>(t1);
    } else {
      thrust::get<0>(res) = thrust::get<0>(t2);
      thrust::get<1>(res) = thrust::get<1>(t2);
    }
    return res;
  }
};

// CustomMin functor
struct CustomMax
{
    template <typename T>
    CUB_RUNTIME_FUNCTION __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b < a) ? a : b;
    }
};

typedef union {
  float floats[2];              // floats[0] = maxvalue
  int ints[2];                  // ints[1] = maxindex
  unsigned long long int ulong; // for atomic update
} my_atomics;

__device__ unsigned long long int my_atomicMax(unsigned long long int *address,
                                               float val1, int val2) {
  my_atomics loc, loctest;
  loc.floats[0] = val1;
  loc.ints[1] = val2;
  loctest.ulong = *address;
  while (loctest.floats[0] < val1)
    loctest.ulong = atomicCAS(address, loctest.ulong, loc.ulong);
  return loctest.ulong;
}

__global__ void my_max_idx(const float *data, const int *keys, const int ds,
                           my_atomics *res) {

  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  if (idx < ds)
    my_atomicMax(&(res[keys[idx]].ulong), data[idx], idx);
}

int main() {
  float *h_vals = new float[vsize];
  int *h_keys = new int[vsize];
  for (int i = 0; i < vsize; i++) {
    h_vals[i] = rand();
    h_keys[i] = rand() % ksize;
  }

  // thrust method
  thrust::device_vector<float> d_vals(h_vals, h_vals + vsize);
  thrust::device_vector<int> d_keys(h_keys, h_keys + vsize);
  thrust::device_vector<int> d_keys_out(ksize);
  thrust::device_vector<float> d_vals_out(ksize);
  thrust::device_vector<int> d_idxs(vsize);
  thrust::device_vector<int> d_idxs_out(ksize);

  thrust::sequence(d_idxs.begin(), d_idxs.end());
  cudaDeviceSynchronize();
  unsigned long long et = dtime_usec(0);

  thrust::sort_by_key(d_keys.begin(), d_keys.end(),
                      thrust::make_zip_iterator(
                          thrust::make_tuple(d_vals.begin(), d_idxs.begin())));
  thrust::reduce_by_key(d_keys.begin(), d_keys.end(),
                        thrust::make_zip_iterator(
                            thrust::make_tuple(d_vals.begin(), d_idxs.begin())),
                        d_keys_out.begin(),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            d_vals_out.begin(), d_idxs_out.begin())),
                        thrust::equal_to<int>(), my_max_func());
  cudaDeviceSynchronize();
  et = dtime_usec(et);
  std::cout << "Thrust time: " << et / (float)USECPSEC << "s" << std::endl;

  // cuda method

  float *vals;
  int *keys;
  my_atomics *results;
  cudaMalloc(&keys, vsize * sizeof(int));
  cudaMalloc(&vals, vsize * sizeof(float));
  cudaMalloc(&results, ksize * sizeof(my_atomics));

  cudaMemset(results, 0,
             ksize * sizeof(my_atomics)); // works because vals are all positive
  cudaMemcpy(keys, h_keys, vsize * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(vals, h_vals, vsize * sizeof(float), cudaMemcpyHostToDevice);
  et = dtime_usec(0);

  my_max_idx<<<(vsize + nTPB - 1) / nTPB, nTPB>>>(vals, keys, vsize, results);
  cudaDeviceSynchronize();
  et = dtime_usec(et);
  std::cout << "CUDA time: " << et / (float)USECPSEC << "s" << std::endl;

  // cub method
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  int *keys_out;
  float *vals_out;
  cudaMalloc(&keys_out, vsize * sizeof(int));
  cudaMalloc(&vals_out, vsize * sizeof(float));
  int *d_num_runs_out;
  cudaMalloc(&d_num_runs_out, sizeof(float));
  CustomMax reduction_op;
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
                                 keys, keys_out, vals, vals_out,
                                 d_num_runs_out, reduction_op, vsize);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  std::cout << "temp storage: " << temp_storage_bytes << std::endl;

  et = dtime_usec(0);
  cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
                                 keys, keys_out, vals, vals_out,
                                 d_num_runs_out, reduction_op, vsize);
  cudaDeviceSynchronize();
  et = dtime_usec(et);
  std::cout << "CUB time: " << et / (float)USECPSEC << "s" << std::endl;

  // verification

  my_atomics *h_results = new my_atomics[ksize];
  cudaMemcpy(h_results, results, ksize * sizeof(my_atomics),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < ksize; i++) {
    if (h_results[i].floats[0] != d_vals_out[i]) {
      std::cout << "value mismatch at index: " << i
                << " thrust: " << d_vals_out[i]
                << " CUDA: " << h_results[i].floats[0] << std::endl;
      return -1;
    }
    if (h_results[i].ints[1] != d_idxs_out[i]) {
      std::cout << "index mismatch at index: " << i
                << " thrust: " << d_idxs_out[i]
                << " CUDA: " << h_results[i].ints[1] << std::endl;
      return -1;
    }
  }

  std::cout << "Success!" << std::endl;
  return 0;
}
