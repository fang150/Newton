/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_MATMUL_OP_H_
#define TENSORFLOW_KERNELS_MATMUL_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"


#define GRADIENT_SAMPLING_SIZE               20
#define HESSIAN_SAMPLING_SIZE		  5


namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct FtorComputeFx {
  void operator()(  const Device& d,  T* devptr,int rows,int cols,int num_classes,  const T* target, T* indicatorVal, int NUM_THREADS  , T* maxdots, int BLOCKS, int BLOCK_SIZE, int BLOCKS_POW_2);
};


template <typename Device, typename T>
struct FtorHxC {
  void operator()(const Device& d,T *A,T *B,T *C, int rows, int cols, int num_classes, int BLOCKS, int BLOCK_SIZE );
};




template <typename Device, typename T>
	struct FtorSampler {
    void operator()(  const Device& d, const T *dataset, int rows, int cols, int *indices, T *dest, int sampleSize, int BLOCK_SIZE   );
  };




template<typename Device,typename T>
struct FtorComputeBlocks {
  void operator()( const Device& d,  int *blocks, int *block_size, int count  );
};

template<typename Device,typename T>
struct FtorComputePOW2 {
  void operator()( const Device& d, int blocks, int *result   );
};

template <typename Device, typename T>
struct FtorComputeHXW {
  void operator()(  const Device& d, T *XW, int rows, int cols, int numclasses, int threads_per_col , int BLOCKS , int BLOCK_SIZE );
};

template <typename Device, typename T>
struct FtorDxSoftmaxInd {
  void operator()(  const Device& d,  T *hxw, const T *target, int rows, int num_classes, T *result, int threads_per_row,int BLOCKS,int BLOCK_SIZE);
};

template <typename Device, typename T>
struct FtorAddRegularizer {
  void operator()(  const Device& d, T *input, T *vector, T lambda, int count, T normalizer, int rblocks, int BLOCK_SIZE   );
};



template <typename Device, typename T>
struct FtorReduce {
  void operator()(  const Device& d,  const  T *input, T *results ,const size_t count, int BLOCKS,int BLOCK_SIZE);
};


template <typename Device, typename T>
	struct FtorReduceLog {
	  void operator()(  const Device& d,  const T *input, T *results, const size_t count, int BLOCKS,int BLOCK_SIZE);
  };



template <typename Device, typename T>
struct FtorReduceVectorWarp{
	  void operator()(  const Device& d,  const T *input, const T *maxdots, T *results, const size_t numcomps, int numblocks, int BLOCKS,int BLOCK_SIZE);

};



template <typename Device, typename T>
struct FtorSoftmaxPredict{
	  void operator()(  const Device& d,  T *test_set, T *weights, 
		int rows, int cols, int numclasses, T *workspace,int pblocks ,int BLOCK_SIZE);

};






}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_MATMUL_OP_H_




