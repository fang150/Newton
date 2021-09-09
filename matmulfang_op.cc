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


// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/matmulfang_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

#if GOOGLE_CUDA
#include "cuda/include/cuda.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

#include <algorithm>
#include <vector>
#include <sys/time.h>
#include <map>
#include <random>
#include <ctime>

namespace tensorflow {

#if GOOGLE_CUDA

namespace {
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace


namespace {
    template <typename T>
    perftools::gputools::DeviceMemoryBase AsBaseMemory(const T* cuda_memory) {
      perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
      
      return wrapped;
    }
    }


#endif  // GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;


template <typename Device, typename T, bool USE_CUBLAS>
struct LaunchBlasDnrm2;
template <typename Device, typename T, bool USE_CUBLAS>
struct LaunchBlasScal;
template <typename Device, typename T, bool USE_CUBLAS>
struct LaunchBlasCopy;
template <typename Device, typename T, bool USE_CUBLAS>
struct LaunchBlasAxpy;
template <typename Device, typename T, bool USE_CUBLAS>
struct LaunchBlasDot;
template <typename Device, typename T, bool USE_CUBLAS>
struct LaunchMemcpyD2D;
template <typename Device, typename T, bool USE_CUBLAS>
struct LaunchMemcpyD2Output;
template <typename Device, typename T, bool USE_CUBLAS>
struct LaunchMemcpyD2H;
template <typename Device, typename T, bool USE_CUBLAS>
struct LaunchMemcpyH2D;

template <typename Device, typename T, bool USE_CUBLAS>
struct LaunchMemZero;


template <typename Device, typename T, bool USE_CUBLAS>
struct ComputeHXW;
template <typename Device, typename T, bool USE_CUBLAS>
struct SoftmaxMulticlassGxOptimized;
template <typename Device, typename T, bool USE_CUBLAS>
struct SoftmaxMulticlassGxSubsampled;
template <typename Device, typename T, bool USE_CUBLAS>
struct SoftmaxMulticlassHxOptimized;
template <typename Device, typename T, bool USE_CUBLAS>
struct SoftmaxMulticlassHxSubsampled;
template <typename Device, typename T, bool USE_CUBLAS>
struct SoftmaxMulticlassFx;
template <typename Device, typename T, bool USE_CUBLAS>
struct SoftmaxPredict;
template <typename Device, typename T, bool USE_CUBLAS>
struct CublasCGMulticlassOptimized;

template <typename Device, typename T, bool USE_CUBLAS>
struct CGLineSearch;


double Get_Time( )
{
  struct timeval tim;
  
  gettimeofday(&tim, NULL );
  return( tim.tv_sec + (tim.tv_usec / 1000000.0) );
}

double Get_Timing_Info( double t_start )
{
  struct timeval tim;
  double t_end;
  
  gettimeofday(&tim, NULL );
  t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
  return (t_end - t_start);
}


//#if GOOGLE_CUDA


namespace functor{
template <>
void FtorComputePOW2<GPUDevice,double>::operator()(const GPUDevice& d, int blocks, int *result  );
extern template struct FtorComputePOW2 <GPUDevice,double>;
}

namespace functor{
template<>
void FtorAddRegularizer<GPUDevice,double>::operator()(  const GPUDevice& d, double *input, double *vector, double lambda, int count, double normalizer, int rblocks, int BLOCK_SIZE   );
extern template struct FtorAddRegularizer<GPUDevice,double>;

}

namespace functor{
template <>
void FtorComputeBlocks<GPUDevice,double>::operator()(const GPUDevice& d, int *blocks, int *block_size, int count  );
extern template struct FtorComputeBlocks<GPUDevice,double>;
}

namespace functor{
template <>
void FtorComputeHXW<GPUDevice,double>::operator()(  const GPUDevice& d, double *XW, int rows, int cols, int numclasses, int threads_per_col, int BLOCKS, int BLOCK_SIZE );
extern template struct FtorComputeHXW<GPUDevice,double>; 
}


namespace functor{
template <>
void FtorHxC<GPUDevice,double>::operator()(const GPUDevice& d,double *A,double *B,double *C, int rows, int cols, int num_classes, int BLOCKS, int BLOCK_SIZE );
extern template struct FtorHxC<GPUDevice,double>;
}


namespace functor{

template <>
  void FtorComputeFx<GPUDevice, double>::operator()(const GPUDevice& d,  double* devptr,int rows,int cols,int num_classes,  const double* target, double* indicatorVal ,int NUM_THREADS  , double* maxdots, int BLOCKS, int BLOCK_SIZE, int BLOCKS_POW_2);
extern template struct FtorComputeFx <GPUDevice,double>;

}

namespace functor{
template <>
void FtorDxSoftmaxInd<GPUDevice,double> ::operator()(  const GPUDevice& d,  double *hxw, const double *target, int rows, int num_classes, double *result, int threads_per_row,int BLOCKS,int BLOCK_SIZE);
extern template struct FtorDxSoftmaxInd<GPUDevice,double>; 
 
}


namespace functor{
  template <>
  void FtorReduce<GPUDevice,double> ::operator()( const GPUDevice& d,  const  double *input, double *results ,const size_t count, int BLOCKS,int BLOCK_SIZE);
  extern template struct FtorReduce<GPUDevice,double>; 
   
}


namespace functor{
  template <>
  void FtorReduceLog<GPUDevice,double> ::operator()( const GPUDevice& d, const double *input, double *results, const size_t count, int BLOCKS,int BLOCK_SIZE);
  extern template struct FtorReduceLog<GPUDevice,double>; 
   
}

namespace functor{
  template <>
  void FtorReduceVectorWarp<GPUDevice,double> ::operator()( const GPUDevice& d,  const double *input, const double *maxdots, double *results, const size_t numcomps, int numblocks, int BLOCKS,int BLOCK_SIZE);
  extern template struct FtorReduceVectorWarp<GPUDevice,double>; 
   
}


namespace functor{
  template <>
  void FtorSoftmaxPredict<GPUDevice,double> ::operator()( const GPUDevice& d,  double *test_set, double *weights, 
		int rows, int cols, int numclasses, double *workspace, int pblocks ,  int BLOCK_SIZE);
  extern template struct FtorSoftmaxPredict<GPUDevice,double>; 
   
}


namespace functor{
template <>

  void FtorSampler<GPUDevice,double> ::operator()( const GPUDevice& d, const double *dataset, int rows, int cols, int *indices, double *dest, int sampleSize, int BLOCK_SIZE);
  extern template struct FtorSampler<GPUDevice,double>; 
}





template <typename T>
struct LaunchMemcpyD2D<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel, const Tensor dst, Tensor src, uint64 size) {

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    auto src_ptr = AsDeviceMemory(src.template flat<T>().data());
    auto dst_ptr = AsDeviceMemory(dst.template flat<T>().data());

    bool blas_norm_status =
        stream->ThenMemcpyD2D(&dst_ptr,src_ptr,size).ok(); //check if argument is correct

  }
};

  template <typename T>
    struct LaunchMemcpyD2Output<GPUDevice, T, true > {
      static void launch(
          OpKernelContext* ctx, OpKernel* kernel,  Tensor* dst, Tensor src, uint64 size) {
    
        auto* stream = ctx->op_device_context()->stream();
        OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));
    
        auto src_ptr = AsDeviceMemory(src.template flat<T>().data());
        auto dst_ptr = AsDeviceMemory(dst->template flat<T>().data());
    
        bool blas_norm_status =
            stream->ThenMemcpyD2D(&dst_ptr,src_ptr,size).ok(); //check if argument is correct
    
      }
    };

template <typename T>
struct LaunchMemcpyD2H<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel, const Tensor in, T* out ,uint64 size) {

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

 
        auto in_ptr = AsBaseMemory(in.template flat<T>().data());
        // Memcpy from gpu src to host dst
        stream->ThenMemcpy( out, in_ptr ,size);

  }
};

template <typename T>
struct LaunchMemcpyH2D<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel,  Tensor dst, T* src ,uint64 size) {

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    auto dst_ptr = AsBaseMemory(dst.template flat<T>().data());

      
    stream->ThenMemcpy( &dst_ptr, src ,size);

  }
};



template <typename T>
struct LaunchBlasDnrm2<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel, const Tensor a, Tensor out,int classes_to_solve, int num_features) {

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    auto a_ptr = AsDeviceMemory(a.template flat<T>().data());
    auto c_ptr = AsDeviceMemory(out.template flat<T>().data());
    bool blas_norm_status =
        stream->ThenBlasNrm2(classes_to_solve*num_features, a_ptr,1,&c_ptr).ok();



  }
};


template <typename T>
struct LaunchBlasScal<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel, T alpha, Tensor out,int classes_to_solve, int num_features) {

    
    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    
    auto c_ptr = AsDeviceMemory(out.template flat<T>().data());

    bool blas_norm_status =                                
        stream->ThenBlasScal(classes_to_solve*num_features, alpha ,&c_ptr,1).ok();

  }
};

template <typename T>
struct LaunchBlasCopy<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel, Tensor in,Tensor out,int classes_to_solve, int num_features) {

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    auto in_ptr = AsDeviceMemory(in.template flat<T>().data());
    auto c_ptr = AsDeviceMemory(out.template flat<T>().data());

    bool blas_norm_status =
        stream->ThenBlasCopy(classes_to_solve*num_features, in_ptr,1,&c_ptr,1).ok();

  }
};


template <typename T>
struct LaunchBlasAxpy<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel, T alpha, Tensor in,Tensor out,int classes_to_solve, int num_features) {

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    auto in_ptr = AsDeviceMemory(in.template flat<T>().data());
    auto out_ptr = AsDeviceMemory(out.template flat<T>().data());

    bool blas_norm_status =
        stream->ThenBlasAxpy(classes_to_solve*num_features, alpha ,in_ptr,1,&out_ptr,1).ok();

  }
};



template <typename T>
struct LaunchMemZero<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel, Tensor in, uint64 size) {

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));


    auto in_ptr = AsBaseMemory(in.template flat<T>().data());

  
        stream->ThenMemZero(&in_ptr,size);

  }
};




template <typename T> 
struct LaunchBlasDot<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel ,Tensor in1,Tensor in2,  Tensor out_gpu,int classes_to_solve, int num_features) {

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    auto in1_ptr = AsDeviceMemory(in1.template flat<T>().data());
    auto in2_ptr = AsDeviceMemory(in2.template flat<T>().data());
    auto base_ptr = AsBaseMemory(out_gpu.template flat<T>().data());
    auto c_ptr = AsDeviceMemory(out_gpu.template flat<T>().data());
    // do blasdot on gpu
    bool blas_norm_status =
        stream->ThenBlasDot(classes_to_solve*num_features, in1_ptr,1,in2_ptr,1 ,&c_ptr).ok();
  }


};


template <typename T>
struct ComputeHXW<GPUDevice, T, true > {
    static void launch(
        OpKernelContext* ctx, OpKernel* kernel, const Tensor& features, const Tensor& weight,
        Tensor XW , int rows, int cols, int num_classes,int BLOCKS, int BLOCK_SIZE, T sampling_type  ) {
      perftools::gputools::blas::Transpose trans[] = {
          perftools::gputools::blas::Transpose::kNoTranspose,
          perftools::gputools::blas::Transpose::kTranspose};
      
      auto* stream = ctx->op_device_context()->stream();
      OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));
  
  
      auto features_ptr = AsDeviceMemory(features.template flat<T>().data());
      auto weight_ptr = AsDeviceMemory(weight.template flat<T>().data());
      auto XW_ptr = AsDeviceMemory(XW.template flat<T>().data());
     
 
      bool blas_launch_status =
          stream->ThenBlasGemm(trans[0], trans[0], rows, num_classes, cols, 1.0f,
            features_ptr, rows, weight_ptr,
                               cols, 0.0f, &XW_ptr, rows)
              .ok();

      if (sampling_type >= 1){
          int blocks = rows / BLOCK_SIZE + (((rows % BLOCK_SIZE) == 0) ? 0 : 1 ); 
               
          
          functor::FtorComputeHXW<GPUDevice,double>()(
            ctx->eigen_device<GPUDevice>(),
            XW.template flat<T>().data(),
            rows ,
            cols,
            num_classes,
            1,
            blocks,
            BLOCK_SIZE   );
          

      } else {


        functor::FtorComputeHXW<GPUDevice,double>()(
          ctx->eigen_device<GPUDevice>(),
          XW.template flat<T>().data(),
          rows ,
          cols,
          num_classes,
          1,
          BLOCKS,BLOCK_SIZE   );
      }	
    }
  };

  template <typename T>
  struct SoftmaxPredict<GPUDevice, T, true > {
      static void launch(
          OpKernelContext* ctx, OpKernel* kernel, const Tensor& test_set, 
          const Tensor& test_labels, const Tensor& weight, int rows, int cols, int numclasses, 
           Tensor temp_hxw, int computeDevice,int BLOCKS,int BLOCK_SIZE,T* out ,int sampling_type ) {

        int pblocks =  (rows / BLOCK_SIZE) + 
            ((rows % BLOCK_SIZE) == 0 ? 0  : 1 );
        T pmax = 0;
        T matches = 0; 
        T nomatches = 0;
        int pclass = -1;
        T sumprob;
        T dot, sumexp, maxdot; 
       
       

        T *h_weights =  (T*) malloc (cols*numclasses  *sizeof(T));        
        T *temp =  (T*) malloc (rows*numclasses *sizeof(T));
        T *h_test_set =  (T*) malloc (rows*cols*sizeof(T));
        T *h_test_labels =  (T*) malloc (rows*sizeof(T));


        LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, test_labels, h_test_labels , sizeof(T)*(rows)  );
        LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, test_set, h_test_set , sizeof(T)*(rows*cols)  );
        if (computeDevice == 1) {


          ComputeHXW<GPUDevice, T, true>::launch(ctx, kernel, test_set, weight,temp_hxw, rows, cols, numclasses ,BLOCKS, BLOCK_SIZE,sampling_type);
          //computeHXW( spTest, test_set, rows, cols, numclasses, weights, devWorkspace, 0 );
          //copy_host_device( temp, devWorkspace, sizeof(real) * numclasses * rows, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST );
          LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, temp_hxw, temp , sizeof(T)*(rows* numclasses)  );
        }else{
     
          //		copy_host_device( h_weights, weights, sizeof(real) * numclasses * cols, cudaMemcpyDeviceToHost, ERROR_MEMCPY_DEVICE_HOST );
          LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, weight, h_weights , sizeof(T)*(cols* numclasses)  );
          for (int i = 0; i < rows; i ++) {
            sumexp = 0; 	
            for (int c = 0; c < numclasses; c ++) {
              dot = 0; 
              for (int j = 0; j < cols; j ++) dot += h_test_set[ j * rows + i ] * h_weights[ c * numclasses + j ];
              sumexp += exp ( dot ); 
            }
            sumexp += 1.;
      
            for (int c = 0; c < numclasses; c ++) {
              dot = 0; 
              for (int j = 0; j < cols; j ++) dot += h_test_set[ j * rows + i ] * h_weights[ c * numclasses + j ];
              temp[ i * numclasses + c ] = exp( dot ) / sumexp; 
            }
          }
        }

        	// classify here, 
	        // Which ever probability is maximum
        
	for (int i = 0; i < rows; i ++){
		
		pmax = 0; 
		pclass = -1;
		sumprob = 0; 
		for (int c = 0; c < numclasses; c ++){

			sumprob += temp[ c * rows + i ];
			if (pmax < temp[ c * rows + i ]){
				pmax = temp[c * rows + i]; 
				pclass = c + 1; 
			}
		}
		

		if ((pmax <= (1. - sumprob)) && (h_test_labels[i] == (numclasses + 1))){ 
			matches ++; 
		} else if ((pmax > (1. - sumprob)) && (pclass == (int)(h_test_labels[i])) ) {
			matches ++; 
		} else 
			nomatches ++; 
 		 

  }
  
    *out=(matches/(matches + nomatches)) * 100.;

    free(h_test_set);
    free(h_weights);
    free(temp);
    free(h_test_labels);
      }
    };
  
template <typename T>
struct CGLineSearch<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx,
      OpKernel* kernel,
      Tensor d,
      const Tensor& weights,
      T rho,
      T beta,
      T maxit,
      const Tensor& features,
      const Tensor& target,
      Tensor indicatorVal,
      Tensor maxdots,
      Tensor alphax,
      Tensor XW,
      T lbd,
      int rows,
      int cols,
      int numclasses,
      Tensor gk,
      Tensor xx,
      int BLOCKS,
      int BLOCK_SIZE,
      int BLOCKS_POW_2,
      T* out) {
   
        T alphak=1.0;
        T fk;
        T fk1;
        T temp=0;
        Tensor temp_gpu;
        ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({ 1,1  }), &temp_gpu);
    
        Tensor x;
        ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({ cols,numclasses  }), &x);
        int iterations = 0; 
        //copy weight to x
        LaunchBlasCopy<GPUDevice,T,true>::launch(ctx,kernel,weights,x,numclasses,cols);      

       if (numclasses ==1){

        int a= 0;
        // implement this later 
       }
       else{
            SoftmaxMulticlassFx<GPUDevice,T,true>::launch(ctx, kernel,features,target,indicatorVal,maxdots, alphax,XW,rows,cols,numclasses,x,lbd,BLOCKS,BLOCK_SIZE,BLOCKS_POW_2,&fk);
            std::cout.precision(10);
            
       }
        //xx = x;
        LaunchBlasCopy<GPUDevice,T,true>::launch(ctx,kernel,x,xx,numclasses,cols);
        //x = x + alphak*d
        LaunchBlasAxpy<GPUDevice,T,true>::launch(ctx,kernel,alphak,d,x,numclasses,cols);
        
        LaunchBlasDnrm2<GPUDevice,T,true>::launch(ctx,kernel,d,temp_gpu,numclasses,cols);
        LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, temp_gpu, &temp , sizeof(T)  );
        if (numclasses ==1){
            
                    int a= 0;
                    // implement this later 
        }
        else{
                    SoftmaxMulticlassFx<GPUDevice,T,true>::launch(ctx, kernel,features,target,indicatorVal, maxdots, alphax,XW,rows,cols,numclasses,x,lbd,BLOCKS,BLOCK_SIZE,BLOCKS_POW_2,&fk1);
                   
        }
        //p^T*gradient
        LaunchBlasDot<GPUDevice,T,true>::launch(ctx,kernel,gk,d,temp_gpu,numclasses,cols);
        LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, temp_gpu, &temp , sizeof(T)  );

        int i=0;
        while (fk1 > fk + beta * alphak * temp){

            iterations ++; 
            if (iterations >= maxit) break;
            alphak = alphak*rho;
            //copy xx to x
            LaunchBlasCopy<GPUDevice,T,true>::launch(ctx,kernel,xx,x,numclasses,cols);
            // x= x+ alpha*p
            LaunchBlasAxpy<GPUDevice,T,true>::launch(ctx,kernel,alphak,d,x,numclasses,cols);
            if (numclasses ==1){
                    int a= 0;
                    // implement this later 
            }
            else{
                    SoftmaxMulticlassFx<GPUDevice,T,true>::launch(ctx, kernel,features,target,indicatorVal,maxdots, alphax,XW,rows,cols,numclasses,x,lbd,BLOCKS,BLOCK_SIZE,BLOCKS_POW_2,&fk1);
            }
        }
        *out =alphak;
    }
  };




template <typename T>
struct CublasCGMulticlassOptimized<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx,
      OpKernel* kernel,
      const Tensor& features,
      Tensor g,
      const Tensor& weight,
      Tensor x,
      Tensor x_best,
      T lbd,
      int rows,
      int cols, 
      int num_classes, 
      Tensor HXW,
      int MAX_ITERATIONS ,
      T tolerance,
      T *rel_residual,
      T *best_rel_residual,
      int BLOCKS, 
      int BLOCK_SIZE,
      Tensor Hg,
      Tensor residual,
      Tensor p,
      Tensor A,
      Tensor B,
      Tensor C,
      T rsold,
      T rsnew,
      T alpha,
      T tmp,
      T cg_alpha,
      T cg_beta,
      int * out,
      Tensor X_subsample_hess,
      T hessianSampleSize,
      T sampling_type,
      Tensor A_hess_sub,
      Tensor B_hess_sub,
      Tensor C_hess_sub,
      Tensor hx_subsample_indices


          ) {
   
    int i =0;
    
    T gradient_norm_host;
    Tensor gradient_norm;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({ 1, 1  }), &gradient_norm);

    T xnorm_host;
    Tensor xnorm;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({ 1, 1  }), &xnorm);


    Tensor rsold_gpu;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({ 1,1  }), &rsold_gpu);



    if(sampling_type>=1){

      std::map<int,int> mm;
   
      std::default_random_engine generator( time(NULL) );
      
      std::uniform_int_distribution<int> unif(0, rows-1);
     
      int  *indices =  (int*) malloc (hessianSampleSize * sizeof(int));
      for (int i = 0; i<hessianSampleSize;i++){
       
        int x = unif(generator);
        if(mm.count(x)==0){
            mm[x]=1;
            indices[i]=x;
        }
        else{
          i--;
        }
      }

      

      LaunchMemcpyH2D<GPUDevice,int,true>::launch(ctx,kernel, hx_subsample_indices, indices , sizeof(int)*(hessianSampleSize)  );
      
      free(indices);
      
      functor::FtorSampler<GPUDevice,double>()(
          ctx->eigen_device<GPUDevice>(),
          features.template flat<T>().data(),
          rows ,
          cols,
          hx_subsample_indices.template flat<int>().data(),
          X_subsample_hess.template flat<T>().data(),
          hessianSampleSize,
          BLOCK_SIZE   );


      
      SoftmaxMulticlassHxSubsampled<GPUDevice,T,true>::launch(ctx, kernel,
            features,
            rows,
            cols,
            num_classes,
            weight,
            x,
            lbd,
            A_hess_sub,//devPtr,
            C_hess_sub,
            Hg,
            B_hess_sub,
            X_subsample_hess,
            hessianSampleSize,
            sampling_type,
            BLOCKS,
            BLOCK_SIZE);

    }else{


    SoftmaxMulticlassHxOptimized<GPUDevice,T,true>::launch(ctx, kernel,
    features,
    rows,
    cols,
    num_classes,
    weight,
    x,
    lbd,
    A,//devPtr,
    C,
    Hg,
    HXW,
    BLOCKS,
    BLOCK_SIZE);
    }



// residual = g - H*g;

LaunchBlasCopy<GPUDevice,T,true>::launch(ctx,kernel,g,residual,num_classes,cols);


cg_alpha =-1.;

LaunchBlasAxpy<GPUDevice,T,true>::launch(ctx,kernel,cg_alpha,Hg,residual,num_classes,cols);




LaunchBlasCopy<GPUDevice,T,true>::launch(ctx,kernel,residual,p,num_classes,cols);




LaunchBlasDot<GPUDevice,T,true>::launch(ctx,kernel,residual,residual,rsold_gpu,num_classes,cols);

// copy parameters from gpu to host
LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, rsold_gpu, &rsold , sizeof(T)  );

LaunchBlasDnrm2<GPUDevice,T,true>::launch(ctx,kernel,g,gradient_norm,num_classes,cols);


LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, gradient_norm, &gradient_norm_host , sizeof(T)  );

*best_rel_residual = sqrt( rsold ) / gradient_norm_host;

// dst , src
LaunchMemcpyD2D<GPUDevice,T,true>::launch(ctx,kernel,x_best,x, num_classes * cols * sizeof(T));

Tensor tmp_gpu;
ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({ 1,1  }), &tmp_gpu);
Tensor rsnew_gpu;
ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({ 1,1  }), &rsnew_gpu);

for(  i = 0; i < MAX_ITERATIONS ; ++i ){ 
    

  if(sampling_type>=1){
       
              SoftmaxMulticlassHxSubsampled<GPUDevice,T,true>::launch(ctx, kernel,
                features,
                rows,
                cols,
                num_classes,
                weight,
                p,
                lbd,
                A_hess_sub,//devPtr,
                C_hess_sub,
                Hg,
                B_hess_sub,
                X_subsample_hess,
                hessianSampleSize,
                sampling_type,
                BLOCKS,
                BLOCK_SIZE);
        }
        
        
        else{

    SoftmaxMulticlassHxOptimized<GPUDevice,T,true>::launch(ctx, kernel,
        features,\
        rows,cols,\
        num_classes,\
        weight,\
        p,\
        lbd,\
        A,\ 
        C,\
        Hg,\
        HXW,\
        BLOCKS,\
        BLOCK_SIZE);
    }
        // tmp_gpu= pk^T * H(pk)
        LaunchBlasDot<GPUDevice,T,true>::launch(ctx,kernel,Hg,p,tmp_gpu,num_classes,cols);
        
        LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, tmp_gpu, &tmp , sizeof(T)  );
        // ak = rk^T*rk/tmp
        alpha= -1.0*(rsold/tmp);
        // rk+1 =rk -ak*H(pk)
        LaunchBlasAxpy<GPUDevice,T,true>::launch(ctx,kernel,alpha,Hg,residual,num_classes,cols);
        alpha= alpha*(-1.0);
        // xk+1 =xk + akpk
        LaunchBlasAxpy<GPUDevice,T,true>::launch(ctx,kernel,alpha,p,x,num_classes,cols);
        
        // xnorm ?
        LaunchBlasDnrm2<GPUDevice,T,true>::launch(ctx,kernel,x,xnorm,num_classes,cols);
        LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, xnorm, &xnorm_host , sizeof(T)  );
        
        LaunchBlasDot<GPUDevice,T,true>::launch(ctx,kernel,residual,residual,rsnew_gpu,num_classes,cols);
        LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, rsnew_gpu, &rsnew , sizeof(T)  );
        *rel_residual= sqrt(rsnew)/gradient_norm_host;

        // rel_residual = sqrt(rk+1^T*rk+1)/gradient_norm_host 
        if(*rel_residual<*best_rel_residual){

            
            *best_rel_residual = *rel_residual;
            LaunchMemcpyD2D<GPUDevice,T,true>::launch(ctx,kernel,x_best,x, num_classes * cols * sizeof(T));
            
        }


        if (*rel_residual <= tolerance) break; 
        
        // alpha= rk+1^T*rk+1/rk^T*rk;
        alpha = rsnew/rsold;

        // pk= alpha*pk
        LaunchBlasScal<GPUDevice,T,true>::launch(ctx, kernel,  alpha, p,num_classes,cols);
        

        alpha=1.0;
        //pk+1=rk+alpha*pk
        LaunchBlasAxpy<GPUDevice,T,true>::launch(ctx,kernel,alpha,residual,p,num_classes,cols);

        rsold=rsnew;

}



    *out =i;
    }
  };


  template <typename T>
  struct SoftmaxMulticlassFx<GPUDevice, T, true /* USE_CUBLAS */> {
    static void launch(
        OpKernelContext* ctx,
         OpKernel* kernel,
          const Tensor& features,
           const Tensor& target,
           Tensor indicatorVal,
           Tensor maxdots,
           Tensor alphax,
           Tensor XW,
           int rows, 
           int cols,
            int num_classes,
             const Tensor& weight,
              T lbd,
               int BLOCKS,
                int BLOCK_SIZE,
                int BLOCKS_POW_2,
                 T* out  ) {
     
      auto* stream = ctx->op_device_context()->stream();
      OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

      int NUM_THREADS = 1;
 
      
      auto features_ptr = AsDeviceMemory(features.template flat<T>().data());
      auto weight_ptr = AsDeviceMemory(weight.template flat<T>().data());
      auto XW_ptr = AsDeviceMemory(XW.template flat<T>().data());

      bool blas_launch_status =
          stream->ThenBlasGemm(perftools::gputools::blas::Transpose::kNoTranspose, 
                               perftools::gputools::blas::Transpose::kNoTranspose,
                               rows,
                               num_classes,
                               cols,
                               1.0f,
                               features_ptr,
                               rows,
                               weight_ptr,
                               cols,
                               0.0f,
                               &XW_ptr,
                               rows).ok();
  

      if (!blas_launch_status) {
        ctx->SetStatus(errors::Internal(
            "Blas SGEMM launch failed "));
      }


      functor::FtorComputeFx<GPUDevice,double>()(
        ctx->eigen_device<GPUDevice>(),
        XW.flat<double>().data(),
        rows,
        cols,
        num_classes,
        target.flat<double>().data(),
        indicatorVal.flat<double>().data(),
        NUM_THREADS,
        maxdots.flat<double>().data(),
        BLOCKS,
        BLOCK_SIZE,
        BLOCKS_POW_2);
  
 
        Tensor maxdots_plus_rows;
        ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  rows, 1 }), &maxdots_plus_rows);
        functor::FtorReduce<GPUDevice,double>()(ctx->eigen_device<GPUDevice>(), 
          maxdots.flat<double>().data(),
          maxdots_plus_rows.flat<double>().data(),
          rows,
          BLOCKS,
          BLOCK_SIZE  );

          Tensor pg3;
          ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  1, 1 }), &pg3);
          functor::FtorReduce<GPUDevice,double>()(ctx->eigen_device<GPUDevice>(), 
          maxdots_plus_rows.flat<double>().data(),
          pg3.flat<double>().data(),
          BLOCKS,
          1,
          BLOCKS_POW_2 );


          Tensor pg0;
          ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  1, 1 }), &pg0);


    
          functor::FtorReduce<GPUDevice,double>()(ctx->eigen_device<GPUDevice>(), 
          indicatorVal.flat<double>().data(),
          pg0.flat<double>().data(),
          BLOCKS,
          1,
          BLOCKS_POW_2 );


          functor::FtorReduceVectorWarp<GPUDevice,double>()(ctx->eigen_device<GPUDevice>(), 
          XW.flat<double>().data(),
          maxdots.flat<double>().data(),
          alphax.flat<double>().data(),
          rows,
          num_classes,
          BLOCKS,
          BLOCK_SIZE);

        Tensor alphax_plus_rows;
        ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  rows, 1 }), &alphax_plus_rows);
        functor::FtorReduceLog<GPUDevice,double>()(ctx->eigen_device<GPUDevice>(), 
        alphax.flat<double>().data(),
        alphax_plus_rows.flat<double>().data(),
        rows,
        BLOCKS,
        BLOCK_SIZE);


        Tensor pg1;
        ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  1, 1 }), &pg1);
        functor::FtorReduce<GPUDevice,double>()(ctx->eigen_device<GPUDevice>(), 
        alphax_plus_rows.flat<double>().data(),
        pg1.flat<double>().data(),
        BLOCKS,
        1,
        BLOCKS_POW_2 );
        
        Tensor pg2;
        ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({ 1,1  }), &pg2);
        LaunchBlasDnrm2<GPUDevice,T,true>::launch(ctx,kernel,weight,pg2,num_classes,cols);

        T pg0_host,pg1_host,pg2_host,pg3_host;
        LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, pg0, &pg0_host , sizeof(T)  );
        LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, pg1, &pg1_host , sizeof(T)  );
        LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, pg2, &pg2_host , sizeof(T)  );
        LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, pg3, &pg3_host , sizeof(T)  );

        *out = (pg3_host + pg1_host) - pg0_host + (lbd/2.0) * pow(pg2_host, 2.);
  

    }
  };
  

template <typename T>
struct SoftmaxMulticlassHxOptimized<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel, const Tensor& features,int rows,int cols, int num_classes, const Tensor& weight, Tensor vector, T lbd,Tensor A,Tensor C, Tensor Hv ,Tensor B,int BLOCKS, int BLOCK_SIZE  ) {
   

        auto* stream = ctx->op_device_context()->stream();
        OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

        auto features_ptr = AsDeviceMemory(features.template flat<T>().data());
        auto vector_ptr = AsDeviceMemory(vector.flat<T>().data());
        auto A_ptr = AsDeviceMemory(A.flat<T>().data());
    
        bool blas_launch_status =
        stream->ThenBlasGemm(perftools::gputools::blas::Transpose::kNoTranspose, 
                            perftools::gputools::blas::Transpose::kNoTranspose,
                             rows, num_classes, cols, 1.0f,
                             features_ptr, rows, vector_ptr,
                             cols , 0.0f, &A_ptr, rows)
            .ok();
        if (!blas_launch_status) {
            ctx->SetStatus(errors::Internal(
             "Blas SGEMM launch failed "));
        }    

 

    functor::FtorHxC<GPUDevice,double>()(ctx->eigen_device<GPUDevice>(),  A.flat<T>().data(),B.flat<T>().data(),C.flat<T>().data(), rows, cols , num_classes,BLOCKS,BLOCK_SIZE   );
    


    
    auto C_ptr = AsDeviceMemory(C.flat<T>().data());
    auto Hv_ptr = AsDeviceMemory(Hv.flat<T>().data());
    
    blas_launch_status =
        stream->ThenBlasGemm(perftools::gputools::blas::Transpose::kTranspose,
                            perftools::gputools::blas::Transpose::kNoTranspose,
                            cols, num_classes, rows, 1.0f,
                            features_ptr, rows, C_ptr,
                            rows , 0.0f, &Hv_ptr, cols)
            .ok();
    if (!blas_launch_status) {
      ctx->SetStatus(errors::Internal(
          "Blas SGEMM launch failed "));
    } 

    if(lbd!=0){ 
    
        int rblocks = ((num_classes * cols) / BLOCK_SIZE) +
                 (((num_classes * cols) % BLOCK_SIZE == 0) ? 0 : 1 );
   
        functor::FtorAddRegularizer<GPUDevice,double>()(ctx->eigen_device<GPUDevice>(),  Hv.flat<T>().data(),vector.flat<T>().data(),
        lbd, num_classes*cols, 1.0 , rblocks, BLOCK_SIZE   ); 

      }

  }
};





template <typename T>
struct SoftmaxMulticlassHxSubsampled<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx,
      OpKernel* kernel,
      const Tensor& features,
      int rows,
      int cols, 
      int num_classes, 
      const Tensor& weight,
       Tensor vector, 
       T lbd,
       Tensor A_hess_sub,
       Tensor C_hess_sub, 
       Tensor Hv ,
       Tensor HXW_hess_sub,//B
       Tensor sampledDataset,
       int sampleSize,
       int samplingType,
       int BLOCKS,
      int BLOCK_SIZE  ) {
   
        
        auto* stream = ctx->op_device_context()->stream();
        OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

        
        // A=XV
        auto sampled_features_ptr = AsDeviceMemory(sampledDataset.template flat<T>().data());
        auto vector_ptr = AsDeviceMemory(vector.flat<T>().data());
        auto A_hess_sub_ptr = AsDeviceMemory(A_hess_sub.flat<T>().data());
    
        bool blas_launch_status =
        stream->ThenBlasGemm(perftools::gputools::blas::Transpose::kNoTranspose, 
                            perftools::gputools::blas::Transpose::kNoTranspose,
                            sampleSize, num_classes, cols, 1.0f,
                            sampled_features_ptr, sampleSize, vector_ptr,
                             cols , 0.0f, &A_hess_sub_ptr, sampleSize)
            .ok();
        if (!blas_launch_status) {
            ctx->SetStatus(errors::Internal(
             "Blas SGEMM launch failed "));
        }
        
      //compute B here. for sub sample part of the feautre matrix here.   
      ComputeHXW<GPUDevice, T, true>::launch(ctx, kernel, sampledDataset, weight,HXW_hess_sub, sampleSize, cols, num_classes ,BLOCKS, BLOCK_SIZE,samplingType);
     //Compute C Here. 


     int blocks = sampleSize / BLOCK_SIZE + (((sampleSize % BLOCK_SIZE) == 0) ? 0 : 1);

     if (samplingType == 2) {
        int a =0;
        // do it later
      // ker_hx_C_scale <<< blocks, BLOCK_SIZE >>>
      //  (A, B, C, sampleSize, cols, num_classes, scaleTerms); 
    } else {


      functor::FtorHxC<GPUDevice,double>()(ctx->eigen_device<GPUDevice>(),
        A_hess_sub.flat<T>().data(),
        HXW_hess_sub.flat<T>().data(),
        C_hess_sub.flat<T>().data(), 
        sampleSize,
        cols ,
        num_classes,
        blocks,
        BLOCK_SIZE   );
      //ker_hx_C <<< blocks, BLOCK_SIZE >>>
       // (A, B, C, sampleSize, cols, num_classes); 
    }

    //Compute the final Matvec Here.     
    auto C_hess_sub_ptr = AsDeviceMemory(C_hess_sub.flat<T>().data());
    auto Hv_ptr = AsDeviceMemory(Hv.flat<T>().data());
    
    blas_launch_status =
        stream->ThenBlasGemm(perftools::gputools::blas::Transpose::kTranspose,
                            perftools::gputools::blas::Transpose::kNoTranspose,
                            cols, num_classes, sampleSize, 1.0f,
                            sampled_features_ptr, sampleSize, C_hess_sub_ptr,
                            sampleSize , 0.0f, &Hv_ptr, cols)
            .ok();
    if (!blas_launch_status) {
      ctx->SetStatus(errors::Internal(
          "Blas SGEMM launch failed "));
    } 

    if(lbd!=0){ 
    
        int rblocks = ((num_classes * cols) / BLOCK_SIZE) +
                 (((num_classes * cols) % BLOCK_SIZE == 0) ? 0 : 1 );
   
        functor::FtorAddRegularizer<GPUDevice,double>()(ctx->eigen_device<GPUDevice>(),  Hv.flat<T>().data(),vector.flat<T>().data(),
        lbd, num_classes*cols, 1.0 , rblocks, BLOCK_SIZE   ); 
 

      }

  }
};



template <typename T>
struct SoftmaxMulticlassGxSubsampled<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx, 
      OpKernel* kernel, 
      const Tensor& features,
      const Tensor& target,
      int rows, int cols, int num_classes, 
      const Tensor& weight,
      T lbd,
      Tensor HXW,
      Tensor gradient, 
      Tensor gradientDataset,
      Tensor gradientLabels ,
      Tensor HXWIND,
      int sampleSize,
      int samplingType,
      int BLOCKS, int BLOCK_SIZE  ) {
   

      int blocks; 

      LaunchMemZero<GPUDevice,T,true>::launch(ctx, kernel,  gradient, num_classes*cols*sizeof(T));
      //computeHXW Here. 
      ComputeHXW<GPUDevice, T, true>::launch(ctx, kernel, gradientDataset, weight,HXW, sampleSize, cols, num_classes ,BLOCKS, BLOCK_SIZE,samplingType);
      blocks = sampleSize / BLOCK_SIZE + ((( sampleSize % BLOCK_SIZE ) == 0) ? 0 : 1 ); 

      functor::FtorDxSoftmaxInd<GPUDevice,double>()(
        ctx->eigen_device<GPUDevice>(),
        HXW.flat<double>().data(),
        gradientLabels.flat<double>().data(),
        sampleSize,
        num_classes,
        HXWIND.flat<double>().data(),
        1,
        blocks,
        BLOCK_SIZE
        );


    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    auto gradientDataset_ptr = AsDeviceMemory(gradientDataset.template flat<T>().data());
    auto hxwInd_ptr = AsDeviceMemory(HXWIND.flat<T>().data());
    auto gradient_ptr = AsDeviceMemory(gradient.flat<T>().data());

    bool blas_launch_status =
        stream->ThenBlasGemm(perftools::gputools::blas::Transpose::kTranspose, 
          perftools::gputools::blas::Transpose::kNoTranspose, 
          cols, num_classes, sampleSize, 1.0f,
          gradientDataset_ptr, sampleSize, hxwInd_ptr,
          sampleSize , 0.0f, &gradient_ptr, cols)
            .ok();
    if (!blas_launch_status) {
      ctx->SetStatus(errors::Internal(
          "Blas SGEMM launch failed "));
    }

    if(samplingType>=2){

      int a =0;
      // do it later
    }
 
    LaunchBlasAxpy<GPUDevice,T,true>::launch(ctx,kernel,lbd,weight,gradient,num_classes,cols);
 

  }
};





template <typename T>
struct SoftmaxMulticlassGxOptimized<GPUDevice, T, true /* USE_CUBLAS */> {
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel, const Tensor& features,const Tensor& target,int rows,int cols, int num_classes, const Tensor& weight, T lbd, Tensor HXW ,Tensor gradient, Tensor HXWIND,int BLOCKS, int BLOCK_SIZE  ) {
   

    T alpha;
    T beta;

    T dxnrm;
    T gxnrm;



    
    LaunchMemZero<GPUDevice,T,true>::launch(ctx, kernel,  gradient, num_classes*cols*sizeof(T));




    functor::FtorDxSoftmaxInd<GPUDevice,double>()(
        ctx->eigen_device<GPUDevice>(),
        HXW.flat<double>().data(),
        target.flat<double>().data(),
        rows,
        num_classes,
        HXWIND.flat<double>().data(),
        1,
        BLOCKS,
        BLOCK_SIZE
        );

       /* T *hxwoutput =  (T*) malloc (rows * num_classes*sizeof(T));
        //T goutput[cols* numclasses] ;
          //copy parameters from gpu to host
        LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, HXW, hxwoutput , sizeof(T)*(rows* num_classes)  );
        std::cout<<"hxwoutput"<<std::endl;
        std::cout<<hxwoutput[0]<<std::endl;
        std::cout<<hxwoutput[1]<<std::endl;
        std::cout<<hxwoutput[2]<<std::endl;
        std::cout<<hxwoutput[3]<<std::endl;
         free(hxwoutput);

*/

        /* T *xwoutput =  (T*) malloc (rows * cols*sizeof(T));
         //T goutput[cols* numclasses] ;
           //copy parameters from gpu to host
         LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, features, xwoutput , sizeof(T)*(rows* cols)  );
         std::cout<<"xwoutput"<<std::endl;
         
         
         for (int i =0 ;i<rows*cols;i++){

          std::cout<<xwoutput[i]<<std::endl;
         }

          free(xwoutput);


         T *hxwindoutput =  (T*) malloc (rows * num_classes*sizeof(T));
         //T goutput[cols* numclasses] ;
           //copy parameters from gpu to host
           std::cout<<"hxwindoutput"<<std::endl;
         LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, HXWIND, hxwindoutput , sizeof(T)*(rows* num_classes)  );
         for (int i =0 ;i<rows*num_classes;i++){
          
                    std::cout<<hxwindoutput[i]<<std::endl;
          }
          free(hxwindoutput);*/
   
    alpha=1.0;
    beta=0.0;

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    auto features_ptr = AsDeviceMemory(features.template flat<T>().data());
    auto hxwInd_ptr = AsDeviceMemory(HXWIND.flat<T>().data());
    auto gradient_ptr = AsDeviceMemory(gradient.flat<T>().data());

    bool blas_launch_status =
        stream->ThenBlasGemm(perftools::gputools::blas::Transpose::kTranspose, 
          perftools::gputools::blas::Transpose::kNoTranspose, 
          cols, num_classes, rows, 1.0f,
                             features_ptr, rows, hxwInd_ptr,
                             rows , 0.0f, &gradient_ptr, cols)
            .ok();
    if (!blas_launch_status) {
      ctx->SetStatus(errors::Internal(
          "Blas SGEMM launch failed "));
    }


    ////regularizer here. 
   // cublasCheckError( cublasDaxpy( cublasHandle, num_classes * cols, &lambda, weights, 1, gradient, 1) );
    LaunchBlasAxpy<GPUDevice,T,true>::launch(ctx,kernel,lbd,weight,gradient,num_classes,cols);
 
 

  

  }
};

//#endif  // GOOGLE_CUDA

template <typename Device, typename T, bool USE_CUBLAS>
//NewtonTypeOp
class MatMulFangOp : public OpKernel {
 public:
  explicit MatMulFangOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
  }
  
  perftools::gputools::DeviceMemory<T> Memory_Allocate( const Tensor& a){

    return AsDeviceMemory(a.template flat<T>().data());
  }

  //NewtonCG
  void Newton_CG_Multi_Optimized(OpKernelContext* ctx,
    OpKernel* kernel,
    const Tensor& X,\
    const Tensor& W,\
    T lbd ,\
    T tolerance,\
    Tensor* out,\
    const Tensor& target,\
    T maxiteration,
    const Tensor& X2,
    const Tensor& Y2,
    T max_cg_iterations,
    T max_cg_tolerance,
    T linesearch_beta,
    T linesearch_rho,
    T linesearch_maxit,
    T sampling_type
    ){


    /* Compute BLOCKS , BLOCK_SIZE, and BLOCKS_POW_2*/

    int BLOCKS;
    int BLOCK_SIZE;
    int BLOCKS_POW_2;
    functor::FtorComputeBlocks<GPUDevice,double>()(ctx->eigen_device<GPUDevice>(), &BLOCKS, &BLOCK_SIZE, X.dim_size(0)  );
    functor::FtorComputePOW2<GPUDevice,double>()(ctx->eigen_device<GPUDevice>(), BLOCKS, &BLOCKS_POW_2  );

	int iterations, cg_iterations;
    

	  T alpha, alphak;

    
    T rel_residual;
    T best_rel_residual;
    T snorm;
    T gxnorm;
 
    T train_accuracy, test_accuracy;
    double iteration_start, iteration_total;
    T train_function, test_function;
    Tensor temp;

    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({ 1, 1  }), &temp);

    int classes_to_solve = W.dim_size(1);
  
    int num_features =W.dim_size(0);
    int rows = X.dim_size(0);


    int gradientSampleSize = (GRADIENT_SAMPLING_SIZE * rows) / 100; 
	  int hessianSampleSize = (HESSIAN_SAMPLING_SIZE * rows)/100; 

    std::cout<<" gradientSampleSize "<< gradientSampleSize<<std::endl;
    std::cout<<" hessianSampleSize "<< hessianSampleSize<<std::endl;

    //Tensor HXW_subsample_grad;
    //Tensor X_subsample_grad;
    Tensor X_subsample_hess;
    Tensor gradientDataset;
    Tensor gradientLabels;
    //Tensor HXWIND_subsample_grad;
    Tensor gx_subsample_indices;
    Tensor hx_subsample_indices;
    Tensor B_hess_sub; 
    Tensor A_hess_sub;
	  Tensor C_hess_sub;

    //if(sampling_type>=1){
     // ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  gradientSampleSize, classes_to_solve }), &HXW_subsample_grad);
      ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  gradientSampleSize, num_features }), &gradientDataset);
      ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  gradientSampleSize, 1 }), &gradientLabels);
     // ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  gradientSampleSize, classes_to_solve }), &HXWIND_subsample_grad);
      ctx->allocate_temp( DataTypeToEnum<int>::value,TensorShape({  gradientSampleSize, 1 }), &gx_subsample_indices);
      //ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  gradientSampleSize, num_features }), &X_subsample_grad);
      

      
      ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  hessianSampleSize, classes_to_solve }), &A_hess_sub);
      ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  hessianSampleSize, classes_to_solve }), &B_hess_sub);
      ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({ hessianSampleSize, classes_to_solve  }), &C_hess_sub);
      ctx->allocate_temp( DataTypeToEnum<int>::value,TensorShape({  hessianSampleSize, 1 }), &hx_subsample_indices);
      ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  hessianSampleSize, num_features }), &X_subsample_hess);
    //}

    Tensor xx;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({ num_features, classes_to_solve  }), &xx);
    Tensor s;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  num_features, classes_to_solve }), &s);
    Tensor s_best;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  num_features, classes_to_solve }), &s_best);
    Tensor gradient; 
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  num_features, classes_to_solve }), &gradient);
    Tensor Hv;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  num_features, classes_to_solve }), &Hv);
    Tensor HXW;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  rows, classes_to_solve }), &HXW);
    Tensor HXWIND;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  rows, classes_to_solve }), &HXWIND);
 

    T s_norm, s_best_norm;



    if(sampling_type>=1){

      std::map<int,int> mm;
      
      std::default_random_engine generator( time(NULL) );
    
      std::uniform_int_distribution<int> unif(0, rows-1);
      
      int  *indices =  (int*) malloc (gradientSampleSize * sizeof(int));
      for (int i = 0; i<gradientSampleSize;i++){
        
        int x = unif(generator);
        if(mm.count(x)==0){
            mm[x]=1;
            indices[i]=x;
        }
        else{
          i--;
        }
      }

      LaunchMemcpyH2D<GPUDevice,int,true>::launch(ctx,kernel, gx_subsample_indices, indices , sizeof(int)*(gradientSampleSize)  );
      
      free(indices);

          functor::FtorSampler<GPUDevice,double>()(
            ctx->eigen_device<GPUDevice>(),
            X.template flat<T>().data(),
            rows ,
            num_features,
            gx_subsample_indices.template flat<int>().data(),
            gradientDataset.template flat<T>().data(),
            gradientSampleSize,
            BLOCK_SIZE   );


            functor::FtorSampler<GPUDevice,double>()(
              ctx->eigen_device<GPUDevice>(),
              target.template flat<T>().data(),
              0 ,
              1,
              gx_subsample_indices.template flat<int>().data(),
              gradientLabels.template flat<T>().data(),
              gradientSampleSize,
              BLOCK_SIZE   );

      SoftmaxMulticlassGxSubsampled<Device,T,USE_CUBLAS>::launch(ctx,
         this,
          X,
          target,
          rows,
          num_features,
          classes_to_solve,
          W,
          lbd, 
          HXW,
          gradient,
          gradientDataset,
          gradientLabels,
          HXWIND,
          gradientSampleSize,
          sampling_type,
          BLOCKS,
          BLOCK_SIZE);

    }else{

    //1.  get the hessian and gradient. 
    ComputeHXW<Device, T, USE_CUBLAS>::launch(ctx, this, X, W,HXW, rows, num_features, classes_to_solve ,BLOCKS, BLOCK_SIZE,sampling_type);
    SoftmaxMulticlassGxOptimized<Device,T,USE_CUBLAS>::launch(ctx, this, X, target, rows, num_features, classes_to_solve, W, lbd, HXW, gradient,HXWIND,BLOCKS,BLOCK_SIZE);

    }
      
  
    Tensor gxnorm_gpu;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({ 1, 1  }), &gxnorm_gpu);
    Tensor Hg;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  num_features, classes_to_solve }), &Hg);
    Tensor residual;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  num_features, classes_to_solve }), &residual);
    Tensor B; 
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  rows, classes_to_solve }), &B);
    Tensor p;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  num_features, classes_to_solve }), &p);
    Tensor A;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  rows, classes_to_solve }), &A);
	  Tensor C;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({ rows, classes_to_solve  }), &C);


    Tensor indicatorVal;
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  rows, 1 }), &indicatorVal);
    Tensor maxdots; //M(x)
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  rows, 1 }), &maxdots);
    Tensor alphax;  //alpha(x)
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  rows, 1 }), &alphax);

    Tensor XW; 
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  rows, classes_to_solve }), &XW);



    T rsold;
    T rsnew;
    T tmp;
    T cg_alpha;
    T cg_beta;


    //2. Initialization Here.
    iterations = 0;
    snorm = 100;
    gxnorm = 100;
    rel_residual = 100;

    LaunchBlasDnrm2<GPUDevice,T,true>::launch(ctx,kernel,gradient,gxnorm_gpu,classes_to_solve,num_features);
   
    //__STATISTICS__
    iteration_total = 0; 

    Tensor temp_hxw;
    
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  rows, classes_to_solve }), &temp_hxw);
    
    Tensor temp2_hxw;
    
    ctx->allocate_temp( DataTypeToEnum<T>::value,TensorShape({  Y2.dim_size(0), classes_to_solve }), &temp2_hxw);

    SoftmaxMulticlassFx<GPUDevice,T,true>::launch(
       ctx, kernel,X,target,indicatorVal,
       maxdots, alphax , XW , rows , num_features, classes_to_solve, W, lbd,
       BLOCKS,BLOCK_SIZE,BLOCKS_POW_2,&train_function);
    SoftmaxMulticlassFx<GPUDevice,T,true>::launch(
        ctx, kernel,X2,Y2,indicatorVal,
        maxdots, alphax , XW , Y2.dim_size(0) , num_features, classes_to_solve, W, lbd,
        BLOCKS,BLOCK_SIZE,BLOCKS_POW_2,&test_function);

    SoftmaxPredict<GPUDevice,T,true>::launch(
      ctx, kernel,X,target,W,
      rows,num_features,classes_to_solve, temp_hxw,1, BLOCKS,BLOCK_SIZE,&train_accuracy,sampling_type
            );

    SoftmaxPredict<GPUDevice,T,true>::launch(
      ctx, kernel,X2,Y2,W,
      Y2.dim_size(0),num_features,classes_to_solve, temp2_hxw,1, BLOCKS,BLOCK_SIZE,&test_accuracy,sampling_type
                  );
    
    fprintf( stderr, "iteration \t norm(gradient) \t Rel_Residual \t CG-ITERATIONS \t Train_Accu \t Obj_Val_Train \t Test_Accu \t Obj_Val_Test \n");
    fprintf( stderr, "%9d \t %e \t %e \t %d \t %3.2f \t %e \t  %3.2f \t %e \t %d\n", 
    iterations, gxnorm, rel_residual, 0, train_accuracy, train_function, test_accuracy, test_function, (unsigned int)(iteration_total * 1000) );
                                 
    // end __STATISTICS__
    
   
    while(iterations<maxiteration){

      iteration_start = Get_Time( );
      alpha = -1.;

     
    
    
    //negative gradient
    LaunchBlasScal<Device,T,USE_CUBLAS>::launch(ctx, this,  alpha, gradient,classes_to_solve,num_features);
    
    
 
    LaunchMemZero<Device,T,USE_CUBLAS>::launch(ctx, this,  s, classes_to_solve*num_features*sizeof(T));
    LaunchMemZero<Device,T,USE_CUBLAS>::launch(ctx, this,  s_best, classes_to_solve*num_features*sizeof(T));

  
     CublasCGMulticlassOptimized<Device, T, USE_CUBLAS >::launch(
        ctx,
        this,
        X,
        gradient,
        W,
        s,
        s_best,
        lbd,
        rows,
        num_features,
        classes_to_solve,
        HXW,
        max_cg_iterations,
        max_cg_tolerance ,
        &rel_residual,
        &best_rel_residual,
        BLOCKS,
        BLOCK_SIZE,
        Hg,
        residual,
        p,
        A,
        B,
        C,
        rsold,
        rsnew,
        alpha,
        tmp,
        cg_alpha,
        cg_beta,
        &cg_iterations,
        X_subsample_hess,
        hessianSampleSize,
        sampling_type,
        A_hess_sub,
        B_hess_sub,
        C_hess_sub,
        hx_subsample_indices
    );
    LaunchBlasDnrm2<GPUDevice,T,true>::launch(ctx,kernel,gradient,gxnorm_gpu,classes_to_solve,num_features);
    LaunchMemcpyD2H<GPUDevice,T,true>::launch(ctx,kernel, gxnorm_gpu, &gxnorm , sizeof(T))  ;
  

    
    CGLineSearch<Device, T, USE_CUBLAS >::launch(ctx,this,s_best,W,linesearch_rho,linesearch_beta,linesearch_maxit,X,target,indicatorVal,maxdots,alphax,XW,lbd,rows,num_features,classes_to_solve,gradient,xx,BLOCKS,BLOCK_SIZE,BLOCKS_POW_2,&alphak);

    T lbd,

    alpha = alphak;
    
    // w= w+ alphak*s_best
    LaunchBlasAxpy<GPUDevice,T,true>::launch(ctx,kernel,alpha,s_best,W,classes_to_solve,num_features);


    if(sampling_type>=1){
      
            std::map<int,int> mm;
     
            std::default_random_engine generator( time(NULL) );
           
            std::uniform_int_distribution<int> unif(0, rows-1);
            //float x = unif(generator);
            int  *indices =  (int*) malloc (gradientSampleSize * sizeof(int));
            for (int i = 0; i<gradientSampleSize;i++){
         
              int x = unif(generator);
              if(mm.count(x)==0){
                  mm[x]=1;
                  indices[i]=x;
              }
              else{
                i--;
              }
            }
      
      
            LaunchMemcpyH2D<GPUDevice,int,true>::launch(ctx,kernel, gx_subsample_indices, indices , sizeof(int)*(gradientSampleSize)  );
            
            free(indices);
            
            functor::FtorSampler<GPUDevice,double>()(
                  ctx->eigen_device<GPUDevice>(),
                  X.template flat<T>().data(),
                  rows ,
                  num_features,
                  gx_subsample_indices.template flat<int>().data(),
                  gradientDataset.template flat<T>().data(),
                  gradientSampleSize,
                  BLOCK_SIZE   );
            functor::FtorSampler<GPUDevice,double>()(
                    ctx->eigen_device<GPUDevice>(),
                    target.template flat<T>().data(),
                    rows ,
                    1,
                    gx_subsample_indices.template flat<int>().data(),
                    gradientLabels.template flat<T>().data(),
                    gradientSampleSize,
                    BLOCK_SIZE   );
      

      SoftmaxMulticlassGxSubsampled<Device,T,USE_CUBLAS>::launch(ctx,
         this,
          X,
          target,
          rows,
          num_features,
          classes_to_solve,
          W,
          lbd, 
          HXW,
          gradient,
          gradientDataset,
          gradientLabels,
          HXWIND,
          gradientSampleSize,
          sampling_type,
          BLOCKS,
          BLOCK_SIZE);
      
      }
      else{
      //update hxw
      ComputeHXW<Device, T, USE_CUBLAS>::launch(ctx, this, X, W ,HXW,rows,num_features,classes_to_solve,BLOCKS, BLOCK_SIZE,sampling_type);
     //update g
      SoftmaxMulticlassGxOptimized<Device,T,USE_CUBLAS>::launch(ctx, this, X, target, rows, num_features, classes_to_solve, W, lbd, HXW, gradient,HXWIND,BLOCKS,BLOCK_SIZE);
      }
    iteration_total = Get_Timing_Info( iteration_start );
    
   
    SoftmaxMulticlassFx<GPUDevice,T,true>::launch(
       ctx, kernel,X,target,indicatorVal,
       maxdots, alphax , XW , rows , num_features, classes_to_solve, W, lbd,
       BLOCKS,BLOCK_SIZE,BLOCKS_POW_2,&train_function);

    SoftmaxMulticlassFx<GPUDevice,T,true>::launch(
      ctx, kernel,X2,Y2,indicatorVal,
      maxdots, alphax , XW , Y2.dim_size(0) , num_features, classes_to_solve, W, lbd,
      BLOCKS,BLOCK_SIZE,BLOCKS_POW_2,&test_function);
    

    SoftmaxPredict<GPUDevice,T,true>::launch(
      ctx, kernel,X,target,W,
      rows,num_features,classes_to_solve, temp_hxw,1, BLOCKS,BLOCK_SIZE,&train_accuracy,sampling_type
            );
    SoftmaxPredict<GPUDevice,T,true>::launch(
        ctx, kernel,X2,Y2,W,
        Y2.dim_size(0),num_features,classes_to_solve, temp2_hxw,1, BLOCKS,BLOCK_SIZE,&test_accuracy,sampling_type
         );

    fprintf( stderr, "%9d \t %e \t %e \t %d \t %3.2f \t %e \t %3.2f \t %e \t %d\n", 
        iterations+1, gxnorm, rel_residual, cg_iterations, 
          train_accuracy, train_function, test_accuracy, test_function, (unsigned int)(iteration_total * 1000) );
          
    
    iterations ++; 
    if (gxnorm <= tolerance) break;

    }
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& X = ctx->input(0);
    const Tensor& target = ctx->input(1);
    const Tensor& X2 = ctx->input(2);
    const Tensor& Y2 = ctx->input(3);
    const Tensor& W = ctx->input(4);
    const Tensor& maxit = ctx->input(5);
    const Tensor& tol = ctx->input(6);
    const Tensor& maxcgiter = ctx->input(7);
    const Tensor& maxcgtol = ctx->input(8);
    const Tensor& lsbeta = ctx->input(9);
    const Tensor& lsrho = ctx->input(10);
    const Tensor& lsmaxit = ctx->input(11);
    const Tensor& lambda = ctx->input(12);
    const Tensor& sampling = ctx->input(13);


    T lbd,tolerance,maxiteration;
    T max_cg_iterations,max_cg_tolerance;
    T linesearch_beta,linesearch_rho,linesearch_maxit;
    T sampling_type;
    // copy parameters from gpu to host
    LaunchMemcpyD2H<GPUDevice,T,USE_CUBLAS>::launch(ctx,this, lambda, &lbd , sizeof(T)  );
    LaunchMemcpyD2H<GPUDevice,T,USE_CUBLAS>::launch(ctx,this, tol, &tolerance , sizeof(T)  );
    LaunchMemcpyD2H<GPUDevice,T,USE_CUBLAS>::launch(ctx,this, maxit, &maxiteration , sizeof(T)  );
    LaunchMemcpyD2H<GPUDevice,T,USE_CUBLAS>::launch(ctx,this, maxcgiter, &max_cg_iterations , sizeof(T)  );
    LaunchMemcpyD2H<GPUDevice,T,USE_CUBLAS>::launch(ctx,this, maxcgtol, &max_cg_tolerance , sizeof(T)  );
    LaunchMemcpyD2H<GPUDevice,T,USE_CUBLAS>::launch(ctx,this, lsbeta, &linesearch_beta , sizeof(T)  );
    LaunchMemcpyD2H<GPUDevice,T,USE_CUBLAS>::launch(ctx,this, lsrho, &linesearch_rho , sizeof(T)  );
    LaunchMemcpyD2H<GPUDevice,T,USE_CUBLAS>::launch(ctx,this, lsmaxit, &linesearch_maxit , sizeof(T)  );
    LaunchMemcpyD2H<GPUDevice,T,USE_CUBLAS>::launch(ctx,this, sampling, &sampling_type , sizeof(T)  );
    // Check that the dimensions of input.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(X.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(target.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(X2.shape()),
                errors::InvalidArgument("In[2] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(Y2.shape()),
                errors::InvalidArgument("In[3] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(W.shape()),
                errors::InvalidArgument("In[4] is not a matrix"));
    TensorShape out_shape(
        {W.dim_size(0), W.dim_size(1)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

     Newton_CG_Multi_Optimized(ctx,
                                          this,
                                          X,
                                          W,
                                          lbd,
                                          tolerance,
                                          out,
                                          target,
                                          maxiteration,
                                          X2,
                                          Y2,
                                          max_cg_iterations,
                                          max_cg_tolerance,
                                          linesearch_beta,
                                          linesearch_rho,
                                          linesearch_maxit,
                                          sampling_type
                                        );
                              
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
};



#define REGISTER_GPU(T)                                            \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("MatMulFang").Device(DEVICE_GPU).TypeConstraint<T>("T"),    \
      MatMulFangOp<GPUDevice, T, true /* cublas, true by default */>); \
  REGISTER_KERNEL_BUILDER(Name("MatMulFang")                           \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T")              \
                              .Label("cublas"),                    \
                          MatMulFangOp<GPUDevice, T, true /* cublas */>)

//TF_CALL_float(REGISTER_CPU);
//TF_CALL_double(REGISTER_CPU);
//TF_CALL_half(REGISTER_CPU);

//TF_CALL_int32(REGISTER_CPU);
//TF_CALL_complex64(REGISTER_CPU);
//TF_CALL_complex128(REGISTER_CPU);

//#if GOOGLE_CUDA
//TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
//TF_CALL_complex64(REGISTER_GPU);
//TF_CALL_complex128(REGISTER_GPU);
//#if CUDA_VERSION >= 7050
//TF_CALL_half(REGISTER_GPU);
//#endif
//#endif  // GOOGLE_CUDA

}  // namespace tensorflow
