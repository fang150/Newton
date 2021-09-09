#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "matmulfang_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "cuda_types.h"
#include "classification_kernels.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow{

#define EIGEN_USE_GPU

typedef Eigen::GpuDevice GPUDevice;
// Define the CUDA kernel.


#define cudaCheckError()    __cudaCheckError(  )
inline void __cudaCheckError(  )
{
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		fprintf (stderr, "Failed ..  -- gpu erro code %d:%s\n",  err, cudaGetErrorString( err ) );
		exit( -1 );
	}

	return;
}

template <typename T>
GLOBAL void ker_dx_softmax_ind( T *hxw, const T *target, int rows, int num_classes, T *result, int threads_per_row)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    int myClrId = idx % threads_per_row; 	
    int myRowId = idx / threads_per_row; 

    T indicator = 0; 
    int r = 0; 
    
    //for (int r = idx; r < rows; r += gridDim.x * blockDim.x){
    
    if (idx < rows ) {
        r = idx; 
        for (int clr = 0; clr < num_classes; clr ++ ){
            result[ clr * rows + r ] = hxw[ clr * rows + r ];
            if (clr == (int)(target[ r ] - 1.)) result[ clr * rows + r ] -= 1.; 

            //result[ clr * rows + r ] = 0;
            //if (clr == (int)(target[ r ] - 1.)) result[ clr * rows + r ] = 1; 
        }
    }
}



template <typename T>
GLOBAL void ker_add_regularizer ( T *input, T *vector, T lambda, int count, T normalizer)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (idx < count) input[ idx ] += lambda * vector[ idx ] ;
}



template <typename T>
GLOBAL void ker_sample_dataset(const T *src, int rows, int cols, int *indices, T *dest, int sampleSize ){
	
	   int idx = blockDim.x * blockIdx.x + threadIdx.x;
	   int offset = -1;
	
	   if (idx < sampleSize){
		  offset = indices[ idx ];
		  for (int j = 0; j < cols; j ++){
			 dest[ j * sampleSize + idx ] = src[ j * rows + offset ];
		  }
	   }
	}


template <typename T>
GLOBAL void ker_hx_C (T *A, T *B, T *C, int rows, int cols, int num_classes )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 

	T sum = 0; 
	if (idx < rows){
		for (int i = 0; i < num_classes; i ++) 
			sum += A[ idx + i * rows ] * B[ idx + i * rows ];

		for (int i = 0; i < num_classes; i ++) 
			C[ i * rows + idx ] = 
			 	A[ idx + i * rows ] * B[ idx + i * rows ] - 
				B[ idx + i * rows ] * sum ;
	}
}


template <typename T>
GLOBAL void ker_compute_HXW( T *XW, int rows, int cols, int numclasses, int threads_per_col )
{
	int myColId = ( blockIdx.x * blockDim.x + threadIdx.x ) % threads_per_col; 	
	int myRowId = ( blockIdx.x * blockDim.x + threadIdx.x ) / threads_per_col; 	
	int myWarpId = (blockIdx.x * blockDim.x + threadIdx.x ) % WARP_SIZE; 

	T sdata = 0; 
	int i = 0; 

	T maxdot = 0; 

	//for (int i = myRowId; i < rows; i += gridDim.x * blockDim.x){
	if (myRowId < rows) {
		i = myRowId;

		maxdot = 0; 
		for (int j = 0; j < numclasses; j += threads_per_col ) {
			if (maxdot < XW[ j * rows + i ]) maxdot = XW[ j * rows + i ]; 
		}

		sdata = 0; 
		for (int j = 0; j < numclasses; j += threads_per_col ) sdata += exp ( XW[ j * rows + i ] - maxdot ); 

		//for (int offset = threads_per_col/2; offset > 0; offset /= 2) sdata += my_shfl( sdata, myWarpId + offset ); 

		for (int j = 0; j < numclasses; j += threads_per_col ) 
			XW[ j * rows + i ] = exp( XW[ j * rows + i ] - maxdot ) / (exp(-1. * maxdot) + sdata); 
	}
}
		

template <typename T>
GLOBAL void ker_dx_softmax (T *features, T *target, int rows, int cols, int num_classes, 
			T *weights, T lambda, T *wspace )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int lane = threadIdx.x >> 5;
	extern __shared__ T sh_vec[];

	T numerator = 0.; 
	T denominator = 0.; 
	int indicator = 0; 
	T multiplier = 0.; 
	T blk_sum = 0.; 
	T p_i = 0.; 

	T maxdot = 0.;

	if (idx < rows) indicator = (int)(target[ idx ] - 1.); 
	__syncthreads ();

        //maxdot here. 
        for (int i = 0; i < num_classes; i ++){
                if (threadIdx.x < cols) sh_vec[threadIdx.x] = weights[i * cols + threadIdx.x ];
                __syncthreads ();

                numerator = 0.;
                if (idx < rows) {
                        for (int j = 0; j < cols; j ++)
                                numerator += sh_vec[j] * features[ j * rows + idx ];

                        if (maxdot < numerator) maxdot = numerator;
                }
                __syncthreads ();
        }


	//denominator here. 
	for (int i = 0; i < num_classes; i ++){
		if (threadIdx.x < cols) sh_vec[threadIdx.x] = weights[i * cols + threadIdx.x ];	
		__syncthreads ();
	
		numerator = 0.;
		if (idx < rows) {
			for (int j = 0; j < cols; j ++)
				numerator += sh_vec[j] * features[ j * rows + idx ]; 	
			denominator  += exp( numerator - maxdot );
		}
		__syncthreads ();
	}

	//numerator here. 
	//dw_i (j) here. 
	for (int i = 0; i < num_classes; i ++){
		if (threadIdx.x < cols) sh_vec[threadIdx.x] = weights[i * cols + threadIdx.x];	
		__syncthreads ();

		numerator = 0; 
		if ( idx < rows ){
			for (int j = 0; j < cols; j ++)
				numerator += sh_vec[j] * features[ j * rows + idx ]; 	
			numerator = exp( numerator - maxdot );
			//p_i = numerator / (1 + denominator); 
			p_i = numerator / (exp(1. * maxdot) + denominator); 

			if (i == indicator) multiplier = 1.0;
			else multiplier = 0.;
		}
		__syncthreads ();

		for (int j = 0; j < cols; j ++){ 
			blk_sum = 0.; 
			if (idx < rows)
				blk_sum = (p_i - multiplier) * features[ j * rows + idx ];
			
        		__syncthreads ();

			// block level reduction here. 
        		blk_sum  = warpSum( blk_sum);
        		if (threadIdx.x % WARP_SIZE == 0) sh_vec[lane] = blk_sum;
        		__syncthreads ();

        		if (blockDim.x/WARP_SIZE == 0)
        			blk_sum = (threadIdx.x < 1) ? sh_vec[threadIdx.x] : 0;
        		else
        			blk_sum = (threadIdx.x < (blockDim.x / WARP_SIZE) ) ? sh_vec[ threadIdx.x ] : 0;
        		__syncthreads ();

        		if (lane == 0) blk_sum = warpSum( blk_sum );
        		if (threadIdx.x == 0) wspace[ (blockIdx.x * num_classes * cols) +  ( i * cols + j )  ] = blk_sum;
        		__syncthreads ();
		}
	}
}


template <typename T>
GLOBAL void reduce_vector_warp( const T *input, const T *maxdots, T *results, const size_t numcomps, int numblocks )
{
	extern __shared__ T my_results[]; 

	unsigned int lane  = threadIdx.x >> 5; 
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x; 

	T sdata; 
        sdata = 0.;

	if (idx < numcomps ){
		for (int c = 0; c < numblocks; c ++) sdata += input [ c * numcomps + idx ]; 
		results[ idx ] = sdata + exp( -1. * maxdots[ idx ] ); 
	}
}

template <typename T>
GLOBAL void reduce_log(const T *input, T *results, const size_t count) {
        extern __shared__ T my_results[];
        unsigned int lane = threadIdx.x >> 5;
        unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

        T sdata;
        T x = 0;

        sdata = 0;
        my_results[ lane ] = 0;
        if(idx < count) x = log(input [idx] );
        sdata = x;

        sdata = warpSum ( sdata );
        if (threadIdx.x % WARP_SIZE == 0) my_results[lane] = sdata;
        __syncthreads ();

        if (blockDim.x/WARP_SIZE == 0)
        	sdata = (threadIdx.x < 1) ? my_results[threadIdx.x] : 0;
        else
        	sdata = (threadIdx.x < (blockDim.x/WARP_SIZE)) ? my_results[threadIdx.x] : 0;
        __syncthreads ();

        if (lane == 0) sdata = warpSum( sdata );
        if(threadIdx.x == 0) results [ blockIdx.x  ] =  sdata;
}


////Hessian functions here. 

template <typename T>
GLOBAL void ker_hx_Xv ( T *features, T *vector, int rows, int cols, int num_classes, T *A ) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	extern __shared__ T sh_vec[];

	T dot = 0; 

	for (int i = 0; i < num_classes; i ++){
		if (threadIdx.x < cols) sh_vec[threadIdx.x] = vector [i * cols + threadIdx.x ];	
		__syncthreads ();
	
		if (idx < rows) {
			dot = 0; 
			for (int j = 0; j < cols; j ++) dot += sh_vec[j] * features[ j * rows + idx ]; 	
			A[ idx + i * rows ] = dot;  // column major format here. 
		}
		__syncthreads ();
	}
}


template <typename T>
GLOBAL void ker_compute_fx (T *matvec, int rows, int cols, int numclasses, 
				const T *target, T *indicatorVal, int NUM_THREADS, T *maxdots )
{
	extern __shared__ T my_results[];

	int idx =  blockIdx.x * blockDim.x + threadIdx.x; 
	int myClrId = idx % NUM_THREADS; 
	int myRowId = idx / NUM_THREADS; 
        unsigned int lane = threadIdx.x >> 5;

	T sdata = 0; 
	T maxdot = 0; 
	
	//if (myRowId < rows) {
	for (int r = myRowId; r < rows; r += gridDim.x * blockDim.x ) {
		maxdot = 0; 
		 for (int i = myClrId; i < numclasses; i += NUM_THREADS){
			if (maxdot < matvec[ i * rows + r ]) maxdot = matvec[ i * rows + r]; 
	 	 }

		maxdots[ r ] = maxdot; 

		 for (int i = myClrId; i < numclasses; i += NUM_THREADS){
			if ((int)target[ r ] == (i + 1)) sdata += matvec[ i * rows + r ]; 
			matvec[ i * rows + r ] = exp( matvec[ i * rows + r ]  - maxdot); 
		 } 
	}
	__syncthreads (); 

        sdata = warpSum ( sdata );
        if (threadIdx.x % WARP_SIZE == 0) my_results[lane] = sdata ;
        __syncthreads ();

        if (blockDim.x/WARP_SIZE == 0)
        	sdata = (threadIdx.x < 1) ? my_results[threadIdx.x] : 0;
        else
        	sdata = (threadIdx.x < (blockDim.x/WARP_SIZE)) ? my_results[threadIdx.x] : 0;
        __syncthreads ();

        if (lane == 0) sdata = warpSum( sdata );
        if(threadIdx.x == 0) indicatorVal [ blockIdx.x  ] =  sdata;
}


template <typename T>
GLOBAL void ker_softmax_predict( T *test_set, T *weights, 
	int rows, int cols, int numclasses, T *workspace)
{
extern __shared__ T sh_vec[];
int idx = blockIdx.x * blockDim.x + threadIdx.x;
T dot = 0;
T sumexp; 
T sumprob; 

//probability terms here. 
sumexp = 0; 
for (int i = 0; i < numclasses; i ++){
if (threadIdx.x < cols) sh_vec[threadIdx.x] = weights[i * cols + threadIdx.x];	
__syncthreads ();

if ( idx < rows ){
dot = 0; 
for (int j = 0; j < cols; j ++) dot += sh_vec[j] * test_set[ j * rows + idx ]; 	
sumexp += exp( dot );
}
__syncthreads ();
}

for (int c = 0; c < numclasses; c ++) {
if (threadIdx.x < cols) sh_vec[ threadIdx.x ] = weights[ c * cols + threadIdx.x ];
__syncthreads ();

if (idx < rows){
dot = 0.; 
for (int i = 0; i < cols; i ++) dot += test_set[i * rows + idx] * sh_vec[i];
workspace[ idx * numclasses + c ] = exp(dot) / (1 + sumexp);
}
__syncthreads ();
}
}


namespace functor{
	template<typename T>
	struct FtorSoftmaxPredict<GPUDevice,T> {
	  void operator()(  const GPUDevice& d, T *test_set, T *weights, 
		int rows, int cols, int numclasses, T *workspace,int pblocks ,int BLOCK_SIZE   ){		
			ker_softmax_predict <<< pblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(T) >>> 
			( test_set, weights, rows, cols, numclasses, workspace);
			cudaThreadSynchronize (); 
			cudaCheckError();	 
	}
	};	
	}






template <typename T>
GLOBAL void reduce(const T *input, T *results, const size_t count) {
        extern __shared__ T my_results[];
        unsigned int lane = threadIdx.x >> 5;
        unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

        T sdata;
        T x = 0;

        sdata = 0;
        my_results[ lane ] = 0;
        if(idx < count) x = input [idx];
        sdata = x;

        sdata = warpSum ( sdata );
        if (threadIdx.x % WARP_SIZE == 0) my_results[lane] = sdata;
        __syncthreads ();

        if (blockDim.x/WARP_SIZE == 0)
         sdata = (threadIdx.x < 1) ? my_results[threadIdx.x] : 0;
        else
         sdata = (threadIdx.x < (blockDim.x/WARP_SIZE)) ? my_results[threadIdx.x] : 0;
        __syncthreads ();

        if (lane == 0) sdata = warpSum( sdata );
        if(threadIdx.x == 0) results [ blockIdx.x  ] =  sdata;
}


 void compute_blocks ( int *blocks, int *block_size, int count )
 {
         *block_size = CUDA_BLOCK_SIZE;
         *blocks = (count / CUDA_BLOCK_SIZE ) + (count % CUDA_BLOCK_SIZE == 0     ? 0 : 1);
 }


 void compute_nearest_pow_2 (int blocks, int *result)
 {
         int power = 1;
         while (power < blocks) power *= 2;
 
         *result = power;
 }



namespace functor{

	template<typename T>
	struct FtorSampler<GPUDevice,T> {
	  void operator()(  const GPUDevice& d, const T *dataset, int rows, int cols, int *indices, T *dest, int sampleSize, int BLOCK_SIZE   ){
	
		 int blocks = (sampleSize / BLOCK_SIZE) + ((sampleSize % BLOCK_SIZE) == 0 ? 0 : 1);
		 //fprintf( stderr, "Blocks for sampling: %d, %d, %d \n", blocks, blocks * BLOCK_SIZE, sampleSize );
		 ker_sample_dataset <<< blocks, BLOCK_SIZE >>>
			(dataset, rows, cols, indices, dest, sampleSize );
		 cudaThreadSynchronize ();
		 cudaCheckError ();
	
	}
	
	};
	

}


/*void mySampler( real *dataset, int rows, int cols, int *indices, real *dest, int sampleSize){
	
	   int blocks = (sampleSize / BLOCK_SIZE) + ((sampleSize % BLOCK_SIZE) == 0 ? 0 : 1);
	   fprintf( stderr, "Blocks for sampling: %d, %d, %d \n", blocks, blocks * BLOCK_SIZE, sampleSize );
	
	   ker_sample_dataset <<< blocks, BLOCK_SIZE >>>
		  (dataset, rows, cols, indices, dest, sampleSize );
	   cudaThreadSynchronize ();
	   cudaCheckError ();
	}
*/
namespace functor{

template<typename T>
struct FtorAddRegularizer<GPUDevice,T> {
  void operator()(  const GPUDevice& d, T *input, T *vector, T lambda, int count, T normalizer, int rblocks, int BLOCK_SIZE   ){
   
         ker_add_regularizer <<< rblocks, BLOCK_SIZE >>>
         (input, vector, lambda, count, 1. );
         cudaThreadSynchronize ();
		 cudaCheckError();
}

};

}


namespace functor{

template<typename T>
struct FtorComputeBlocks<GPUDevice,T> {
  void operator()(const GPUDevice& d, int *blocks, int *block_size, int count  ){

	compute_blocks (  blocks,  block_size,  count );


}
};

}

namespace functor{

template<typename T>
struct FtorComputePOW2<GPUDevice,T> {
  void operator()(const GPUDevice& d, int blocks, int *result  ){

    compute_nearest_pow_2 (blocks, result);


}
};

}




namespace functor{

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct FtorComputeHXW<GPUDevice, T> {
  void operator()(const GPUDevice& d,T *XW, int rows, int cols, int numclasses, int threads_per_col, int BLOCKS, int BLOCK_SIZE ) {

		//ker_compute_HXW <<< BLOCKS, BLOCK_SIZE >>> 
		ker_compute_HXW<T> <<< BLOCKS, BLOCK_SIZE,0,d.stream() >>> 
			( XW, rows, cols, numclasses, 1); 
	
	cudaThreadSynchronize (); 
	cudaCheckError ();

  }
};

}




namespace functor{

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct FtorHxC<GPUDevice, T> {
  void operator()(const GPUDevice& d,T *A,T *B,T *C, int rows, int cols, int num_classes, int BLOCKS, int BLOCK_SIZE ) {




		ker_hx_C<T> <<< BLOCKS, BLOCK_SIZE,0,d.stream() >>> 
		           (A, B, C, rows, cols, num_classes);
        cudaThreadSynchronize ();
		cudaCheckError ();



  }
};

}


namespace functor{


template < typename T>
struct FtorDxSoftmaxInd<GPUDevice,T> {
  void operator()(  const GPUDevice& d,  T *hxw, const T *target, int rows, int num_classes, T *result, int threads_per_row,int BLOCKS,int BLOCK_SIZE){
 

	ker_dx_softmax_ind <T><<<BLOCKS,BLOCK_SIZE>>>
    ( hxw, target, rows, num_classes, result, 1);
    cudaDeviceSynchronize ();
	cudaThreadSynchronize ();
	cudaCheckError();
}
};

}



namespace functor{
	template < typename T>
	struct FtorReduce<GPUDevice,T> {
	  void operator()(  const GPUDevice& d,  const  T *input, T *results ,const size_t count, int BLOCKS,int BLOCK_SIZE){

		reduce<T> 
        <<< BLOCKS, BLOCK_SIZE, WARP_SIZE * sizeof (T), d.stream() >>>
          (input, results , count );

		  cudaThreadSynchronize();
		  cudaCheckError();
	}
	};
	
	
}



namespace functor{
	template < typename T>
	struct FtorReduceLog<GPUDevice,T> {
	  void operator()(  const GPUDevice& d,  const T *input, T *results, const size_t count, int BLOCKS,int BLOCK_SIZE){


		reduce_log <<< BLOCKS, BLOCK_SIZE, WARP_SIZE* sizeof(T) >>> 
		
				( input, results, count ); 
			cudaThreadSynchronize ();
			cudaCheckError();
	}
	};
	
	
}



namespace functor{
	template < typename T>
	struct FtorReduceVectorWarp<GPUDevice,T> {
	  void operator()(  const GPUDevice& d,  const T *input, const T *maxdots, T *results, const size_t numcomps, int numblocks, int BLOCKS,int BLOCK_SIZE){

		reduce_vector_warp <<< BLOCKS, BLOCK_SIZE >>> 
		(input, maxdots, results, numcomps, numblocks ); 
		  cudaThreadSynchronize();
		  cudaCheckError();
	}
	};
	
	
}


namespace functor{

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct FtorComputeFx<GPUDevice, T> {
  void operator()(  const GPUDevice& d,  T* devptr,int rows,int cols,int num_classes,  const T* target, T* indicatorVal ,int NUM_THREADS  , T* maxdots,int BLOCKS,int BLOCK_SIZE,int BLOCKS_POW_2 ) {

    NUM_THREADS=1;
    ker_compute_fx<T> 
        <<< BLOCKS * NUM_THREADS, BLOCK_SIZE, WARP_SIZE * sizeof(T),d.stream()  >>> 
			( devptr, rows, cols, num_classes, target, indicatorVal, NUM_THREADS, maxdots); 
	cudaThreadSynchronize ();
	cudaCheckError();
    
  }
};
}


/*namespace functor{

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct FtorComputeFx<GPUDevice, T> {
  void operator()(  const GPUDevice& d,  T* devptr,int rows,int cols,int num_classes,  const T* target, T* indicatorVal ,int NUM_THREADS  , T* maxdots,int BLOCKS,int BLOCK_SIZE,int BLOCKS_POW_2, T* pageLckPtr,T* alphax) {


    //std::cout<<"block "<<BLOCKS<<std::endl;
    
    NUM_THREADS=1;
    ker_compute_fx<T> 
        <<< BLOCKS * NUM_THREADS, BLOCK_SIZE, WARP_SIZE * sizeof(T),d.stream()  >>> 
			( devptr, rows, cols, num_classes, target, indicatorVal, NUM_THREADS, maxdots); 
	cudaThreadSynchronize ();
    reduce<T> 
        <<< BLOCKS, BLOCK_SIZE, WARP_SIZE * sizeof (T),d.stream() >>>
          (maxdots, maxdots + rows, rows );

   cudaThreadSynchronize();
   //cudaCheckError(); 

   reduce <<< 1, BLOCKS_POW_2, WARP_SIZE * sizeof( T ) >>> 

		(maxdots + rows, &pageLckPtr[3], BLOCKS ); 
	cudaThreadSynchronize (); 
 
	
	
	// final value of the indicator
	reduce <<< 1, BLOCKS_POW_2, WARP_SIZE * sizeof(T) >>> 

		( indicatorVal, &pageLckPtr[0], BLOCKS ); 
	cudaThreadSynchronize (); 
	//cudaCheckError ();	


	//compute the log part here. 
	int warp_blocks = ((rows * WARP_SIZE) / BLOCK_SIZE) + 
				(((rows * WARP_SIZE) % BLOCK_SIZE == 0) ? 0 : 1); 

	reduce_vector_warp <<< BLOCKS, BLOCK_SIZE >>> 
		(devptr, maxdots, alphax, rows, num_classes ); 
	cudaThreadSynchronize (); 


	//final log part here. 
	reduce_log <<< BLOCKS, BLOCK_SIZE, WARP_SIZE* sizeof(T) >>> 

		( alphax, alphax + rows, rows ); 
	cudaThreadSynchronize ();


	reduce <<< 1, BLOCKS_POW_2, WARP_SIZE * sizeof(T) >>> 

		( alphax + rows, &pageLckPtr[1], BLOCKS);	
	cudaThreadSynchronize ();


	
    
  }
};

}*/





//}

// Instantiate functors for the types of OpKernels registered.
template struct functor::FtorAddRegularizer<GPUDevice,double>;
template struct functor::FtorComputeBlocks<GPUDevice,double>;
template struct functor::FtorComputePOW2<GPUDevice,double>;
template struct functor::FtorComputeFx<GPUDevice, double>;
template struct functor::FtorHxC<GPUDevice, double>;
template struct functor::FtorComputeHXW<GPUDevice, double>;
template struct functor::FtorDxSoftmaxInd<GPUDevice,double>;
template struct functor::FtorReduce<GPUDevice,double>;
template struct functor::FtorReduceVectorWarp<GPUDevice,double>;
template struct functor::FtorReduceLog<GPUDevice,double>;
template struct functor::FtorSoftmaxPredict<GPUDevice,double>;
template struct functor::FtorSampler<GPUDevice,double>;




}
#endif  // GOOGLE_CUDA
