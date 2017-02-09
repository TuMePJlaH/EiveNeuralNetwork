#define CUDA_DEBUG

#ifdef CUDA_DEBUG
  #define CUDA_CHECK_ERROR(err)           \
    if (err != cudaSuccess) {          \
      printf("Cuda error: %s\n", cudaGetErrorString(err));    \
      printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
    }                 \

#else
  #define CUDA_CHECK_ERROR(err)
#endif
