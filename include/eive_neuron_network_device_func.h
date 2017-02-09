//функция устройства для расчёта полносвязного подключения
__global__ void gpuFc(float *X, int X_s, float *w, float *b, float *Y, int Y_s, int fA);
//функция устройства для свёртки входного изображения
__global__ void gpuConvI(float *X, int X_w, int X_h, float *k, int k_s, int k_n, float *b,float *Y, int Y_w, int Y_h, int fA);
//Функция устройства для свёртки субдискрет. слоя или свёрточного
__global__ void gpuConvValid(float *X, int X_w, int X_h,int X_n,float *k, int k_s, int k_n, float *adj,float *b,float *Y, int Y_w, int Y_h, int fA);
//Функция устройства субдискретизации
__global__ void gpuMaxPool(float *X, int X_w, int X_h, int C_s, float *Y, int Y_w, int Y_h, int C_n, float *mask);
//Функция устройства для расчёта ошибки на последнем слое
__global__ void gpuDeltaO(float *d, float *Y, int Y_s, float *delta, int fA);
//Функция устройства для расчёта ошибки следующего за полносвязным слоем
__global__ void gpuDeltaFC(float *deltaN, int dN_n, float *w, float *delta, float *Y, int Y_s, int fA);
//Функция устройста для расчёта ошибки слоя следующего за свёрточным слоем
__global__ void gpuKorFull(float *deltaN, int dN_w, int dN_h, float *k, int k_s, int k_n, float *adj, float *Y, int Y_w, int Y_h, int Y_n, float *delta, int fA);
//Функция для устройства для расчёта ошибки слоя следующего за субдискретизирующим
__global__ void gpuDeltaMP(float *deltaN, int dN_w, int dN_h, int C_s, int C_n, float *delta, int d_w, int d_h, float *mask);
//Функция корректировки веса для полносвязного слоя
__global__ void gpyChangeWeightFC(float *delta, float *w, int w_n, float *X, int X_s, float *b, float speed);
//Функция корректировки веса для первого свёрточного слоя
__global__ void gpuChangeWeightCNNI(float *delta,int d_w, int d_h, float *X, int X_w, int X_h, float *k, int k_s, int k_n, float *b, float speed);
//Функция корректировки веса для последующих свёрточных слоёв
__global__ void gpuChangeWeightCNN(float *delta, int d_w, int d_h, float *X, int X_w, int X_h, int X_n, float *k, int k_s, int k_n, float *adj, float *b, float speed);
