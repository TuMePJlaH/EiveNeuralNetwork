//Исходник класса EIVE нейронной сети v 1.0.0
//Функции устройства
#include "eive_neuron_layer.h"
#include "eive_neuron_network.h"

//функция устройства для расчёта полносвязного подключения
__global__ void gpuFc
(
	float *X,
	int X_s,
	float *w,
	float *b,
	float *Y,
	int Y_s,
	int fA
)
{
	int indx = blockDim.x*blockIdx.x + threadIdx.x;

	if(indx < Y_s) {
		float Y_loc = 0;
		for(int i = 0; i < X_s; i++) {
			Y_loc += X[i]*w[i + indx*X_s];
		}
		Y_loc += b[indx];
		Y[indx] = Y_loc;
		switch(fA) {
			case FT_SIGMOID:
				Y[indx] = 1/(1 + exp(-(Y_loc)));
				break;

			case FT_TANH:
				Y[indx] = tanhf(Y_loc);
				break;
		}
	}
}
//----------------------------------------------------------------------------------------------------------
//функция устройства для свёртки входного изображения
__global__ void gpuConvI
(
	float *X,
	int X_w,
	int X_h,
	float *k,
	int k_s,
	int k_n,
	float *b,
	float *Y,
	int Y_w,
	int Y_h,
	int fA
)
{
	int xIndex = blockDim.x*blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y*blockIdx.y + threadIdx.y;
	if((xIndex < Y_w) && (yIndex < Y_h)) {
		for(int z = 0; z < k_n; z++) {
			float Y_loc = 0;
			for(int i = 0; i < k_s; i++) {
				for(int j = 0; j < k_s; j++) {
					int inIndex_x = xIndex + j;
					int inIndex_y = yIndex + i;
					if((inIndex_x >= 0) && (inIndex_y >= 0) && (inIndex_x < X_w) && (inIndex_y < X_h)) {
						Y_loc += k[(k_s-1-j)+(k_s-1-i)*k_s + z*k_s*k_s]*X[inIndex_x + inIndex_y*X_w];
					}
				}
			}
			Y_loc += b[z];
			int outIndex = xIndex + yIndex*Y_w + z*Y_w*Y_h;
			Y[outIndex] = Y_loc;
			switch(fA) {
				case FT_SIGMOID:
					Y[outIndex] = 1/(1 + exp(-(Y_loc)));
					break;

				case FT_TANH:
					Y[outIndex] = tanhf(Y_loc);
					break;
			}
		}
	}
}
//----------------------------------------------------------------------------------------------------------
//Функция устройства для свёртки субдискрет. слоя или свёрточного
__global__ void gpuConvValid
(
	float *X,
	int X_w,
	int X_h,
	int X_n,
	float *k,
	int k_s,
	int k_n,
	float *adj,
	float *b,
	float *Y,
	int Y_w,
	int Y_h,
	int fA
)
{
	int xIndex = blockDim.x*blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y*blockIdx.y + threadIdx.y;

	if((xIndex < Y_w) && (yIndex < Y_h)) {
		for(int z = 0; z < k_n; z++) {
			float Y_loc = 0;
			for(int i = 0; i < k_s; i++) {
				for(int j = 0; j < k_s; j++)
				{
					int inIndex_x = xIndex + j;
					int inIndex_y = yIndex + i;
					if((inIndex_x >= 0) && (inIndex_y >= 0) && (inIndex_x < X_w) && (inIndex_y < X_h))
					{
						float X_loc = 0;
						for(int l = 0; l < X_n; l++) {
							if(adj[l + z*X_n]) {
								X_loc += X[inIndex_x + inIndex_y*X_w + l*X_w*X_h];
							}
						}
						Y_loc += k[(k_s-1-j)+(k_s-1-i)*k_s + z*k_s*k_s]*X_loc;
					}
				}
			}
			Y_loc += b[z];
			int outIndex = xIndex + yIndex*Y_w + z*Y_w*Y_h;
			switch(fA)	{
				case FT_SIGMOID:
					Y[outIndex] = 1/(1 + exp(-(Y_loc)));
					break;

				case FT_TANH:
					Y[outIndex] = tanhf(Y_loc);
					break;
			}
		}
	}
}
//----------------------------------------------------------------------------------------------------------
//Функция устройства субдискретизации
__global__ void gpuMaxPool
(
	float *X,
	int X_w,
	int X_h,
	int C_s,
	float *Y,
	int Y_w,
	int Y_h,
	int C_n,
	float *mask
)
{
	int xIndex = blockDim.x*blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y*blockIdx.y + threadIdx.y;

	if((xIndex < Y_w) && (yIndex < Y_h)) {
		for(int z = 0; z < C_n; z++) {
			int inIndex_x = 0;
			int inIndex_y = 0;
			int inIndex = 0;
			float max = 0;
			inIndex_x = xIndex*C_s;
			inIndex_y = yIndex*C_s;
			inIndex = inIndex_x + inIndex_y*X_w + z*X_w*X_h;
			max = X[inIndex];
			int max_idx = inIndex;
			for(int i = 0; i < C_s; i++) {
				for(int j = 0; j < C_s; j++)
				{
					inIndex_x = xIndex*C_s + j;
					inIndex_y = yIndex*C_s + i;
					inIndex = inIndex_x + inIndex_y*X_w + z*X_w*X_h;
					mask[inIndex] = 0;
					if(max < X[inIndex]) {
						max = X[inIndex];
						max_idx = inIndex;
					}
				}
			}
			mask[max_idx] = 1;
			Y[xIndex + yIndex*Y_w + z*Y_w*Y_h] = max;
		}
	}
}
//----------------------------------------------------------------------------------------------------------
//Функция устройства для расчёта ошибки на последнем слое
__global__ void gpuDeltaO
(
	float *d,
	float *Y,
	int Y_s,
	float *delta,
	int fA
)
{
	int Index = blockDim.x*blockIdx.x + threadIdx.x;
	if(Index < Y_s) {
		float delta_loc = (d[Index] - Y[Index]);
		switch(fA) {
			case FT_SIGMOID:
				delta[Index] = delta_loc*(Y[Index]*(1 - Y[Index]));
				break;

			case FT_TANH:
				delta[Index] = delta_loc*(1-pow(tanhf(Y[Index]),2.0f));
				break;
		}
	}
}
//----------------------------------------------------------------------------------------------------------
//Функция устройства для расчёта ошибки следующего за полносвязным слоем
__global__ void gpuDeltaFC
(
	float *deltaN,
	int dN_n,
	float *w,
	float *delta,
	float *Y,
	int Y_s,
	int fA
)
{
	int Index = blockDim.x*blockIdx.x + threadIdx.x;
	if(Index < Y_s)	{
		float delta_loc = 0;
		for(int i = 0; i < dN_n; i++) {
			delta_loc += deltaN[i]*w[Index + i*Y_s];
		}
		switch(fA) {
			case FT_SIGMOID:
				delta[Index] = delta_loc*(Y[Index]*(1 - Y[Index]));
				break;

			case FT_TANH:
				delta[Index] = delta_loc*(1-pow(tanhf(Y[Index]),2.0f));
				break;
		}
	}
}
//----------------------------------------------------------------------------------------------------------
//Функция устройста для расчёта ошибки слоя следующего за свёрточным слоем
__global__ void gpuKorFull
(
	float *deltaN,
	int dN_w,
	int dN_h,
	float *k,
	int k_s,
	int k_n,
	float *adj,
	float *Y,
	int Y_w,
	int Y_h,
	int Y_n,
	float *delta,
	int fA
)
{
	int xIndex = blockDim.x*blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y*blockIdx.y + threadIdx.y;

	if((xIndex < Y_w) && (yIndex < Y_h)) {
		for(int z = 0; z < Y_n; z++) {
			float delta_loc = 0;
			for(int l = 0; l < k_n; l++) {
				if(adj[l*Y_n + z]) {
					for(int i = 0; i < k_s; i++) {
						for(int j = 0; j < k_s; j++) {
							int inIndex_x = xIndex - j;
							int inIndex_y = yIndex - i;
							int inIndex = inIndex_x + inIndex_y*dN_w + l*dN_w*dN_h;
							if((inIndex_x >= 0) && (inIndex_y >= 0) && (inIndex_x < dN_w) && (inIndex_y < dN_h)) {
								delta_loc += k[(k_s-1-j) + (k_s-1-i)*k_s + l*k_s*k_s]*deltaN[inIndex];
							}
						}
					}
				}
			}

			int Index = xIndex + yIndex*Y_w + z*Y_w*Y_h;

			switch(fA) {
				case FT_SIGMOID:
					delta[Index] = delta_loc*(Y[Index]*(1 - Y[Index]));
					break;

				case FT_TANH:
					delta[Index] = delta_loc*(1-pow(tanhf(Y[Index]),2.0f));
					break;
			}
		}
	}
}
//----------------------------------------------------------------------------------------------------------
//Функция для устройства для расчёта ошибки слоя следующего за субдискретизирующим
__global__ void gpuDeltaMP
(
	float *deltaN,
	int dN_w,
	int dN_h,
	int C_s,
	int C_n,
	float *delta,
	int d_w,
	int d_h,
	float *mask
)
{
	int xIndex = blockDim.x*blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y*blockIdx.y + threadIdx.y;
	if((xIndex < dN_w) && (yIndex < dN_h)) {
		for(int z = 0; z < C_n; z++) {
			for(int i = 0; i < C_s; i++) {
				for(int j = 0; j < C_s; j++) {
					int outIndex_x = xIndex*C_s + j;
					int outIndex_y = yIndex*C_s + i;
					int outIndex = outIndex_x + outIndex_y*d_w + z*d_w*d_h;
					delta[outIndex] = deltaN[xIndex + yIndex*dN_w + z*dN_w*dN_h]*mask[outIndex];
				}
			}
		}
	}
}
//----------------------------------------------------------------------------------------------------------
//Функция корректировки веса для полносвязного слоя
__global__ void gpyChangeWeightFC
(
	float *delta,
	float *w,
	int w_n,
	float *X,
	int X_s,
	float *b,
	float speed
)
{
	int Index = blockDim.x*blockIdx.x + threadIdx.x;
	if(Index < w_n)	{
		float d_w = 0;
		for(int i = 0; i < X_s; i++) {
			d_w = delta[Index]*X[i];
			w[i + Index*X_s] += speed*d_w;
		}
		float d_b = speed*delta[Index];
		b[Index] += d_b;
	}
}
//----------------------------------------------------------------------------------------------------------
//Функция корректировки веса для первого свёрточного слоя
__global__ void gpuChangeWeightCNNI
(
	float *delta,
	int d_w,
	int d_h,
	float *X,
	int X_w,
	int X_h,
	float *k,
	int k_s,
	int k_n,
	float *b,
	float speed
)
{
	int Index = blockDim.x*blockIdx.x + threadIdx.x;
	if(Index < k_n) {
		for(int i = 0; i < k_s; i++) {
			for(int j = 0; j < k_s; j++) {
				float dk_loc = 0;
				for(int l = 0; l < d_h; l++) {
					for(int o = 0; o < d_w; o++) {
						dk_loc += X[(j+o) + (i+l)*X_w]*delta[o + l*d_w + Index*d_w*d_h];
					}
				}
				k[(k_s-1-j) + (k_s-1-i)*k_s + Index*k_s*k_s] += speed*dk_loc;
			}
		}

		float db_loc = 0;
		for(int l = 0; l < d_h; l++) {
			for(int o = 0; o < d_w; o++) {
				db_loc += delta[o + l*d_w + Index*d_w*d_h];
			}
		}
		b[Index] += speed*db_loc;
	}
}
//----------------------------------------------------------------------------------------------------------
//Функция корректировки веса для последующих свёрточных слоёв
__global__ void gpuChangeWeightCNN
(
	float *delta,
	int d_w,
	int d_h,
	float *X,
	int X_w,
	int X_h,
	int X_n,
	float *k,
	int k_s,
	int k_n,
	float *adj,
	float *b,
	float speed
)
{
	int Index = blockDim.x*blockIdx.x + threadIdx.x;
	if(Index < k_n)	{
		for(int i = 0; i < k_s; i++) {
			for(int j = 0; j < k_s; j++) {
				float dk_loc = 0;
				for(int l = 0; l < d_h; l++) {
					for(int o = 0; o < d_w; o++) {
						for(int z = 0; z < X_n; z++) {
							if(adj[z + Index*X_n]) {
								dk_loc += X[(j+o) + (i+l)*X_w + z*X_w*X_h]*delta[o + l*d_w + Index*d_w*d_h];
							}
						}
					}
				}
				k[(k_s-1-j) + (k_s-1-i)*k_s + Index*k_s*k_s] += speed*dk_loc;
			}
		}

		float db_loc = 0;
		for(int l = 0; l < d_h; l++) {
			for(int o = 0; o < d_w; o++) {
				db_loc += delta[o + l*d_w + Index*d_w*d_h];
			}
		}
		b[Index] += speed*db_loc;
	}
}
