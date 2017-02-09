//Исходник класса EIVE нейронной сети v 1.0.0
//функция для работы
#include "cuda_debug.h"
#include "eive_neuron_network.h"
#include "eive_neuron_network_device_func.h"
#include "iostream"

//функция прямого прогона сети
int EiveNeuronNetwork::colculateNetwork(float *inputData)
{
  if(!L) {	//если сеть не создана
    std::cout << "[ERROR][F:colculateNetwork]:the network is not created!" << std::endl;
    return EIVE_ERROR;
  }

  //необходимые локальные переменные
  cudaEvent_t syncEvent;
  cudaEventCreate(&syncEvent);
  dim3 gridSize;
  dim3 blockSize;

  //копируем данные в память устройства
  CUDA_CHECK_ERROR(cudaMemcpy(inDataDev, inputData, inDataSize*sizeof(float), cudaMemcpyHostToDevice));

  //для первого слоя
  switch(L[0].layerType) {
    case LT_FULLYCONNECTED:	//если слой полносвязный
      gridSize = dim3((L[0].fc_nNeuron + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1, 1);
      blockSize = dim3(MIN_BLOCK_SIZE, 1, 1);
      gpuFc<<<gridSize, blockSize>>>
      (
        inDataDev,
        inDataSize,
        L[0].fc_w_dev,
        L[0].fc_b_dev,
        L[0].fc_y_dev,
        L[0].fc_nNeuron,
        L[0].fActiv
      );
#ifdef DEBUG_NETWORK	//для отладки сети
      CUDA_CHECK_ERROR(cudaMemcpy(L[0].fc_y, L[0].fc_y_dev, L[0].fc_nNeuron*sizeof(float), cudaMemcpyDeviceToHost));
#endif
      break;

    case LT_CONVOLUTION:	//если слой свёрточный
      gridSize = dim3((L[0].cnn_sMapW + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, (L[0].cnn_sMapH + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1);
      blockSize = dim3(MIN_BLOCK_SIZE, MIN_BLOCK_SIZE, 1);
      gpuConvI<<<gridSize, blockSize>>>
      (
        inDataDev,
        inDataW,
        inDataH,
        L[0].cnn_k_dev,
        L[0].cnn_sCore,
        L[0].cnn_nCore,
        L[0].cnn_b_dev,
        L[0].cnn_y_dev,
        L[0].cnn_sMapW,
        L[0].cnn_sMapH,
        L[0].fActiv
      );
#ifdef DEBUG_NETWORK	//для отладки сети
      CUDA_CHECK_ERROR(cudaMemcpy(L[0].cnn_y, L[0].cnn_y_dev, L[0].cnn_sMap*L[0].cnn_nCore*sizeof(float), cudaMemcpyDeviceToHost));
#endif
      break;
  }

  //синхронизируем
  cudaEventRecord(syncEvent, 0);
  cudaEventSynchronize(syncEvent);

  //и бежим по всем слоям
  for(int i = 1; i < nLayer-1; i++) {
    switch(L[i].layerType) {	//проверяем тип слоя
      case LT_FULLYCONNECTED:	//если слой полносвязный
        gridSize = dim3((L[i].fc_nNeuron + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1, 1);
        blockSize = dim3(MIN_BLOCK_SIZE, 1, 1);
        switch(L[i-1].layerType) {	//проверяем тип предыдущего слоя
          case LT_FULLYCONNECTED: //если слой полносвязный
            gpuFc<<<gridSize, blockSize>>>
            (
              L[i-1].fc_y_dev,
              L[i-1].fc_nNeuron,
              L[i].fc_w_dev,
              L[i].fc_b_dev,
              L[i].fc_y_dev,
              L[i].fc_nNeuron,
              L[i].fActiv
            );
            break;

          case LT_CONVOLUTION:	//если слой свёрточный
            gpuFc<<<gridSize, blockSize>>>
            (
              L[i-1].cnn_y_dev,
              L[i-1].cnn_sMap*L[i-1].cnn_nCore,
              L[i].fc_w_dev,
              L[i].fc_b_dev,
              L[i].fc_y_dev,
              L[i].fc_nNeuron,
              L[i].fActiv
            );
            break;

          case LT_MAXPOOLING:	//если слой субдискретизирующий
            gpuFc<<<gridSize, blockSize>>>
            (
              L[i-1].mp_y_dev,
              L[i-1].mp_sMap*L[i-1].mp_nCore,
              L[i].fc_w_dev,
              L[i].fc_b_dev,
              L[i].fc_y_dev,
              L[i].fc_nNeuron,
              L[i].fActiv
            );
            break;
        }
#ifdef DEBUG_NETWORK	//для отладки сети
        CUDA_CHECK_ERROR(cudaMemcpy(L[i].fc_y, L[i].fc_y_dev, L[i].fc_nNeuron*sizeof(float), cudaMemcpyDeviceToHost));
#endif
        break;

      case LT_CONVOLUTION: //если слой свёрточный
        gridSize = dim3((L[i].cnn_sMapW + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, (L[i].cnn_sMapH + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1);
        blockSize = dim3(MIN_BLOCK_SIZE, MIN_BLOCK_SIZE, 1);
        switch(L[i-1].layerType) {	//проверяем тип предыдущего слоя
          case LT_CONVOLUTION: //если слой свёрточный
            gpuConvValid<<<gridSize, blockSize>>>
            (
              L[i-1].cnn_y_dev,
              L[i-1].cnn_sMapW,
              L[i-1].cnn_sMapH,
              L[i-1].cnn_nCore,
              L[i].cnn_k_dev,
              L[i].cnn_sCore,
              L[i].cnn_nCore,
              L[i].cnn_adj_dev,
              L[i].cnn_b_dev,
              L[i].cnn_y_dev,
              L[i].cnn_sMapW,
              L[i].cnn_sMapH,
              L[i].fActiv
            );
            break;

          case LT_MAXPOOLING:	//если слой субдискретизирующий
            gpuConvValid<<<gridSize, blockSize>>>
            (
              L[i-1].mp_y_dev,
              L[i-1].mp_sMapW,
              L[i-1].mp_sMapH,
              L[i-1].mp_nCore,
              L[i].cnn_k_dev,
              L[i].cnn_sCore,
              L[i].cnn_nCore,
              L[i].cnn_adj_dev,
              L[i].cnn_b_dev,
              L[i].cnn_y_dev,
              L[i].cnn_sMapW,
              L[i].cnn_sMapH,
              L[i].fActiv
            );
            break;
        }
#ifdef DEBUG_NETWORK	//для отладки сети
        CUDA_CHECK_ERROR(cudaMemcpy(L[i].cnn_y, L[i].cnn_y_dev, L[i].cnn_sMap*L[i].cnn_nCore*sizeof(float), cudaMemcpyDeviceToHost));
#endif
        break;

      case LT_MAXPOOLING:	//если слой субдискретизирующий
        gridSize = dim3((L[i].mp_sMapW + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, (L[i].mp_sMapH + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1);
        blockSize = dim3(MIN_BLOCK_SIZE, MIN_BLOCK_SIZE, 1);
        gpuMaxPool<<<gridSize, blockSize>>>
        (
          L[i-1].cnn_y_dev,
          L[i-1].cnn_sMapW,
          L[i-1].cnn_sMapH,
          L[i].mp_sCore,
          L[i].mp_y_dev,
          L[i].mp_sMapW,
          L[i].mp_sMapH,
          L[i].mp_nCore,
          L[i].mp_mask_dev
        );
#ifdef DEBUG_NETWORK	//для отладки сети
        CUDA_CHECK_ERROR(cudaMemcpy(L[i].mp_y, L[i].mp_y_dev, L[i].mp_sMap*L[i].mp_nCore*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaMemcpy(L[i].mp_mask, L[i].mp_mask_dev, L[i-1].cnn_sMapW*L[i-1].cnn_sMapH*L[i-1].cnn_nCore*sizeof(float), cudaMemcpyDeviceToHost));
#endif
        break;
    }
    //синхронизируем
    cudaEventRecord(syncEvent, 0);
    cudaEventSynchronize(syncEvent);
  }

  //рассчёт для последнего слоя
  gridSize = dim3((L[nLayer-1].fc_nNeuron + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1, 1);
  blockSize = dim3(MIN_BLOCK_SIZE, 1, 1);
  switch(L[nLayer-2].layerType) {	//проверяем тип предыдущего слоя
    case LT_FULLYCONNECTED: //если слой полносвязный
      gpuFc<<<gridSize, blockSize>>>
      (
        L[nLayer-2].fc_y_dev,
        L[nLayer-2].fc_nNeuron,
        L[nLayer-1].fc_w_dev,
        L[nLayer-1].fc_b_dev,
        outDataDev,
        L[nLayer-1].fc_nNeuron,
        L[nLayer-1].fActiv
      );
      break;

    case LT_CONVOLUTION:	//если слой свёрточный
      gpuFc<<<gridSize, blockSize>>>
      (
        L[nLayer-2].cnn_y_dev,
        L[nLayer-2].cnn_sMap*L[nLayer-2].cnn_nCore,
        L[nLayer-1].fc_w_dev,
        L[nLayer-1].fc_b_dev,
        outDataDev,
        L[nLayer-1].fc_nNeuron,
        L[nLayer-1].fActiv);
      break;

    case LT_MAXPOOLING:	//если слой субдискретизирующий
      gpuFc<<<gridSize, blockSize>>>
      (
      L[nLayer-2].mp_y_dev,
      L[nLayer-2].mp_sMap*L[nLayer-2].mp_nCore,
      L[nLayer-1].fc_w_dev,
      L[nLayer-1].fc_b_dev,
      outDataDev,
      L[nLayer-1].fc_nNeuron,
      L[nLayer-1].fActiv
      );
      break;
  }
  //синхронизируем
  cudaEventRecord(syncEvent, 0);
  cudaEventSynchronize(syncEvent);
  //копируем выходные данные
  CUDA_CHECK_ERROR(cudaMemcpy(outData, outDataDev, outDataSize*sizeof(float), cudaMemcpyDeviceToHost));
  //освобождаем ресурсы
  CUDA_CHECK_ERROR(cudaEventDestroy(syncEvent));

  return EIVE_GOOD;
}
