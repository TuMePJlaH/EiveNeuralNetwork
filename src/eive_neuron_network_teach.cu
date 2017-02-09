//Исходник класса EIVE нейронной сети v 1.0.0
//функция для обучения

#include "cuda_debug.h"
#include "eive_neuron_network.h"
#include "eive_neuron_network_device_func.h"
#include "iostream"

//функция обучения сети
int EiveNeuronNetwork::teachNetwork(float *inputData, float *neadOutput, float max_error, float *real_error)
{
  if(!L) {	//если сеть не создана
    std::cout << "[ERROR][F:teachNetwork]:the network is not created!" << std::endl;
    return EIVE_ERROR;
  }

  //делаем прямой проход
  colculateNetwork(inputData);

  //копируем в наш массив
  memcpy(neadOutData, neadOutput, outDataSize*sizeof(float));

  //рассчитываем ошибку
  *real_error = 0;
  for(int i = 0; i < outDataSize; i++) {
    *real_error += pow(neadOutput[i] - outData[i], 2);
  }
  //если ошибка меньше максимальной, то обучение не проводим
  if(*real_error < max_error) {
    std::cout << "[INFO]:training is not required" << std::endl;
    return EIVE_GOOD;
  }

  //копируем в устройство
  CUDA_CHECK_ERROR(cudaMemcpy(neadOutDataDev, neadOutData, outDataSize*sizeof(float), cudaMemcpyHostToDevice));

  //Локальные переменные для расчёта
  cudaEvent_t syncEvent;
  cudaEventCreate(&syncEvent);
  dim3 gridSize;
  dim3 blockSize;

  //рассчитываем для последнего слоя
  gridSize = dim3((L[nLayer-1].fc_nNeuron + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1, 1);
  blockSize = dim3(MIN_BLOCK_SIZE, 1, 1);
  gpuDeltaO<<<gridSize, blockSize>>>
  (
    neadOutDataDev,
    outDataDev,
    outDataSize,
    L[nLayer-1].fc_delta_dev,
    L[nLayer-1].fActiv
  );

#ifdef DEBUG_NETWORK	//для отладки сети
  CUDA_CHECK_ERROR(cudaMemcpy(L[nLayer-1].fc_delta, L[nLayer-1].fc_delta_dev, outDataSize*sizeof(float), cudaMemcpyDeviceToHost));
#endif

  //Идём обратно по всем слоям поочереди
  for(int i = nLayer-1; i > 0; i--) {
    switch(L[i].layerType) {
      case LT_FULLYCONNECTED:
        switch(L[i-1].layerType) {
          case LT_FULLYCONNECTED:
            gridSize = dim3((L[i-1].fc_nNeuron + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1, 1);
            blockSize = dim3(MIN_BLOCK_SIZE, 1, 1);
            gpuDeltaFC<<<gridSize, blockSize>>>
            (
              L[i].fc_delta_dev,
              L[i].fc_nNeuron,
              L[i].fc_w_dev,
              L[i-1].fc_delta_dev,
              L[i-1].fc_y_dev,
              L[i-1].fc_nNeuron,
              L[i-1].fActiv
            );
#ifdef DEBUG_NETWORK	//для отладки сети
            CUDA_CHECK_ERROR(cudaMemcpy(L[i-1].fc_delta, L[i-1].fc_delta_dev, L[i-1].fc_nNeuron*sizeof(float), cudaMemcpyDeviceToHost));
#endif
            break;

          case LT_CONVOLUTION:
            gridSize = dim3((L[i-1].cnn_sMap*L[i-1].cnn_nCore + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1, 1);
            blockSize = dim3(MIN_BLOCK_SIZE, 1, 1);
            gpuDeltaFC<<<gridSize, blockSize>>>
            (
              L[i].fc_delta_dev,
              L[i].fc_nNeuron,
              L[i].fc_w_dev,
              L[i-1].cnn_delta_dev,
              L[i-1].cnn_y_dev,
              L[i-1].cnn_sMap*L[i-1].cnn_nCore,
              L[i-1].fActiv
            );
#ifdef DEBUG_NETWORK	//для отладки сети
            CUDA_CHECK_ERROR(cudaMemcpy(L[i-1].cnn_delta, L[i-1].cnn_delta_dev, L[i-1].cnn_sMap*L[i-1].cnn_nCore*sizeof(float), cudaMemcpyDeviceToHost));
#endif
            break;

          case LT_MAXPOOLING:
            gridSize = dim3((L[i-1].mp_sMap*L[i-1].mp_nCore + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1, 1);
            blockSize = dim3(MIN_BLOCK_SIZE, 1, 1);
            gpuDeltaFC<<<gridSize, blockSize>>>
            (
              L[i].fc_delta_dev,
              L[i].fc_nNeuron,
              L[i].fc_w_dev,
              L[i-1].mp_delta_dev,
              L[i-1].mp_y_dev,
              L[i-1].mp_sMap*L[i-1].mp_nCore,
              L[i-2].fActiv
            );
#ifdef DEBUG_NETWORK	//для отладки сети
            CUDA_CHECK_ERROR(cudaMemcpy(L[i-1].mp_delta, L[i-1].mp_delta_dev, L[i-1].mp_sMap*L[i-1].mp_nCore*sizeof(float), cudaMemcpyDeviceToHost));
#endif
            break;
        }
        break;

      case LT_CONVOLUTION:
        switch(L[i-1].layerType) {
          case LT_CONVOLUTION:
            gridSize = dim3((L[i-1].cnn_sMapW + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, (L[i-1].cnn_sMapH + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1);
            blockSize = dim3(MIN_BLOCK_SIZE, MIN_BLOCK_SIZE, 1);
            gpuKorFull<<<gridSize, blockSize>>>
            (
              L[i].cnn_delta_dev,
              L[i].cnn_sMapW,
              L[i].cnn_sMapH,
              L[i].cnn_k_dev,
              L[i].cnn_sCore,
              L[i].cnn_nCore,
              L[i].cnn_adj_dev,
              L[i-1].cnn_y_dev,
              L[i-1].cnn_sMapW,
              L[i-1].cnn_sMapH,
              L[i-1].cnn_nCore,
              L[i-1].cnn_delta_dev,
              L[i-1].fActiv
            );
#ifdef DEBUG_NETWORK	//для отладки сети
            CUDA_CHECK_ERROR(cudaMemcpy(L[i-1].cnn_delta, L[i-1].cnn_delta_dev, L[i-1].cnn_sMap*L[i-1].cnn_nCore*sizeof(float), cudaMemcpyDeviceToHost));
#endif
            break;

          case LT_MAXPOOLING:
            gridSize = dim3((L[i-1].mp_sMapW + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, (L[i-1].mp_sMapH + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1);
            blockSize = dim3(MIN_BLOCK_SIZE, MIN_BLOCK_SIZE, 1);
            gpuKorFull<<<gridSize, blockSize>>>
            (
              L[i].cnn_delta_dev,
              L[i].cnn_sMapW,
              L[i].cnn_sMapH,
              L[i].cnn_k_dev,
              L[i].cnn_sCore,
              L[i].cnn_nCore,
              L[i].cnn_adj_dev,
              L[i-1].mp_y_dev,
              L[i-1].mp_sMapW,
              L[i-1].mp_sMapH,
              L[i-1].mp_nCore,
              L[i-1].mp_delta_dev,
              L[i-2].fActiv
            );
#ifdef DEBUG_NETWORK	//для отладки сети
            CUDA_CHECK_ERROR(cudaMemcpy(L[i-1].mp_delta, L[i-1].mp_delta_dev, L[i-1].mp_sMap*L[i-1].mp_nCore*sizeof(float), cudaMemcpyDeviceToHost));
#endif
            break;
        }
        break;
      case LT_MAXPOOLING:
        gridSize = dim3((L[i].mp_sMapW + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, (L[i].mp_sMapH + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1);
        blockSize = dim3(MIN_BLOCK_SIZE, MIN_BLOCK_SIZE, 1);
        gpuDeltaMP<<<gridSize, blockSize>>>
        (
          L[i].mp_delta_dev,
          L[i].mp_sMapW,
          L[i].mp_sMapH,
          L[i].mp_sCore,
          L[i].mp_nCore,
          L[i-1].cnn_delta_dev,
          L[i-1].cnn_sMapW,
          L[i-1].cnn_sMapH,
          L[i].mp_mask_dev
        );
#ifdef DEBUG_NETWORK	//для отладки сети
        CUDA_CHECK_ERROR(cudaMemcpy(L[i-1].cnn_delta, L[i-1].cnn_delta_dev, L[i-1].cnn_sMap*L[i-1].cnn_nCore*sizeof(float), cudaMemcpyDeviceToHost));
#endif
        break;
    }
    //синхронизируем
    cudaEventRecord(syncEvent, 0);
    cudaEventSynchronize(syncEvent);
  }

  //Проводим корректировку весов в прямом направлении для каждого слоя
  //Для первого слоя
  switch(L[0].layerType) {
    case LT_FULLYCONNECTED:	//если слой полносвязный
      gridSize = dim3((L[0].fc_nNeuron + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1, 1);
      blockSize = dim3(MIN_BLOCK_SIZE, 1, 1);
      gpyChangeWeightFC<<<gridSize,blockSize>>>
      (
        L[0].fc_delta_dev,
        L[0].fc_w_dev,
        L[0].fc_nNeuron,
        inDataDev,
        inDataSize,
        L[0].fc_b_dev,
        speedTeach
      );
#ifdef DEBUG_NETWORK	//для отладки сети
      CUDA_CHECK_ERROR(cudaMemcpy(L[0].fc_w, L[0].fc_w_dev, L[0].wkSize*sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK_ERROR(cudaMemcpy(L[0].fc_b, L[0].fc_b_dev, L[0].fc_nNeuron*sizeof(float), cudaMemcpyDeviceToHost));
#endif
      break;

    case LT_CONVOLUTION:	//если слой свёрточный
      gridSize = dim3((L[0].cnn_nCore + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1, 1);
      blockSize = dim3(MIN_BLOCK_SIZE, 1, 1);
      gpuChangeWeightCNNI<<<gridSize, blockSize>>>
      (
        L[0].cnn_delta_dev,
        L[0].cnn_sMapW,
        L[0].cnn_sMapH,
        inDataDev,
        inDataW,
        inDataH,
        L[0].cnn_k_dev,
        L[0].cnn_sCore,
        L[0].cnn_nCore,
        L[0].cnn_b_dev,
        speedTeach
      );
#ifdef DEBUG_NETWORK	//для отладки сети
      CUDA_CHECK_ERROR(cudaMemcpy(L[0].cnn_k, L[0].cnn_k_dev, L[0].cnn_sCore*L[0].cnn_sCore*L[0].cnn_nCore*sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK_ERROR(cudaMemcpy(L[0].cnn_b, L[0].cnn_b_dev, L[0].cnn_nCore*sizeof(float), cudaMemcpyDeviceToHost));
#endif
      break;
  }

  //синхронизируем
  cudaEventRecord(syncEvent, 0);
  cudaEventSynchronize(syncEvent);

  //и бежим по всем слоям
  for(int i = 1; i < nLayer; i++)	{
    switch(L[i].layerType) {	//проверяем тип слоя
      case LT_FULLYCONNECTED:	//если слой полносвязный
        gridSize = dim3((L[i].fc_nNeuron + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1, 1);
        blockSize = dim3(MIN_BLOCK_SIZE, 1, 1);
        switch(L[i-1].layerType) {	//проверяем тип предыдущего слоя
          case LT_FULLYCONNECTED: //если слой полносвязный
            gpyChangeWeightFC<<<gridSize,blockSize>>>
            (
              L[i].fc_delta_dev,
              L[i].fc_w_dev,
              L[i].fc_nNeuron,
              L[i-1].fc_y_dev,
              L[i-1].fc_nNeuron,
              L[i].fc_b_dev,
              speedTeach
            );
            break;

          case LT_CONVOLUTION:	//если слой свёрточный
            gpyChangeWeightFC<<<gridSize,blockSize>>>
            (
              L[i].fc_delta_dev,
              L[i].fc_w_dev,
              L[i].fc_nNeuron,
              L[i-1].cnn_y_dev,
              L[i-1].cnn_nCore*L[i-1].cnn_sMap,
              L[i].fc_b_dev,
              speedTeach
            );
            break;

          case LT_MAXPOOLING:	//если слой субдискретизирующий
            gpyChangeWeightFC<<<gridSize,blockSize>>>
            (
              L[i].fc_delta_dev,
              L[i].fc_w_dev,
              L[i].fc_nNeuron,
              L[i-1].mp_y_dev,
              L[i-1].mp_nCore*L[i-1].mp_sMap,
              L[i].fc_b_dev,
              speedTeach
            );
            break;
        }
#ifdef DEBUG_NETWORK	//для отладки сети
        CUDA_CHECK_ERROR(cudaMemcpy(L[i].fc_w, L[i].fc_w_dev, L[i].wkSize*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaMemcpy(L[i].fc_b, L[i].fc_b_dev, L[i].fc_nNeuron*sizeof(float), cudaMemcpyDeviceToHost));
#endif
        break;

      case LT_CONVOLUTION: //если слой свёрточный
        gridSize = dim3((L[i].cnn_nCore + MIN_BLOCK_SIZE - 1)/MIN_BLOCK_SIZE, 1, 1);
        blockSize = dim3(MIN_BLOCK_SIZE, 1, 1);
        switch(L[i-1].layerType) {	//проверяем тип предыдущего слоя
          case LT_CONVOLUTION: //если слой свёрточный
            gpuChangeWeightCNN<<<gridSize, blockSize>>>
            (
              L[i].cnn_delta_dev,
              L[i].cnn_sMapW,
              L[i].cnn_sMapH,
              L[i-1].cnn_y_dev,
              L[i-1].cnn_sMapW,
              L[i-1].cnn_sMapH,
              L[i-1].cnn_nCore,
              L[i].cnn_k_dev,
              L[i].cnn_sCore,
              L[i].cnn_nCore,
              L[i].cnn_adj_dev,
              L[i].cnn_b_dev,
              speedTeach
            );
            break;

          case LT_MAXPOOLING:	//если слой субдискретизирующий
            gpuChangeWeightCNN<<<gridSize, blockSize>>>
            (
              L[i].cnn_delta_dev,
              L[i].cnn_sMapW,
              L[i].cnn_sMapH,
              L[i-1].mp_y_dev,
              L[i-1].mp_sMapW,
              L[i-1].mp_sMapH,
              L[i-1].mp_nCore,
              L[i].cnn_k_dev,
              L[i].cnn_sCore,
              L[i].cnn_nCore,
              L[i].cnn_adj_dev,
              L[i].cnn_b_dev,
              speedTeach
            );
            break;
        }
#ifdef DEBUG_NETWORK	//для отладки сети
        CUDA_CHECK_ERROR(cudaMemcpy(L[i].cnn_k, L[i].cnn_k_dev, L[i].cnn_sCore*L[0].cnn_sCore*L[i].cnn_nCore*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaMemcpy(L[i].cnn_b, L[i].cnn_b_dev, L[i].cnn_nCore*sizeof(float), cudaMemcpyDeviceToHost));
#endif
        break;

      case LT_MAXPOOLING:	//если слой субдискретизирующий
        //ничего не корректируем
        break;
    }
    //синхронезируем
    cudaEventRecord(syncEvent, 0);
    cudaEventSynchronize(syncEvent);
  }
  CUDA_CHECK_ERROR(cudaEventDestroy(syncEvent));

  return EIVE_GOOD;
}
