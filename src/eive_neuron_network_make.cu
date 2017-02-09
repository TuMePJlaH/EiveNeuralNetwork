//Исходник класса EIVE нейронной сети v 1.0.0
//Функции создания сети, сохранения и загрузки
#include "cuda_debug.h"
#include "eive_neuron_network.h"
#include "iostream"
#include "fstream"
#include <cstring>

//----------------------------------------------------------------------------------------------------------
//конструктор пустой сети
EiveNeuronNetwork::EiveNeuronNetwork()
{
  nLayer = 0;
  inDataW = 0;
  inDataH = 0;
  inData = nullptr;
  inDataDev = nullptr;
  outDataSize = 0;
  outData = nullptr;
  outDataDev = nullptr;
  L = nullptr;
  speedTeach = 0;
  std::cout << "[INFO]:Empty network created" << std::endl;
  return;
}
//----------------------------------------------------------------------------------------------------------
EiveNeuronNetwork::~EiveNeuronNetwork()
{
  if(L) {
    for(int i = 0; i < nLayer; i++) {
      switch(L[i].layerType) {
        case LT_FULLYCONNECTED:
          if(L[i].fc_w) { delete[] L[i].fc_w; }
          if(L[i].fc_w_dev) { CUDA_CHECK_ERROR(cudaFree(L[i].fc_w_dev)); }
          if(L[i].fc_y) { delete[] L[i].fc_y; }
          if(L[i].fc_y_dev) { CUDA_CHECK_ERROR(cudaFree(L[i].fc_y_dev)); }
          if(L[i].fc_delta) { delete[] L[i].fc_delta; }
          if(L[i].fc_delta_dev) { CUDA_CHECK_ERROR(cudaFree(L[i].fc_delta_dev)); }
          if(L[i].fc_b) { delete[] L[i].fc_b; }
          if(L[i].fc_b_dev) { CUDA_CHECK_ERROR(cudaFree(L[i].fc_b_dev)); }
          break;

        case LT_CONVOLUTION:
          if(L[i].cnn_k) { delete[] L[i].cnn_k; }
          if(L[i].cnn_k_dev) { CUDA_CHECK_ERROR(cudaFree(L[i].cnn_k_dev)); }
          if(L[i].cnn_y) { delete[] L[i].cnn_y; }
          if(L[i].cnn_y_dev) { CUDA_CHECK_ERROR(cudaFree(L[i].cnn_y_dev)); }
          if(L[i].cnn_delta) { delete[] L[i].cnn_delta; }
          if(L[i].cnn_delta_dev) { CUDA_CHECK_ERROR(cudaFree(L[i].cnn_delta_dev)); }
          if(L[i].cnn_adj) { delete[] L[i].cnn_adj; }
          if(L[i].cnn_adj_dev) { CUDA_CHECK_ERROR(cudaFree(L[i].cnn_adj_dev)); }
          if(L[i].cnn_b) { delete[] L[i].cnn_b; }
          if(L[i].cnn_b_dev) { CUDA_CHECK_ERROR(cudaFree(L[i].cnn_b_dev)); }
          break;

        case LT_MAXPOOLING:
          if(L[i].mp_y) { delete[] L[i].mp_y; }
          if(L[i].mp_y_dev) { CUDA_CHECK_ERROR(cudaFree(L[i].mp_y_dev)); }
          if(L[i].mp_delta) { delete[] L[i].mp_delta; }
          if(L[i].mp_delta_dev) { CUDA_CHECK_ERROR(cudaFree(L[i].mp_delta_dev)); }
          if(L[i].mp_mask) { delete[] L[i].mp_mask; }
          if(L[i].mp_mask_dev) { CUDA_CHECK_ERROR(cudaFree(L[i].mp_mask_dev)); }
          break;
      }
    }
    delete[] L;
  }
  if(inData) { delete[] inData; }
  if(inDataDev) { CUDA_CHECK_ERROR(cudaFree(inDataDev)); }
  if(outData) { delete[] outData; }
  if(outDataDev) { CUDA_CHECK_ERROR(cudaFree(outDataDev)); }
  if(neadOutData) { delete[] neadOutData; }
  if(neadOutDataDev) { CUDA_CHECK_ERROR(cudaFree(neadOutDataDev)); }
  std::cout << "[INFO]:the network is destroyed" << std::endl;
  return;
}
//----------------------------------------------------------------------------------------------------------
int EiveNeuronNetwork::setNLayer(int nLayer)
{
  if(nLayer <= 0) {
    std::cout << "[ERROR][F:setNLayer]:nLayer is not correct!" << std::endl;
    return EIVE_ERROR;
  }
  this->nLayer = nLayer;
  return EIVE_GOOD;
}
//----------------------------------------------------------------------------------------------------------
int EiveNeuronNetwork::setInDataW(int inDataW)
{
  if(inDataW <= 0) {
    std::cout << "[ERROR][F:setInDataW]:inDataW is not correct!" << std::endl;
    return EIVE_ERROR;
  }
  this->inDataW = inDataW;
  return EIVE_GOOD;
}
//----------------------------------------------------------------------------------------------------------
int EiveNeuronNetwork::setInDataH(int inDataH)
{
  if(inDataH <= 0) {
    std::cout << "[ERROR][F:setInDataH]:inDataH is not correct!" << std::endl;
    return EIVE_ERROR;
  }
  this->inDataH = inDataH;
  return EIVE_GOOD;
}
//----------------------------------------------------------------------------------------------------------
int EiveNeuronNetwork::setSpeedTeach(float speedTeach)
{
  if(speedTeach <= 0) {
    std::cout << "[ERROR][F:setInDataH]:setSpeedTeach is not correct!" << std::endl;
    return EIVE_ERROR;
  }
  this->speedTeach = speedTeach;
  return EIVE_GOOD;
}
//----------------------------------------------------------------------------------------------------------
//функция создания сети
int EiveNeuronNetwork::createNetwork()
{
  if(!nLayer) {				//если количество слоёв не задано
    std::cout << "[ERROR][F:createNetwork]:nLayer is not correct!" << std::endl;
    return EIVE_ERROR;
  }

  if(!inDataW || !inDataH) {	//задан ли размер входных данных
    std::cout << "[ERROR][F:createNetwork]:inDataW or inDataH are not correct!" << std::endl;
    return EIVE_ERROR;
  }

  L = new EiveNeuronLayer [nLayer];	//создаём слои
  inDataSize = inDataW*inDataH;
  inData = new float [inDataSize];
  CUDA_CHECK_ERROR(cudaMalloc((void**)&inDataDev, inDataSize*sizeof(float)));
  std::cout << "[INFO]:Create Network with " << nLayer << " layer" << std::endl;
  std::cout << "[INFO]:Input data width	 = " << inDataW << std::endl;
  std::cout << "[INFO]:           height = " << inDataH << std::endl;
  return EIVE_GOOD;
}
//----------------------------------------------------------------------------------------------------------
//функция создания свёрточного слоя
int EiveNeuronNetwork::createCNNLayer(int layerNumber, int numberOfCortex, int cortexSize, int activatFunction)
{
  if(!L) {	//если сеть не создана
    std::cout << "[ERROR][F:createCNNLayer]:the network is not created!" << std::endl;
    return EIVE_ERROR;
  }

  if((layerNumber < 0) ||	(layerNumber >= nLayer)) {	//если номер слоя некорректен
    std::cout << "[ERROR][F:createCNNLayer]:layerNumber is not correct!" << std::endl;
    return EIVE_ERROR;
  }

  if(layerNumber == nLayer-1) {	//проверяем не последний ли это слой
    std::cout << "[ERROR][F:createCNNLayer]:Convolution layer can't be the last!" << std::endl;
    return EIVE_ERROR;
  }

  if((layerNumber) && //если слой не первый, а предыдущей слой не свёрточный или субдискретизирующий
  ((L[layerNumber-1].layerType != LT_CONVOLUTION) && (L[layerNumber-1].layerType != LT_MAXPOOLING)))	{
    std::cout << "[ERROR][F:createCNNLayer]:previous layer is not correct!" << std::endl;
    return EIVE_ERROR;
  }

  if((numberOfCortex <= 0) || (cortexSize <= 0)) {	//проверяем на корректность входные данные
    std::cout << "[ERROR][F:createCNNLayer]:input data is not correct!" << std::endl;
    return EIVE_ERROR;
  }

  if((activatFunction < 0) &&	(activatFunction > 1)) {	//проверяем корректность функции активации
    std::cout << "[ERROR][F:createCNNLayer]:activatFunction is not supported!" << std::endl;
    return EIVE_ERROR;
  }

  int mapH = 0;
  int mapW = 0;
  int adjS = 0;

  if(!layerNumber) {	//для случая если этот слой первый
    if((cortexSize > inDataW) || (cortexSize > inDataH)) {	//если размер ядра свёртки больше изображения
      std::cout << "[ERROR][F:createCNNLayer]:cortexSize is not correct!" << std::endl;
      return EIVE_ERROR;
    }

    //рассчитываем размеры карты признаков
    mapW = inDataW-cortexSize + 1;
    mapH = inDataH-cortexSize + 1;
  }
  else {	//Для случая если слой не первый
    switch(L[layerNumber-1].layerType) {	//проверяем тип предыдущего слоя
      case LT_CONVOLUTION:	//если предыдущей слой свёрточный
        if((cortexSize > L[layerNumber-1].cnn_sMapW) ||	//если размер ядра свёртки больше карты признаков
        (cortexSize > L[layerNumber-1].cnn_sMapH)) {		//предыдущего слоя
          std::cout << "[ERROR][F:createCNNLayer]:cortexSize is not correct!" << std::endl;
          return EIVE_ERROR;
        }

        //рассчитываем размеры карты признаков
        mapW = L[layerNumber-1].cnn_sMapW-cortexSize + 1;
        mapH = L[layerNumber-1].cnn_sMapH-cortexSize + 1;
        //размер матрицы смежности
        adjS = numberOfCortex*L[layerNumber-1].cnn_nCore;
        break;

      case LT_MAXPOOLING:		//если предыдущей слой субдискретизирующий
        if((cortexSize > L[layerNumber-1].mp_sMapW) ||	//если размер ядра свёртки больше карты признаков
        (cortexSize > L[layerNumber-1].mp_sMapH)) {     //предыдущего слоя
          std::cout << "[ERROR][F:createCNNLayer]:cortexSize is not correct!" << std::endl;
          return EIVE_ERROR;
        }

        //рассчитываем размеры карты признаков
        mapW = L[layerNumber-1].mp_sMapW-cortexSize + 1;
        mapH = L[layerNumber-1].mp_sMapH-cortexSize + 1;
        //размер матрицы смежности
        adjS = numberOfCortex*L[layerNumber-1].mp_nCore;
        break;
    }
  }
  //если всё хорошо записываем результаты в слой
  L[layerNumber].cnn_sMapW = mapW;
  L[layerNumber].cnn_sMapH = mapH;
  L[layerNumber].cnn_adjS = adjS;
  L[layerNumber].cnn_nCore = numberOfCortex;
  L[layerNumber].cnn_sCore = cortexSize;
  L[layerNumber].layerNumber = layerNumber;
  L[layerNumber].cnn_sMap = L[layerNumber].cnn_sMapW*L[layerNumber].cnn_sMapH;
  L[layerNumber].fActiv = activatFunction;
  L[layerNumber].layerType = LT_CONVOLUTION;
  L[layerNumber].wkSize = L[layerNumber].cnn_nCore*L[layerNumber].cnn_sCore*L[layerNumber].cnn_sCore;
  //и выделяем под это всё дело память
  L[layerNumber].cnn_k = new float [L[layerNumber].wkSize];
  CUDA_CHECK_ERROR(cudaMalloc((void**)&L[layerNumber].cnn_k_dev, L[layerNumber].wkSize*sizeof(float)));
  L[layerNumber].cnn_y = new float [L[layerNumber].cnn_nCore*L[layerNumber].cnn_sMap];
  CUDA_CHECK_ERROR(cudaMalloc((void**)&L[layerNumber].cnn_y_dev, L[layerNumber].cnn_nCore*L[layerNumber].cnn_sMap*sizeof(float)));
  L[layerNumber].cnn_delta = new float [L[layerNumber].cnn_nCore*L[layerNumber].cnn_sMap];
  CUDA_CHECK_ERROR(cudaMalloc((void**)&L[layerNumber].cnn_delta_dev, L[layerNumber].cnn_nCore*L[layerNumber].cnn_sMap*sizeof(float)));
  L[layerNumber].cnn_b = new float [L[layerNumber].cnn_nCore];
  CUDA_CHECK_ERROR(cudaMalloc((void**)&L[layerNumber].cnn_b_dev, L[layerNumber].cnn_nCore*sizeof(float)));
  if(L[layerNumber].cnn_adjS > 0)	{
    L[layerNumber].cnn_adj = new float [L[layerNumber].cnn_adjS];
    CUDA_CHECK_ERROR(cudaMalloc((void**)&L[layerNumber].cnn_adj_dev, L[layerNumber].cnn_adjS*sizeof(float)));
  }

  //Заполняем необходимые массивы
  //массив весовых коэффициентов
  float v1 = pow((float)10, RAND_THRESHOLD);
  float v2 = (MAX_W_RAND-(MIN_W_RAND))*v1;
  float v3 = MIN_W_RAND * v1;
  for(int i = 0; i < L[layerNumber].wkSize; i++) {
    L[layerNumber].cnn_k[i] = ((float)(rand()%(int)v2 + v3))/v1;
  }
  CUDA_CHECK_ERROR(cudaMemcpy(L[layerNumber].cnn_k_dev, L[layerNumber].cnn_k, L[layerNumber].wkSize*sizeof(float), cudaMemcpyHostToDevice));
  //массив коэффициентов сдвига
  for(int i = 0; i < L[layerNumber].cnn_nCore; i++) {
    L[layerNumber].cnn_b[i] = ((float)(rand()%(int)v2 + v3))/v1;
  }
  CUDA_CHECK_ERROR(cudaMemcpy(L[layerNumber].cnn_b_dev, L[layerNumber].cnn_b, L[layerNumber].cnn_nCore*sizeof(float), cudaMemcpyHostToDevice));
  //матрица смежности
  if(L[layerNumber].cnn_adjS > 0)	{
    for(int i = 0; i < L[layerNumber].cnn_adjS; i++) {
      L[layerNumber].cnn_adj[i] = 1;
    }
    CUDA_CHECK_ERROR(cudaMemcpy(L[layerNumber].cnn_adj_dev, L[layerNumber].cnn_adj, L[layerNumber].cnn_adjS*sizeof(float), cudaMemcpyHostToDevice));
  }

  //выводим информацию на экран
  std::cout << "[INFO]:Create Convolution layer:" << std::endl;
  std::cout << "[INFO]:-number of a layer = " << L[layerNumber].layerNumber << std::endl;
  std::cout << "[INFO]:-number of cores = " << L[layerNumber].cnn_nCore << std::endl;
  std::cout << "[INFO]:-core size = " << L[layerNumber].cnn_sCore << "x" << L[layerNumber].cnn_sCore << std::endl;
  return EIVE_GOOD;
}
//----------------------------------------------------------------------------------------------------------
//функция создания полносвязного слоя
int EiveNeuronNetwork::createFCLayer(int layerNumber, int numberOfNeuron, int activatFunction)
{
  if(!L) {	//если сеть не создана
    std::cout << "[ERROR][F:createFCLayer]:the network is not created!" << std::endl;
    return EIVE_ERROR;
  }

  if((layerNumber < 0) ||	(layerNumber >= nLayer)) {	//если номер слоя некорректен
    std::cout << "[ERROR][F:createFCLayer]:layerNumber is not correct!" << std::endl;
    return EIVE_ERROR;
  }

  if(numberOfNeuron <= 0) {	//проверяем на корректность входные данные
    std::cout << "[ERROR][F:createCNNLayer]:numberOfNeuron is not correct!" << std::endl;
    return EIVE_ERROR;
  }

  if((activatFunction < 0) &&	(activatFunction > 1)) {	//проверяем корректность функции активации
    std::cout << "[ERROR][F:createCNNLayer]:activatFunction is not supported!" << std::endl;
    return EIVE_ERROR;
  }

  L[layerNumber].layerType = LT_FULLYCONNECTED;	//тип слоя
  L[layerNumber].fActiv = activatFunction;			//активационная функция
  L[layerNumber].layerNumber = layerNumber;			//номер слоя
  L[layerNumber].fc_nNeuron = numberOfNeuron;		//запоминаем количество нейронов

  if(!layerNumber) {	//если слой первый
    //Выделяем память под весовые коэффициенты
    L[layerNumber].wkSize = inDataSize*L[layerNumber].fc_nNeuron;
    L[layerNumber].fc_w = new float [L[layerNumber].wkSize];
    CUDA_CHECK_ERROR(cudaMalloc((void**)&L[layerNumber].fc_w_dev, L[layerNumber].wkSize*sizeof(float)));
  }
  else {
    switch(L[layerNumber-1].layerType) {
      case LT_FULLYCONNECTED:
        //Выделяем память под весовые коэффициенты
        L[layerNumber].wkSize = L[layerNumber-1].fc_nNeuron*L[layerNumber].fc_nNeuron;
        L[layerNumber].fc_w = new float [L[layerNumber].wkSize];
        CUDA_CHECK_ERROR(cudaMalloc((void**)&L[layerNumber].fc_w_dev, L[layerNumber].wkSize*sizeof(float)));
        break;

      case LT_CONVOLUTION:
        //Выделяем память под весовые коэффициенты
        L[layerNumber].wkSize = L[layerNumber-1].cnn_nCore*L[layerNumber-1].cnn_sMap*L[layerNumber].fc_nNeuron;
        L[layerNumber].fc_w = new float [L[layerNumber].wkSize];
        CUDA_CHECK_ERROR(cudaMalloc((void**)&L[layerNumber].fc_w_dev, L[layerNumber].wkSize*sizeof(float)));
        break;

      case LT_MAXPOOLING:
        //Выделяем память под весовые коэффициенты
        L[layerNumber].wkSize = L[layerNumber-1].mp_nCore*L[layerNumber-1].mp_sMap*L[layerNumber].fc_nNeuron;
        L[layerNumber].fc_w = new float [L[layerNumber].wkSize];
        CUDA_CHECK_ERROR(cudaMalloc((void**)&L[layerNumber].fc_w_dev, L[layerNumber].wkSize*sizeof(float)));
        break;
    }
  }
  //Заполняем массив весовых коэффициентов случайными величинами
  //переменные для расчёта весовых коэффициентов
  float v1 = pow((float)10, RAND_THRESHOLD);
  float v2 = (MAX_W_RAND-(MIN_W_RAND))*v1;
  float v3 = MIN_W_RAND * v1;
  for(int i = 0; i < L[layerNumber].wkSize; i++) {
    L[layerNumber].fc_w[i] = ((float)(rand()%(int)v2 + v3))/v1;
  }
  //И копируем в память видеокарты
  CUDA_CHECK_ERROR(cudaMemcpy(L[layerNumber].fc_w_dev, L[layerNumber].fc_w, L[layerNumber].wkSize*sizeof(float), cudaMemcpyHostToDevice));
  //и коэффициенты сдвига
  L[layerNumber].fc_b = new float [L[layerNumber].fc_nNeuron];
  CUDA_CHECK_ERROR(cudaMalloc((void**)&L[layerNumber].fc_b_dev, L[layerNumber].fc_nNeuron*sizeof(float)));
  for(int i = 0; i < L[layerNumber].fc_nNeuron; i++) {
    L[layerNumber].fc_b[i] = ((float)(rand()%(int)v2 + v3))/v1;
  }
  //И копируем в память видеокарты
  CUDA_CHECK_ERROR(cudaMemcpy(L[layerNumber].fc_b_dev, L[layerNumber].fc_b, L[layerNumber].fc_nNeuron*sizeof(float), cudaMemcpyHostToDevice));

  L[layerNumber].fc_y = new float [L[layerNumber].fc_nNeuron];
  CUDA_CHECK_ERROR(cudaMalloc((void**)&L[layerNumber].fc_y_dev, L[layerNumber].fc_nNeuron*sizeof(float)));
  L[layerNumber].fc_delta = new float[L[layerNumber].fc_nNeuron];
  CUDA_CHECK_ERROR(cudaMalloc((void**)&L[layerNumber].fc_delta_dev, L[layerNumber].fc_nNeuron*sizeof(float)));
  std::cout << "[INFO]:Create Fullyconnected layer:" << std::endl;
  std::cout << "[INFO]:-number of neurons = " << L[layerNumber].fc_nNeuron << std::endl;
  if(layerNumber == nLayer-1) {
    outDataSize = numberOfNeuron;
    outData = new float [outDataSize];
    CUDA_CHECK_ERROR(cudaMalloc((void**)&outDataDev, outDataSize*sizeof(float)));
    neadOutData = new float [outDataSize];
    CUDA_CHECK_ERROR(cudaMalloc((void**)&neadOutDataDev, outDataSize*sizeof(float)));
    std::cout << "[INFO]:This layer the last" << std::endl;
  }
  return EIVE_GOOD;
}
//----------------------------------------------------------------------------------------------------------
//функцяи создания субдискретизируещего слоя
int EiveNeuronNetwork::createMPLayer(int layerNumber, int cortexSize)
{
  if(!L) {	//если сеть не создана
    std::cout << "[ERROR][F:createMPLayer]:the network is not created!" << std::endl;
    return EIVE_ERROR;
  }

  if((layerNumber < 0) ||	(layerNumber >= nLayer)) {	//если номер слоя некорректен
    std::cout << "[ERROR][F:createMPLayer]:layerNumber is not correct!" << std::endl;
    return EIVE_ERROR;
  }

  if(layerNumber == nLayer-1) {	//проверяем не последний ли это слой
    std::cout << "[ERROR][F:createCNNLayer]:Maxpooling layer can't be the last!" << std::endl;
    return EIVE_ERROR;
  }

  if((!layerNumber) ||			//проверяем, что предыдущей слой должен быть обязательно свёрточным
  (L[layerNumber-1].layerType != LT_CONVOLUTION)) {
    std::cout << "[ERROR][F:createMPLayer]:previous layer is not correct!" << std::endl;
    return EIVE_ERROR;
  }

  if((cortexSize < 2) ||	//проверяем на корректность входные данные
  (L[layerNumber-1].cnn_sMapW%cortexSize) ||
  (L[layerNumber-1].cnn_sMapH%cortexSize))	{
    std::cout << "[ERROR][F:createMPLayer]:cortexSize is not correct!" << std::endl;
    return EIVE_ERROR;
  }

  //Создаём субдискретизирующий слой
  L[layerNumber].layerType = LT_MAXPOOLING;
  L[layerNumber].layerNumber = layerNumber;
  L[layerNumber].mp_sCore = cortexSize;
  L[layerNumber].mp_nCore = L[layerNumber-1].cnn_nCore;
  L[layerNumber].mp_sMapW = L[layerNumber-1].cnn_sMapW/L[layerNumber].mp_sCore;
  L[layerNumber].mp_sMapH = L[layerNumber-1].cnn_sMapH/L[layerNumber].mp_sCore;
  L[layerNumber].mp_sMap = L[layerNumber].mp_sMapW*L[layerNumber].mp_sMapH;
  L[layerNumber].mp_y = new float [L[layerNumber].mp_nCore*L[layerNumber].mp_sMap];
  CUDA_CHECK_ERROR(cudaMalloc((void**)&L[layerNumber].mp_y_dev, L[layerNumber].mp_nCore*L[layerNumber].mp_sMap*sizeof(float)));
  L[layerNumber].mp_delta = new float[L[layerNumber].mp_nCore*L[layerNumber].mp_sMap];
  CUDA_CHECK_ERROR(cudaMalloc((void**)&L[layerNumber].mp_delta_dev, L[layerNumber].mp_nCore*L[layerNumber].mp_sMap*sizeof(float)));
  L[layerNumber].mp_mask = new float [L[layerNumber-1].cnn_sMapW*L[layerNumber-1].cnn_sMapH*L[layerNumber-1].cnn_nCore];
  CUDA_CHECK_ERROR(cudaMalloc((void**)&L[layerNumber].mp_mask_dev, L[layerNumber-1].cnn_sMapW*L[layerNumber-1].cnn_sMapH*L[layerNumber-1].cnn_nCore*sizeof(float)));
  std::cout << "[INFO]:Create Maxpooling layer:" << std::endl;
  std::cout << "[INFO]:-core size = " << L[layerNumber].mp_sCore << "x" << L[layerNumber].mp_sCore << std::endl;
  return EIVE_GOOD;
}
//----------------------------------------------------------------------------------------------------------
//функция установки весовых коэффициентов
int EiveNeuronNetwork::setWeight(int layerNumber, float *w)
{
  if(!L) {	//если сеть не создана
    std::cout << "[ERROR][F:setWeight]:the network is not created!" << std::endl;
    return EIVE_ERROR;
  }

  if((layerNumber < 0) ||	(layerNumber >= nLayer)) {	//если номер слоя некорректен
    std::cout << "[ERROR][F:setWeight]:layerNumber is not correct!" << std::endl;
    return EIVE_ERROR;
  }

  if(L[layerNumber].layerType == -1) {	//если слой ещё не сконфигурирован
    std::cout << "[ERROR][F:setWeight]:the layer is not created!" << std::endl;
    return EIVE_ERROR;
  }

  if(L[layerNumber].layerType == LT_MAXPOOLING) {	//если слой не имеет весовых коэффициентов
    std::cout << "[ERROR][F:setWeight]:layer has no weight factors!" << std::endl;
    return EIVE_ERROR;
  }

  //копируем коэффициенты в слой
  switch(L[layerNumber].layerType) {
    case LT_CONVOLUTION:
      memcpy(L[layerNumber].cnn_k, w, L[layerNumber].wkSize*sizeof(float));
      CUDA_CHECK_ERROR(cudaMemcpy(L[layerNumber].cnn_k_dev, L[layerNumber].cnn_k, L[layerNumber].wkSize*sizeof(float), cudaMemcpyHostToDevice));
      break;

    case LT_FULLYCONNECTED:
      memcpy(L[layerNumber].fc_w, w, L[layerNumber].wkSize*sizeof(float));
      CUDA_CHECK_ERROR(cudaMemcpy(L[layerNumber].fc_w_dev, L[layerNumber].fc_w, L[layerNumber].wkSize*sizeof(float), cudaMemcpyHostToDevice));
      break;
  }
  std::cout << "[INFO]:weight factors are set!" << std::endl;
  return EIVE_GOOD;
}
//----------------------------------------------------------------------------------------------------------
//функция установки коэффициентов сдвига
int EiveNeuronNetwork::setB(int layerNumber, float *b)
{
  if(!L) {	//если сеть не создана
    std::cout << "[ERROR][F:setB]:the network is not createde!" << std::endl;
    return EIVE_ERROR;
  }

  if((layerNumber < 0) ||	(layerNumber >= nLayer)) {	//если номер слоя некорректен
    std::cout << "[ERROR][F:setB]:layerNumber is not correct!" << std::endl;
    return EIVE_ERROR;
  }

  if(L[layerNumber].layerType == -1) {	//если слой ещё не сконфигурирован
    std::cout << "[ERROR][F:setB]:the layer is not created!" << std::endl;
    return EIVE_ERROR;
  }

  if(L[layerNumber].layerType == LT_MAXPOOLING) {	//если слой не имеет коэффициентов сдвига
    std::cout << "[ERROR][F:setB]:layer has no weight factors!" << std::endl;
    return EIVE_ERROR;
  }

  //копируем коэффициенты в нужный слой
  switch(L[layerNumber].layerType) {
    case LT_CONVOLUTION:
      memcpy(L[layerNumber].cnn_b, b, L[layerNumber].cnn_nCore*sizeof(float));
      CUDA_CHECK_ERROR(cudaMemcpy(L[layerNumber].cnn_b_dev, L[layerNumber].cnn_b, L[layerNumber].cnn_nCore*sizeof(float), cudaMemcpyHostToDevice));
      break;

    case LT_FULLYCONNECTED:
      memcpy(L[layerNumber].fc_b, b, L[layerNumber].fc_nNeuron*sizeof(float));
      CUDA_CHECK_ERROR(cudaMemcpy(L[layerNumber].fc_b_dev, L[layerNumber].fc_b, L[layerNumber].fc_nNeuron*sizeof(float), cudaMemcpyHostToDevice));
      break;
  }
  std::cout << "[INFO]:b factors are set!" << std::endl;
  return EIVE_GOOD;
}
//----------------------------------------------------------------------------------------------------------
//функция установки матрицы смежности
int EiveNeuronNetwork::setAdjM(int layerNumber, float *adjM)
{
  if(!L) {	//если сеть не создана
    std::cout << "[ERROR][F:setAdjM]:the network is not created!" << std::endl;
    return EIVE_ERROR;
  }

  if((layerNumber < 0) ||	(layerNumber >= nLayer)) {	//если номер слоя некорректен
    std::cout << "[ERROR][F:setAdjM]:layerNumber is not correct!" << std::endl;
    return EIVE_ERROR;
  }

  if(L[layerNumber].layerType != LT_CONVOLUTION) {	//если слой не свёрточный
    std::cout << "[ERROR][F:setAdjM]:layer not convolution!" << std::endl;
    return EIVE_ERROR;
  }

  if(!L[layerNumber].cnn_adjS) {	//если слой ещё не сконфигурирован
    std::cout << "[ERROR][F:setAdjM]:adjacency matrix not found!" << std::endl;
    return EIVE_ERROR;
  }

  //копируем матрицу смежности
  memcpy(L[layerNumber].cnn_adj, adjM, L[layerNumber].cnn_adjS*sizeof(float));
  //и записываем её в память
  CUDA_CHECK_ERROR(cudaMemcpy(L[layerNumber].cnn_adj_dev, L[layerNumber].cnn_adj, L[layerNumber].cnn_adjS*sizeof(float), cudaMemcpyHostToDevice));
  std::cout << "[INFO]:adjacency matrix are set!" << std::endl;
  return EIVE_GOOD;
}
//----------------------------------------------------------------------------------------------------------
//функция сохранения сети в файл
int EiveNeuronNetwork::saveToFile(char *file_name)
{
  if(!L) {	//если сеть не создана
    std::cout << "[ERROR][F:saveToFile]:the network is not created!" << std::endl;
    return EIVE_ERROR;
  }

  for(int i = 0; i < nLayer; i++) {	//если хотя бы один из слоёв не сконфигурирован
    if(L[i].layerType == -1) {
      std::cout << "[ERROR][F:saveToFile]:the network is not filled!" << std::endl;
      return EIVE_ERROR;
    }
  }

  std::ofstream file;
  file.open(file_name);
  if(!file.good()) {	//если не удалось открыть файл
    std::cout << "[ERROR][F:saveToFile]:no such file or directory!" << std::endl;
    return EIVE_ERROR;
  }

  //Если всё прошло удачно, формируем файл
  file << "EIVENETWORK ";		//помечаем файл
  file << nLayer << ' ';		//записываем количество слоёв
  file << inDataW << ' ';	//ширину входных данных
  file << inDataH << ' ';	//высоту входных данных

  //далее записываем все слои по очереди
  for(int i = 0; i < nLayer; i++)	{
    file << L[i].layerType << ' ';	//записываем тип слоя
    switch(L[i].layerType) {
      case LT_FULLYCONNECTED:
        file << L[i].fActiv << ' ';		//и функцию активации
        file << L[i].fc_nNeuron << ' ';		//записываем количество нейронов в слое
        //считываем весовые коэффициенты из устройства
        CUDA_CHECK_ERROR(cudaMemcpy(L[i].fc_w, L[i].fc_w_dev, L[i].wkSize*sizeof(float), cudaMemcpyDeviceToHost));
        for(int j = 0; j < L[i].wkSize; j++) {
          file << L[i].fc_w[j] << ' ';	//весовые коэффициенты
        }
        //считываем коэффициенты сдвига из устройства
        CUDA_CHECK_ERROR(cudaMemcpy(L[i].fc_b, L[i].fc_b_dev, L[i].fc_nNeuron*sizeof(float), cudaMemcpyDeviceToHost));
        for(int j = 0; j < L[i].fc_nNeuron; j++) {
          file << L[i].fc_b[j] << ' ';	//коэффициенты сдвига
        }
        break;

      case LT_CONVOLUTION:
        file << L[i].fActiv << ' ';		//и функцию активации
        file << L[i].cnn_nCore << ' ';		//записываем количество ядер свёртки
        file << L[i].cnn_sCore << ' ';		//размер ядра свёртки
        //считываем весовые коэффициенты из устройства
        CUDA_CHECK_ERROR(cudaMemcpy(L[i].cnn_k, L[i].cnn_k_dev, L[i].cnn_sCore*L[i].cnn_sCore*L[i].cnn_nCore*sizeof(float), cudaMemcpyDeviceToHost));
        for(int j = 0; j < L[i].wkSize; j++) {
          file << L[i].cnn_k[j] << ' ';	//весовые коэффициенты
        }
        //считываем коэффициенты сдвига из устройства
        CUDA_CHECK_ERROR(cudaMemcpy(L[i].cnn_b, L[i].cnn_b_dev, L[i].cnn_nCore*sizeof(float), cudaMemcpyDeviceToHost));
        for(int j = 0; j < L[i].cnn_nCore; j++) {
          file << L[i].cnn_b[j] << ' ';	//коэффициенты сдвига
        }
        if(i) {	//если слой не первый, то ещё и матрицу смежности
          for(int j = 0; j < L[i].cnn_adjS; j++) {
            file << L[i].cnn_adj[j] << ' ';
          }
        }
        break;

      case LT_MAXPOOLING:
        file << L[i].mp_sCore << ' ';		//записываем размер ядра
        break;
    }
  }
  std::cout << "[INFO]:Saving in the file " << file_name << " is complite" << std::endl;
  return EIVE_GOOD;
}
//----------------------------------------------------------------------------------------------------------
//функция загрузки сети из файла
int EiveNeuronNetwork::loadFromFile(char *file_name)
{
  std::ifstream file;
  file.open(file_name);
  if(!file.good()) {	//если не удалось открыть файл
    std::cout << "[ERROR][F:loadFromFile]:no such file or directory!" << std::endl;
    return EIVE_ERROR;
  }

  char mask[12] = {0};
  char nead_mask[] = "EIVENETWORK";
  for(int i = 0; i < 11; i++) {
    file >> mask[i];
  }
  if((strcmp(mask, nead_mask) == -1) || (strcmp(mask, nead_mask) == 1)) {
    std::cout << "[ERROR][F:loadFromFile]:the file doesn't contain a network!" << std::endl;
    return EIVE_ERROR;
  }

  //Если метка в файле найдена, начинаем считывать сеть
  file >> nLayer;		//считываем количество слоёв
  file >> inDataW;	//считываем ширину входных данных
  file >> inDataH;	//считываем высоту входных данных
  createNetwork();	//создаём сеть
  //И приступаем к её заполнению
  int fc_fA = 0;
  int fc_nN = 0;
  int cnn_fa = 0;
  int cnn_nCore = 0;
  int	cnn_sCore = 0;
  int mp_sCore = 0;
  float *fc_w = 0;
  float *fc_b = 0;
  float *cnn_k = 0;
  float *cnn_b = 0;
  float *cnn_adj = 0;
  for(int i = 0; i < nLayer; i++)
  {
    file >> L[i].layerType;	//записываем тип слоя
    switch(L[i].layerType)
    {
      case LT_FULLYCONNECTED:
        file >> fc_fA >> fc_nN;						//считываем функцию активации и количество нейронов в слое
        createFCLayer(i, fc_nN, fc_fA);				//создаём слой
        fc_w = new float [L[i].wkSize];
        for(int j = 0; j < L[i].wkSize; j++) {
          file >> fc_w[j];					//считываем весовые коэффициенты
        }
        setWeight(i, fc_w);						//устанавливаем
        delete[] fc_w;
        fc_b = new float [L[i].fc_nNeuron];
        for(int j = 0; j < L[i].fc_nNeuron; j++) {
          file >> fc_b[j];					//считываем коэффициенты сдвига
        }
        setB(i, fc_b);							//устанавливаем
        delete[] fc_b;
        break;

      case LT_CONVOLUTION:
        file >> cnn_fa >> cnn_nCore >> cnn_sCore; //считываем фун. актив. количество ядер и их размер
        createCNNLayer(i, cnn_nCore, cnn_sCore, cnn_fa);	//создаём слой
        cnn_k = new float [L[i].wkSize];
        for(int j = 0; j < L[i].wkSize; j++) {
          file >> cnn_k[j];						//считываем весовые коэффициенты
        }
        setWeight(i, cnn_k);						//устанавливаем
        delete[] cnn_k;
        cnn_b = new float [L[i].cnn_nCore];
        for(int j = 0; j < L[i].cnn_nCore; j++) {
          file >> cnn_b[j];						//считываем коэффициенты сдвига
        }
        setB(i, cnn_b);							//устанавливаем
        delete[] cnn_b;
        if(i) {	//если слой не первый, то считываем матрицу смежности
          cnn_adj = new float [L[i].cnn_adjS];
          for(int j = 0; j < L[i].cnn_adjS; j++)
            file >> cnn_adj[j];						//считываем матрицу смежности
          setAdjM(i, cnn_adj);						//устанавливаем
          delete[] cnn_adj;
        }
        break;

      case LT_MAXPOOLING:
        file >> mp_sCore;			//считываем размер ядра
        createMPLayer(i, mp_sCore);	//создаём слой
        break;
    }
  }
  return EIVE_GOOD;
}
//----------------------------------------------------------------------------------------------------------
//получение информации о сети
int EiveNeuronNetwork::getCNNLayerSCore(int numLayer) {
  if(numLayer >= this->nLayer) {
    return EIVE_ERROR;
  }

  return L[numLayer].cnn_sCore;
}
//----------------------------------------------------------------------------------------------------------
int EiveNeuronNetwork::getCNNLayerNCore(int numLayer) {
  if(numLayer >= this->nLayer) {
    return EIVE_ERROR;
  }

  return L[numLayer].cnn_nCore;
}
//----------------------------------------------------------------------------------------------------------
int EiveNeuronNetwork::getCNNLayerSMap(int numLayer) {
  if(numLayer >= this->nLayer) {
    return EIVE_ERROR;
  }

  return L[numLayer].cnn_sMap;
}
//----------------------------------------------------------------------------------------------------------
int EiveNeuronNetwork::getCNNLayerSMapW(int numLayer) {
  if(numLayer >= this->nLayer) {
    return EIVE_ERROR;
  }

  return L[numLayer].cnn_sMapW;
}
//----------------------------------------------------------------------------------------------------------
int EiveNeuronNetwork::getCNNLayerSMapH(int numLayer) {
  if(numLayer >= this->nLayer) {
    return EIVE_ERROR;
  }

  return L[numLayer].cnn_sMapH;
}
//----------------------------------------------------------------------------------------------------------
int EiveNeuronNetwork::getCNNLayerAdjS(int numLayer) {
  if(numLayer >= this->nLayer) {
    return EIVE_ERROR;
  }

  return L[numLayer].cnn_adjS;
}
//----------------------------------------------------------------------------------------------------------
int EiveNeuronNetwork::getMPLayerSCore(int numLayer) {
  if(numLayer >= this->nLayer) {
    return EIVE_ERROR;
  }

  return L[numLayer].mp_sCore;
}
//----------------------------------------------------------------------------------------------------------
int EiveNeuronNetwork::getMPLayerNCore(int numLayer) {
  if(numLayer >= this->nLayer) {
    return EIVE_ERROR;
  }

  return L[numLayer].mp_nCore;
}
//----------------------------------------------------------------------------------------------------------
int EiveNeuronNetwork::getMPLayerSMap(int numLayer) {
  if(numLayer >= this->nLayer) {
    return EIVE_ERROR;
  }

  return L[numLayer].mp_sMap;
}
//----------------------------------------------------------------------------------------------------------
int EiveNeuronNetwork::getMPLayerSMapW(int numLayer) {
  if(numLayer >= this->nLayer) {
    return EIVE_ERROR;
  }

  return L[numLayer].mp_sMapW;
}
//----------------------------------------------------------------------------------------------------------
int EiveNeuronNetwork::getMPLayerSMapH(int numLayer) {
  if(numLayer >= this->nLayer) {
    return EIVE_ERROR;
  }

  return L[numLayer].mp_sMapH;
}
//----------------------------------------------------------------------------------------------------------
void EiveNeuronNetwork::getOutData(float *dst) {
  memcpy(dst, outData, sizeof(float)*outDataSize);
}
