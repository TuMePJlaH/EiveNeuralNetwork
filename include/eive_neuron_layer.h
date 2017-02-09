//Описание класса EIVE нейронного слоя v 1.0.0
#pragma once

//типы слоёв
const int LT_FULLYCONNECTED	=	0;    // полносвязный
const int LT_CONVOLUTION		= 1;    // свёрточный
const int LT_MAXPOOLING			= 2;    // субдискретизирующий

//типы активационных функций
const int FT_SIGMOID	= 0;  // сигмойдная функция: f(x) = 1/(1+e^(-x))
const int FT_TANH			= 1;  // гиперболический тангенс f(x) = tanh(x)

class EiveNeuronLayer {
private:
  int layerType;    //тип слоя
  int fActiv;       //функция активации
  int layerNumber;  //номер слоя
  int wkSize;       //общее количество весовых коэффициентов

  //для LT_FULLYCONNECTED
  int fc_nNeuron;       //общее количество нейронов
  float *fc_w;          //весовые коэффициенты слоя полного подключения
  float *fc_w_dev;      //для CUDA
  float *fc_y;          //выход со слоя полного подключения
  float *fc_y_dev;      //для CUDA
  float *fc_delta;      //ошибки слоя полного подключения
  float *fc_delta_dev;  //для CUDA
  float *fc_b;          //коэффициент сдвига слоя полного подключения
  float *fc_b_dev;      //для CUDA

  //для LT_CONVOLUTION
  int cnn_sCore;        //размер ядра свёртки (s_core x s_core)
  int cnn_nCore;        //количество ядер
  int cnn_sMap;         //размер карты признаков
  int cnn_sMapW;        //ширина карты признаков
  int cnn_sMapH;        //высота карты признаков
  int cnn_adjS;         //размер матрицы смежности
  float *cnn_k;         //весовые коэффициенты свёрточного слоя
  float *cnn_k_dev;     //для CUDA
  float *cnn_y;         //выход со свёрточного слоя (карта признаков)
  float *cnn_y_dev;     //для CUDA
  float *cnn_delta;     //ошибка свёрточного слоя
  float *cnn_delta_dev; //для CUDA
  float *cnn_adj;       //матрица смежноси с предыдущем слоем
  float *cnn_adj_dev;   //для CUDA
  float *cnn_b;         //коэффициент сдвига
  float *cnn_b_dev;     //для CUDA

  //для LT_MAXPOOLING
  int mp_sCore;         //размер ядра субдискретизации
  int mp_nCore;         //количество ядер
  int mp_sMap;          //размер карты признаков
  int mp_sMapW;         //ширина карты признаков
  int mp_sMapH;         //высота карты признаков
  float *mp_y;          //выход со субдискретизирующего слоя (карта признаков)
  float *mp_y_dev;      //для CUDA
  float *mp_delta;      //ошибка субдискретизирующего слоя
  float *mp_delta_dev;  //для CUDA
  float *mp_mask;       //полученная маска, после прямого прохода (необходима для обучения при обратно проходе)
  float *mp_mask_dev;   //для CUDA

public:
  EiveNeuronLayer()  {
    layerType = -1;
    fActiv = -1;
    layerNumber = -1;
    wkSize = -1;
    fc_nNeuron = -1;
    fc_w = nullptr;
    fc_w_dev = nullptr;
    fc_y = nullptr;
    fc_y_dev = nullptr;
    fc_delta = nullptr;
    fc_delta_dev = nullptr;
    fc_b = nullptr;
    fc_b_dev = nullptr;
    cnn_sCore = -1;
    cnn_nCore = -1;
    cnn_sMap = -1;
    cnn_sMapW = -1;
    cnn_sMapH = -1;
    cnn_adjS = -1;
    cnn_k = nullptr;
    cnn_k_dev = nullptr;
    cnn_y = nullptr;
    cnn_y_dev = nullptr;
    cnn_delta = nullptr;
    cnn_delta_dev = nullptr;
    cnn_adj = nullptr;
    cnn_adj_dev = nullptr;
    cnn_b = nullptr;
    cnn_b_dev = nullptr;
    mp_sCore = -1;
    mp_nCore = -1;
    mp_sMap = -1;
    mp_sMapW = -1;
    mp_sMapH = -1;
    mp_y = nullptr;
    mp_y_dev = nullptr;
    mp_delta = nullptr;
    mp_delta_dev = nullptr;
    mp_mask = nullptr;
    mp_mask_dev = nullptr;
    return;
  }
  friend class EiveNeuronNetwork;
};
