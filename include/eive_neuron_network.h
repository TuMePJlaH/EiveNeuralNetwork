//Описание класса EIVE нейронной сети v 1.0.0

#include "eive_neuron_layer.h"

//Раскомментировать при отладки
//#define DEBUG_NETWORK

const int EIVE_ERROR	= -1;
const int EIVE_GOOD		=	0;

//для случайного заполнения весовых коэффициентов
const float MIN_W_RAND			=-0.1;    //минимальное значение весового коэффициента при рандомном заполнении
const float MAX_W_RAND			= 0.1;    //максимальное значение весового коэффициента при рандомном заполнении
const float RAND_THRESHOLD	= 2.0;    //точность в знаках после запятой

//размер грида
const int MIN_BLOCK_SIZE = 10;

class EiveNeuronNetwork{
private:
  int inDataSize;   //размер входных данных
  int nLayer;       //общее количество слоёв
  int inDataW;      //ширина входных данных
  int inDataH;      //высота входных данных
  float *inData;    //входные данные
  float *inDataDev; //для CUDA

  int outDataSize;        //размер выходных данных
  float *outData;         //выходные данные
  float *outDataDev;      //для CUDA
  float *neadOutData;     //необходимые выходные данные (для обучения)
  float *neadOutDataDev;  //для CUDA

  EiveNeuronLayer *L; //слои

  float speedTeach;   //скорость обучения

public:
  //конструкторы и деструкторы класса
  EiveNeuronNetwork();  //конструктор пустой сети
  ~EiveNeuronNetwork();

  //функции управления сетью
  int setNLayer(int nLayer);
  int getNLayer() { return nLayer; }
  int setInDataW(int inDataW);
  int getInDataW() { return inDataW; }
  int setInDataH(int inDataH);
  int getInDataH() { return inDataH; }
  int setSpeedTeach(float speedTeach);
  float getSpeedTeach() { return speedTeach; }
  int createNetwork();					//функция создания сети
  int createCNNLayer(int layerNumber, int numberOfCortex,	int cortexSize, int activatFunction); //функция создания свёрточного слоя
  int createFCLayer(int layerNumber, int numberOfNeuron, int activatFunction); //функция создания полносвязного слоя
  int createMPLayer(int layerNumber, int cortexSize);	//функция создания субдискретизируещего слоя
  int setWeight(int layerNumber, float *w);	//функция установки весовых коэффициентов
  int setB(int layerNumber,	float *b);	//функция установки коэффициентов сдвига
  int setAdjM(int layerNumber, float *adjM);	//функция установки матрицы сопряжения
  int saveToFile(char *file_name);	//функция сохранения сети в файл
  int loadFromFile(char *file_name);	//функция загрузки сети из файла

  //получение информации о сети
  int getCNNLayerSCore(int numLayer);
  int getCNNLayerNCore(int numLayer);
  int getCNNLayerSMap(int numLayer);
  int getCNNLayerSMapW(int numLayer);
  int getCNNLayerSMapH(int numLayer);
  int getCNNLayerAdjS(int numLayer);
  int getMPLayerSCore(int numLayer);
  int getMPLayerNCore(int numLayer);
  int getMPLayerSMap(int numLayer);
  int getMPLayerSMapW(int numLayer);
  int getMPLayerSMapH(int numLayer);
  int getOutDataSize() { return outDataSize; }
  void getOutData(float *dst);

  //функции для работы
  int colculateNetwork(float *inputData);	//функция прямого прогона сети

  //функции для обучения
  int teachNetwork(float *inputData, float *neadOutput, float max_error, float *real_error);	//функция обучения сети
};

