#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <iomanip>

#include "eive_neuron_network.h"

int main(int argc, char **argv) {
  //Создаём нейронную сеть из шести слоёв
  //Размер входных данных 28х28
  //1 - свёрточный с 6 ядрами, размер ядра 5х5
  //2 - субдискретизирующий, размер ядра 2х2
  //3 - свёрточный с 50 ядрами, размер ядра 5х5
  //4 - субдискретизирующий, размер ядра 2х2
  //5 - полносвязный со 100 нейронами
  //6 - полносвязный с 10 нейронами
  //На всех слоях тангенсальная функция активации
  EiveNeuronNetwork _net;
  _net.setInDataW(28);
  _net.setInDataH(28);
  _net.setNLayer(6);
  _net.createNetwork();
  _net.createCNNLayer(0, 6, 5, FT_TANH);
  _net.createMPLayer(1, 2);
  _net.createCNNLayer(2, 50, 5, FT_TANH);
  _net.createMPLayer(3, 2);
  _net.createFCLayer(4, 100, FT_TANH);
  _net.createFCLayer(5, 10, FT_TANH);
  /*float *ADJ = new float [_net.getCNNLayerAdjS(2)];
  for(int i = 0; i < _net.getCNNLayerAdjS(2); i++)
    ADJ[i] = 1;
  _net.setAdjM(2,ADJ);
  delete ADJ;//*/
  _net.setSpeedTeach(0.001);

  //ОБУЧЕНИЕ
  int iterations = 10000;	//количество итерация обучения
  int iterCounter = 0;
  if(argc == 2) {
    iterations = atoi(argv[1]);
  }
  //проверяем, нет ли уже обученной сити, на таком количестве итераций
  char *filename = new char [256];
  std::ifstream nnFile;
  sprintf(filename, "it%d.enn", iterations);
  if(_net.loadFromFile(filename) < 0) {	//если файл не найден, приступаем к обучению
    //открываем файлы для обучения
    std::ifstream fileImages, fileLabels;
    fileImages.open("train-images-idx3-ubyte", std::ios::binary);
    if(!fileImages.good()) {
      std::cout << "!!!train-images-idx3-ubyte file not found, please run get_mnist.sh script!" << std::endl;
      return -1;
    }
    fileLabels.open("train-labels-idx1-ubyte", std::ios::binary);
    if(!fileLabels.good()) {
      std::cout << "!!!train-labels.idx1-ubyte file not found, please run get_mnist.sh script!" << std::endl;
      return -1;
    }
    int magicNumber1 = 0;
    int magicNumber2 = 0;
    int numberOfImages = 0;
    int numberOfLabels = 0;
    int numberOfRows = 0;
    int numberOfColums = 0;
    int imSize;
    unsigned char readBuff[4] = {0};
    fileImages.read((char*)readBuff, 4);
    for(int i = 0; i < 4; i++) {
      magicNumber1 += readBuff[i]*pow((float)256, 3-i);
    }
    fileLabels.read((char*)readBuff, 4);
    for(int i = 0; i < 4; i++) {
      magicNumber2 += readBuff[i]*pow((float)256, 3-i);
    }
    fileImages.read((char*)readBuff, 4);
    for(int i = 0; i < 4; i++) {
      numberOfImages += readBuff[i]*pow((float)256, 3-i);
    }
    fileLabels.read((char*)readBuff, 4);
    for(int i = 0; i < 4; i++) {
      numberOfLabels += readBuff[i]*pow((float)256, 3-i);
    }
    fileImages.read((char*)readBuff, 4);
    for(int i = 0; i < 4; i++) {
      numberOfRows += readBuff[i]*pow((float)256, 3-i);
    }
    fileImages.read((char*)readBuff, 4);
    for(int i = 0; i < 4; i++) {
      numberOfColums += readBuff[i]*pow((float)256, 3-i);
    }
    imSize = numberOfRows*numberOfColums;
    unsigned char *images = new unsigned char [numberOfImages*imSize];
    float *imagesF = new float [numberOfImages*imSize];
    unsigned char *labels = new unsigned char [numberOfLabels];
    fileImages.read((char*)images, numberOfImages*imSize);
    fileLabels.read((char*)labels, numberOfLabels);
    fileImages.close();
    fileLabels.close();

    for(int i = 0; i < numberOfImages*imSize; i++) {
      imagesF[i] = (float)images[i];
    }

    int nImage = 0;
    float maxErr = 0.01;
    float *d = new float[_net.getOutDataSize()];
    float *rezD = new float[_net.getOutDataSize()];
    float *rez = new float[_net.getOutDataSize()];
    //генерация случайной последовательности изображений
    std::vector<int> randomImagesNumber(numberOfImages);
    std::vector<int> objectImages;
    for(int i = 0; i < numberOfImages; i++) {
      objectImages.emplace_back(i);
    }
    for(int i = 0; i < numberOfImages; i++) {
      int n = rand()%(numberOfImages-i);
      randomImagesNumber[i] = objectImages[n];
      objectImages.erase(objectImages.begin() + n);
    }

    //приступаем к обучению с заданным количеством итераций
    while(iterations) {
      memset(d, 0, sizeof(float)*_net.getOutDataSize());
      float err = 0;
      //выставляем единицу на выходном нейроне, в зависимости от лейбла данного изображения
      d[labels[randomImagesNumber[nImage]]] = 1;
      _net.teachNetwork(imagesF + randomImagesNumber[nImage]*imSize, d, maxErr, &err);

      //сортируем для вывода
      int D = labels[randomImagesNumber[nImage]];
      _net.getOutData(rezD);
      memset(rez, 0, sizeof(float)*_net.getOutDataSize());
      for(int i = 0; i < _net.getOutDataSize(); i++) {
        double max = -2;
        int V = 0;
        for(int j = 0; j < _net.getOutDataSize(); j++) {
          if(rezD[j] > max) {
            max = rezD[j];
            V = j;
          }
        }
        rezD[V] = -2;
        rez[i] = V;
      }

      std::cout << std::setw(10) << iterCounter++ << '(' << std::setw(6) << randomImagesNumber[nImage] << ") " << err << "\t\t" << "D:" << D << '\t' << "R:";
      for(int i = 0; i < 10; i++) {
        std::cout << rez[i] << ' ';
      }
      std::cout << std::endl;
      nImage++;
      //если прошли по всем 60000 изображений, перемешиваем их заново
      if(nImage == numberOfImages)	{
        nImage = 0;
        for(int i = 0; i < numberOfImages; i++) {
          objectImages.emplace_back(i);
        }
        for(int i = 0; i < numberOfImages; i++) {
          int n = rand()%(numberOfImages-i);
          randomImagesNumber[i] = objectImages[n];
          objectImages.erase(objectImages.begin() + n);
        }
      }
      iterations--;
    }
    delete[] d;
    delete[] rezD;
    delete[] rez;
    _net.saveToFile(filename);
  }

  //ПРОВЕРКА
  //открываем файлы для проверки
  std::ifstream fileImages, fileLabels;
  fileImages.open("t10k-images-idx3-ubyte", std::ios::binary);
  if(!fileImages.good()) {
    std::cout << "!!!t10k-images-idx3-ubyte file not found, please run get_mnist.sh script!" << std::endl;
    return -1;
  }
  fileLabels.open("t10k-labels-idx1-ubyte", std::ios::binary);
  if(!fileLabels.good()) {
    std::cout << "!!!t10k-labels-idx1-ubyte file not found, please run get_mnist.sh script!" << std::endl;
    return -1;
  }
  int magicNumber1 = 0;
  int magicNumber2 = 0;
  int numberOfImages = 0;
  int numberOfLabels = 0;
  int numberOfRows = 0;
  int numberOfColums = 0;
  int imSize;
  unsigned char readBuff[4] = {0};
  fileImages.read((char*)readBuff, 4);
  for(int i = 0; i < 4; i++) {
    magicNumber1 += readBuff[i]*pow((float)256, 3-i);
  }
  fileLabels.read((char*)readBuff, 4);
  for(int i = 0; i < 4; i++) {
    magicNumber2 += readBuff[i]*pow((float)256, 3-i);
  }
  fileImages.read((char*)readBuff, 4);
  for(int i = 0; i < 4; i++) {
    numberOfImages += readBuff[i]*pow((float)256, 3-i);
  }
  fileLabels.read((char*)readBuff, 4);
  for(int i = 0; i < 4; i++) {
    numberOfLabels += readBuff[i]*pow((float)256, 3-i);
  }
  fileImages.read((char*)readBuff, 4);
  for(int i = 0; i < 4; i++) {
    numberOfRows += readBuff[i]*pow((float)256, 3-i);
  }
  fileImages.read((char*)readBuff, 4);
  for(int i = 0; i < 4; i++) {
    numberOfColums += readBuff[i]*pow((float)256, 3-i);
  }
  imSize = numberOfRows*numberOfColums;
  unsigned char *images = new unsigned char [numberOfImages*imSize];
  float *imagesF = new float [numberOfImages*imSize];
  unsigned char *labels = new unsigned char [numberOfLabels];
  fileImages.read((char*)images, numberOfImages*imSize);
  fileLabels.read((char*)labels, numberOfLabels);
  fileImages.close();
  fileLabels.close();

  for(int i = 0; i < numberOfImages*imSize; i++) {
    imagesF[i] = (float)images[i];
  }

  float *outData = new float[_net.getOutDataSize()];
  int err = 0;
  std::cout << "Testing..." << std::endl;
  for(int i = 0; i < numberOfImages; i++) {
    _net.colculateNetwork(imagesF + i*imSize);
    _net.getOutData(outData);
    float max = -2;
    int maxN = -1;
    for(int j = 0; j < _net.getOutDataSize(); j++) {
      if(outData[j] > max) {
        max = outData[j];
        maxN = j;
      }
    }
    if(maxN != labels[i]) {
      err++;
    }
  }
  std::cout << std::endl;
  delete[] outData;

  std::cout << "Bad recognition = " << err << std::endl;
  std::cout << "Error percent = " << (float)err/(float)numberOfImages << "%" << std::endl;

  return 0;
}
