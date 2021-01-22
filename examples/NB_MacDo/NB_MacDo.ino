/*
   Naive Bayes algorithm implementation for the 
   classification of categorical data

   (c) Lesept, January 2021
*/
#include "NaiveBayes.h"
const int nData = 10;
const int nFeatures = 2;
const uint8_t nClasses = 2;

// features: {OS, DL_framework} <-- just recall the features names
enum OS {MacOS, Linux, Windows};
enum DL_framework {Tensorflow, Keras, Pytorch};
enum Classes {KFC, McDo};

void setup() {
  Serial.begin(115200);
  std::vector<Data>dataset;
  NB myNB(nData, nFeatures, nClasses);
  // create the dataset:
  myNB.addDataCat({MacOS,   Tensorflow, KFC} , dataset);
  myNB.addDataCat({Linux,   Keras,      KFC} , dataset);
  myNB.addDataCat({Linux,   Tensorflow, McDo}, dataset);
  myNB.addDataCat({MacOS,   Keras,      KFC} , dataset);
  myNB.addDataCat({Linux,   Keras,      KFC} , dataset);
  myNB.addDataCat({Windows, Keras,      KFC} , dataset);
  myNB.addDataCat({MacOS,   Pytorch,    McDo}, dataset);
  myNB.addDataCat({Windows, Pytorch,    McDo}, dataset);
  myNB.addDataCat({Linux,   Keras,      KFC} , dataset);
  myNB.addDataCat({MacOS,   Pytorch,    KFC} , dataset);

  // Test
  uint8_t predict;
  predict = myNB.predictCat({MacOS, Pytorch}, dataset);
  Serial.printf("Prediction %d\n", predict);
  predict = myNB.predictCat({Windows, Tensorflow}, dataset);
  Serial.printf("Prediction %d\n", predict);
  predict = myNB.predictCat({Linux, Tensorflow}, dataset);
  Serial.printf("Prediction %d\n", predict);
  myNB.destroyDataset (dataset);
}

void loop() {
}
