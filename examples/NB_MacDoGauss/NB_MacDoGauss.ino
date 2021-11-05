/*
   Naive Bayes algorithm implementation for the 
   classification of categorical data

   (c) Lesept, January 2021
*/
#include "NaiveBayes.h"
const int nData = 10;
const int nFeatures = 2;
const uint8_t nClasses = 2;

// features: {size, weight} <-- just recall the features names
enum Classes {KFC, McDo}; // 0 is KFC, 1 is McDo

void setup() {
  Serial.begin(115200);
  std::vector<Data>dataset;
  NB myNB(nData, nFeatures, nClasses);
  // create the dataset:
  myNB.addDataCat({180,   75,   KFC} , dataset);
  myNB.addDataCat({165,   61,   KFC} , dataset);
  myNB.addDataCat({167,   62,   McDo}, dataset);
  myNB.addDataCat({178,   63,   KFC} , dataset);
  myNB.addDataCat({174,   69,   KFC} , dataset);
  myNB.addDataCat({166,   60,   KFC} , dataset);
  myNB.addDataCat({167,   59,   McDo}, dataset);
  myNB.addDataCat({165,   60,   McDo}, dataset);
  myNB.addDataCat({173,   68,   KFC} , dataset);
  myNB.addDataCat({178,   71,   KFC} , dataset);

  // Test
  myNB.fit(dataset);
  uint8_t predict;
  predict = myNB.predictGau({177, 72}, dataset);
  Serial.printf("Prediction %d\n", predict);
  predict = myNB.predictGau({167, 60}, dataset);
  Serial.printf("Prediction %d\n", predict);
  myNB.destroyDataset (dataset);
}

void loop() {
}
