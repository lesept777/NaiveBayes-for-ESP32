/*
   Naive Bayes algorithm implementation for continuous data
   Application to classification in sectors (above or below 0.5)
   of n-dimension points in a 0-1 space

   200 training data is OK for 2D and 3D, but not enough for more-D

   (c) Lesept, January 2021
*/
#include "NaiveBayes.h"
const int nData = 200;
// in 2D : nFeatures = 2 leads to 4 classes
// in 3D : nFeatures = 3 leads to 8 classes
// in 4D : nFeatures = 4 leads to 16 classes
const int nFeatures = 2;
const uint8_t nClasses = 1 << nFeatures;

uint8_t sector (std::vector<float>x) {
  uint8_t nClass = 0;
  for (int i = 0; i < nFeatures; i++) nClass += (x[i] > 0.5) << i;
  //  in 3D, this is equivalent to:
  //  (x[0] > 0.5) + (x[1] > 0.5) * 2 + (x[2] > 0.5) * 4;
  return nClass;
}


void setup() {
  Serial.begin(115200);
  std::vector<Data>dataset;
  NB myNB(nData, nFeatures, nClasses);
  // create the dataset:
  //    draw random points in [0,1]^nFeatures and
  //    set output to 0 ... nClasses depending on position
  for (int i = 0; i < nData; i++) {
    std::vector<float> x;
    for (int j = 0; j < nFeatures; j++) x.push_back(random(256) / 255.);
    Data temp;
    temp.In = x;
    temp.Out = sector(x);
    dataset.push_back(temp);
  }
  myNB.fit(dataset);

  // Test 100 random data
  int nPred = 100;
  int ok = 0;
  for (int i = 0; i < nPred; i++) {
    std::vector<float> x;
    for (int j = 0; j < nFeatures; j++) x.push_back(random(256) / 255.);
    uint8_t expect = sector(x);
    uint8_t predict = myNB.predict(x, dataset);
    if (expect == predict) ++ok;
  }
  Serial.printf("Success rate %6.2f%%\n", ok * 100.0f / nPred);
  myNB.destroyDataset (dataset);
}

void loop() {
}
