#include <Arduino.h>
#include <Wire.h>
#include <TFT_eSPI.h>
#include <SPI.h>
#define TFT_WIDTH 135
#define TFT_HEIGHT 240
#define BUTTON_1 35
#define BUTTON_2 0

// Variables for the library
#include "NaiveBayes.h"
int nData = 200;
const int nFeatures = 2;
uint8_t nClasses = 5;

// global variables
const int nPred = 100; // number of test data
bool calc = true;
int nOk = 0;
int r = TFT_WIDTH / 2 - 1;
int xc = r;
int yc = TFT_HEIGHT / 2 - 12;
float x, y;

TFT_eSPI tft = TFT_eSPI(TFT_WIDTH, TFT_HEIGHT);

void initTFT() {
  // Prepare the display
  char text[30];
  sprintf (text, "%d data", nData);
  tft.fillScreen (TFT_BLACK);
  tft.drawString(text, xc, 0, 4);
  // Draw the nClasses sectors
  tft.drawCircle (xc, yc, r, TFT_BLUE);
  for (int i = 0; i < nClasses; i++) {
    float t = i * 2 * 3.1415926 / nClasses;
    int x = xc + r * cos(t);
    int y = yc + r * sin(t);
    tft.drawLine (xc, yc, x, y, TFT_BLUE);
  }
}

uint8_t pickData () {
  // pick a random point in the circle of radius 1
  float xr = -1.0 + 2.0 * random(100) / 99.0;
  float yr = -1.0 + 2.0 * random(100) / 99.0;
  while (xr * xr + yr * yr >= 1.0) {
    xr = -1.0 + 2.0 * random(100) / 99.0;
    yr = -1.0 + 2.0 * random(100) / 99.0;
  }
  // then move it on the display
  float t = atan2(yr, xr) + PI;
  x = xc + r * xr;
  y = yc + r * yr;
  uint8_t n = t * nClasses / 2.0 / PI;
  return n;
}

void naiveBayes() {
  std::vector<Data>dataset;
  NB myNB(nData, nFeatures, nClasses);
  // create the dataset
  for (int i = 0; i < nData; i++) {
    std::vector<float> d;
    uint8_t n = pickData ();
    d.push_back(x);
    d.push_back(y);
    myNB.addData(d, n, dataset);
    // Set the pixel of the point of the dataset
    tft.drawPixel(x, y, TFT_WHITE);
  }
  // fit the model
  myNB.fit(dataset);

  // Test on nPred points
  nOk = 0;
  for (int i = 0; i < nPred; i++) {
    std::vector<float> d;
    uint8_t n = pickData ();
    d.push_back(x);
    d.push_back(y);
    uint8_t predict = myNB.predict(d, dataset);
    if (n == predict) {
      ++nOk;
      tft.drawPixel(x, y, TFT_GREEN); // draw green pixel if correct
      tft.drawPixel(x-1, y, TFT_GREEN); // draw green pixel if correct
      tft.drawPixel(x, y-1, TFT_GREEN); // draw green pixel if correct
      tft.drawPixel(x-1, y-1, TFT_GREEN); // draw green pixel if correct
    } else {
      tft.drawPixel(x, y, TFT_RED); // and red if not
      tft.drawPixel(x-1, y, TFT_RED); // and red if not
      tft.drawPixel(x, y-1, TFT_RED); // and red if not
      tft.drawPixel(x-1, y-1, TFT_RED); // and red if not
    }
  }
  Serial.printf("%d classes, %d data : success rate %6.2f%%\n", nClasses, nData, nOk * 100.0f / nPred);
  myNB.destroyDataset (dataset);

  // end
  calc = false;
}

void setup() {
  pinMode(BUTTON_1, INPUT_PULLUP);
  pinMode(BUTTON_2, INPUT_PULLUP);
  Serial.begin(115200);
  tft.begin();
  tft.setRotation(0);  // 0 & 2 Portrait. 1 & 3 landscape
  tft.setTextDatum(TC_DATUM);
  initTFT();
}

void loop() {
  // Button 1 : increase the size of the dataset
  if (digitalRead(BUTTON_1) == LOW) {
    nData = (nData + 50) % 550; // up to 500 data in dataset
    if (nData < 50) nData = 50;
    calc = true;
  }
  // Button 2 : increase the number of sectors
  if (digitalRead(BUTTON_2) == LOW) {
    nClasses = (nClasses + 1) % 16; // Up to 15 classes
    if (nClasses < 2) nClasses = 2;
    calc = true;
  }
  if (calc) {
    initTFT();
    naiveBayes();
    char text[30];
    // Display the success rate
    tft.drawString("Success", xc, TFT_HEIGHT - 50, 4);
    if (nOk == nPred) tft.drawString("100%", xc, TFT_HEIGHT - 25, 4);
    else {
      sprintf (text, "%4.1f%%", nOk * 100.0f / nPred);
      tft.drawString(text, xc, TFT_HEIGHT - 25, 4);
    }
    delay(100);
  }
}
