# Naive Bayes classification for ESP32
This library is designed to be used with the Arduino IDE.



## Dependencies
* no dependency

## Quick start
If you want to test it quickly, try the ["sectors" example](https://github.com/lesept777/NaiveBayes-for-ESP32/tree/master/examples/NB_Sectors)

This example tries to classify n-dimension points in a [0-1]^n space in sectors, depending if the coordinates is lower or higher than 0,5.
In 3D and above, this problem is hard to solve for a standard perceptron, but easily and quickly solved using this classifier. The Naive Bayes is still performant in higher dimensions, provided the training dataset in large enough.

# Guidelines
## Declare an instance
To declare an instance of the classifier. 2 possibilities:
```
NB myNB(nData, nFeatures, nClasses);
```
* `nData` is the size of the training dataset (number of samples),
* `nFeatures` is the number of features of each training data,
* `nClasses` is the number of classes.

or
```
NB myNB(nData, nFeatures, nClasses, true);
```
the last argument is a boolean. If set to `true` each prediction is used to increase the size of the training dataset.

## Create a dataset
The dataset is a vector of `Data`, which is a `struct` defined as follows:
```
typedef struct
{
  std::vector<float> In; // vector of input data
  uint8_t           Out; // output (class)
} Data;
```
To define the dataset, use
```
  std::vector<Data>dataset;
```
then fill in the dataset. For example:
```
  for (int i = 0; i < nData; i++) {
    std::vector<float> x;
    for (int j = 0; j < nFeatures; j++) x.push_back(/* the value of the feature */);
    Data temp;
    temp.In = x;
    temp.Out = /* the value of the class */;
    dataset.push_back(temp);
  }
```

## Fit the classifier
The method for fitting the classifier is:
```
  myNB.fit(dataset);
```
This enables to pre-process the dataset and prepare for prediction.

## Predict values
To predict a class, first create a features vector:
```
    std::vector<float> x;
    for (int j = 0; j < nFeatures; j++) x.push_back(/* the value of the feature */);
```
Then call the `predict`method:
```
uint8_t predict = myNB.predict(x, dataset);
```
At the end, it is possible to free the memory occupied by the dataset, using
```
  myNB.destroyDataset (dataset);
```