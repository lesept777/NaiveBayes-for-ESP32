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

## Fit the classifier
.

## Predict values
The 