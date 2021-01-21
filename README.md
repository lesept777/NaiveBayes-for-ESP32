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
* `nData` is the size of the training dataset (number of samples)
* `nFeatures` is the number of features of each training data
* `nClasses` is the number of classes


## Create a dataset


## Fit the classifier
.

## Predict values
The 