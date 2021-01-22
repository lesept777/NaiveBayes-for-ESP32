# Naive Bayes classification for ESP32
This library is designed to be used with the Arduino IDE.

Naive Bayes classification is described [here](https://en.wikipedia.org/wiki/Naive_Bayes_classifier).

Naive Bayes classifiers are a set of supervised learning algorithms based on applying Bayes' theorem, but with strong independence assumptions between the features given the value of the class variable (hence naive).

This library implements a method of classifying a *categorical* or *continuous* input data into a set of classes, using a set of training data.

This work is inspired by: [Naive Bayes classifier](https://remykarem.github.io/blog/naive-bayes) and [A Guide to the Naive Bayes Algorithm](https://www.analyticsvidhya.com/blog/2021/01/a-guide-to-the-naive-bayes-algorithm/).

## Dependencies
* no dependency

## Quick start for continuous data
If you want to test it quickly, try the ["sectors" example](https://github.com/lesept777/NaiveBayes-for-ESP32/tree/master/examples/NB_Sectors).

This example tries to classify n-dimension points in a [0-1]^n space in sectors, depending if the coordinates are lower or higher than 0,5.
In 3D and above, this problem is hard to solve for a standard perceptron, but easily and quickly solved using this classifier. The Naive Bayes is still performant in higher dimensions, provided the training dataset in large enough.

## Quick start for categorical data
If you want to test it quickly, try the ["MacDo" example](https://github.com/lesept777/NaiveBayes-for-ESP32/tree/master/examples/NB_MacDo).

This example shows how to classify a set of categorical data and classify new data. The features of the training data are categorized in a limited set of values. There are 2 features, collected from 10 engineers: what OS (macOS, Linux or Windows) and deep learning framework (TensorFlow, Keras or PyTorch) they use. The class is their favourite fast-food (KFC or McDo). The task is to predict the favorite fast-food restaurant of a person, knowing its OS and preferred DL framework...

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
the last argument is a boolean. If set to `true` each prediction is used to increase the size of the training dataset, hopefully improving the prediction performance.

## Create a dataset
The dataset is a [vector](http://www.cplusplus.com/reference/vector/vector/) of `Data`, which is a `struct` defined as follows:
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
then fill in the dataset, using the method `addData` in the case of continuous data. For example:
```
  for (int i = 0; i < nData; i++) {
    std::vector<float> x;
    for (int j = 0; j < nFeatures; j++) x.push_back(/* the value of the j-th feature */);
    uint8_t out = sector(x);
    myNB.addData(x, out, dataset);
  }
```

For categorical data, another way to do it is possible, by calling the `addDataCat` method. It takes as arguments a vector of bytes (`uint8_t`) and the dataset created before. You can either use it as follows:
```
myNB.addDataCat ({ feature_1, feature2, ... feature_n, class}, dataset);
```
or within a loop, using a creation function:
```
  for (int i = 0; i < nData; i++) {
    std::vector<uint8_t> x(nFeatures + 1, 0);
    for (int j = 0; j < nFeatures; j++) x[i] = myfunction( ... );
    x[nFeatures] = myclass;
    myNB.addDataCat(x, dataset);
  }
```
or even read the dataset from a file stored in SPIFFS.

## Fit the classifier
For continuous data, the method for fitting the classifier is:
```
  myNB.fit(dataset);
```
This enables to pre-process the dataset and prepare for prediction.

For categorical data, there is no need to pre-process the dataset.

## Predict values
For continuous data: to predict a class, first create a features vector:
```
    std::vector<float> x;
    for (int j = 0; j < nFeatures; j++) x.push_back(/* the value of the j-th feature */);
```
Then call the `predict`method:
```
uint8_t predict = myNB.predict(x, dataset);
```

For categorical data, either create a vector of `uint8_t`containing the features to be classified
```
    std::vector<uint8_t> x(nFeatures, 0);
    for (int j = 0; j < nFeatures; j++) x[i] = /* i-th feature */;
    uint8_t predict = myNB.predictCat(x, dataset);
```
or directly call `predictCat` with the set of features as argument:
```
    uint8_t predict = myNB.predictCat({feature_1, feature_2, ... , feature_n}, dataset);
```

At the end, it is possible to free the memory occupied by the dataset, using
```
  myNB.destroyDataset (dataset);
```