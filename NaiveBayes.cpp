#include "NaiveBayes.h"

/* constructor:
	nData 	 : number of data in the dataset 
	features : number of features
	nClasses : number of classes
	learn    : true if the prediction is added to the original dataset
*/
NB::NB(int nData, int nFeatures, int nClasses, bool learn) {
	_nFeatures = nFeatures;
	_nClasses  = nClasses;
	_nData     = nData;
	_learn     = learn;
	_maxFeature = 0;
}
NB::NB(int nData, int nFeatures, int nClasses) {
	_nFeatures = nFeatures;
	_nClasses  = nClasses;
	_nData     = nData;
	_learn     = false;
	_maxFeature = 0;
}

/*   PRIVATE METHODS   */

NB::~NB() {
}

void NB::countDataset (std::vector<Data> const &dataset) {
	for (int i = 0; i < _nClasses; i++) number.push_back(0);
	for (int i = 0; i < _nData; i++) ++number[dataset[i].Out];
}

// Set all data between 0 and 1
int NB::normalizeDataset (std::vector<Data> const &dataset) {
	valMin.clear();
	valMax.clear();
	_dataset.clear();

	for (int i = 0; i < _nFeatures; i++) {
		valMin.push_back(MAX_float);
		valMax.push_back(MIN_float);
	}

	// Search for extremum values of each feature
	for (int i = 0; i < _nData; i++) {
		if (dataset[i].Out > _nClasses) return -1;
		for (int j = 0; j < _nFeatures; j++) {
			if (dataset[i].In[j] < valMin[j]) 
				valMin[j] = dataset[i].In[j];
			if (dataset[i].In[j] > valMax[j]) 
				valMax[j] = dataset[i].In[j];
		}
	}

	// Normalize values
	for (int i = 0; i < _nData; ++i) {
		Data temp;
		std::vector<float> x;
		for (int j = 0; j < _nFeatures; ++j) {
			float y = (dataset[i].In[j] - valMin[j]) /
			(valMax[j] - valMin[j]);
			x.push_back(y);			
		}
		temp.In = x;
		temp.Out = dataset[i].Out;
		_dataset.push_back(temp);
	}

	return 0;
}

// Compute the distance of each element of the neighbourhood
// to the data
float NB::computeDistance(std::vector<float> const &data, std::vector<Data> const &dataset, int index) {
	float distance = 0.0f;
	for (int i = 0; i < _nFeatures; i++) {
		distance += abs(data[i] - dataset[index].In[i]);
	}
	return distance;
}

// Returns the number of data elements in the neighbourhood
int NB::countNeighbours(std::vector<float> const &data, std::vector<Data> const &dataset) {
	int neighbours = 0;
	for (int i = 0; i < _nData; i++)
		if (computeDistance(data, dataset, i) <= _radius) ++neighbours;
	return neighbours;
}

// returns the best class for the considered data
uint8_t NB::findBestClass (std::vector<float> const &data, std::vector<Data> const &dataset) {
	float marginal = _neighbours * 1.0 / _nData;
	float best = 0.0f;
	uint8_t bestClass = 255;
	for (int classe = 0; classe < _nClasses; classe++) {
		int nOk = 0;
		for (int i = 0; i < _nData; i++)
			if (computeDistance(data, dataset, i) <= _radius && dataset[i].Out == classe) ++nOk;
		// Bayes theorem:
		float likelihood = (float)nOk / number[classe];
		float priorProba = (float)number[classe] / _nData;
		float postProba = likelihood * priorProba / marginal;
		Serial.printf("Class %d : probability %.2f%\n", classe, 100 * postProba);

		if (postProba > best) {
			best = postProba;
			bestClass = classe;
		}
	}
	return bestClass;
}

// Compute Gaussian probability of one data
float NB::gaussProb(float x, float mean, float variance) {
	float den = sqrt(2.0f * PI * variance);
	float arg = pow(x - mean, 2) / 2.0f / variance;
	return exp(-arg) / den;
}

/*   PUBLIC METHODS   */

// Method to call after defining the dataset
void NB::fit (std::vector<Data> const &dataset) {
	if (normalizeDataset(dataset)) {
		Serial.println("Problem with the dataset");
		while (1);
	}
	countDataset(dataset);
}

// Method used to get the class prediction
uint8_t NB::predict (std::vector<float> &data, std::vector<Data> &dataset) {
	// 0: normalize data
	for (int i = 0; i < _nFeatures; i++)
		data[i] = (data[i] - valMin[i]) /
			(valMax[i] - valMin[i]);

	// 1: search a valid neighbourhood
	const int minNeighbours = _nData / 13;
	_radius = 0.05;
	_neighbours = countNeighbours(data, dataset);
	while (_neighbours < minNeighbours) {
		_radius += 0.05;
		_neighbours = countNeighbours(data, dataset);
	}

	// 2: select best class
	int best = findBestClass (data, dataset);

	// 3: add the new data to the dataset 
	// does not always improve learning...
	if (_learn) {
		Data temp;
		temp.In = data;
		temp.Out = best;
		dataset.push_back(temp);
		++_nData;
		fit(dataset);
	}

	return best;
}

// Free the memory used by the dataset
void NB::destroyDataset(std::vector<Data> &dataset) {
	dataset.erase(dataset.begin(), dataset.end());
    dataset.shrink_to_fit();
}

// Add a data to the current dataset, providing a vector of floats 
// containing the features and the class
void NB::addData (std::vector<float> const &data, std::vector<Data> &dataset) {
	Data temp;
    temp.Out = (uint8_t)data.back();  // get the class first (last element of the vector)
    temp.In = data;                   // then get the features
    dataset.push_back(temp);
}

// Add a data to the current dataset, providing a vector of floats 
// containing the features and an uint8_t for the class
void NB::addData (std::vector<float> const &data, uint8_t out, std::vector<Data> &dataset) {
	Data temp;
    temp.In = data;  // the features
    temp.Out = out;  // the class
    dataset.push_back(temp);
}

// Add a data to the current dataset (categorical data)
void NB::addDataCat (std::vector<uint8_t> const &data, std::vector<Data> &dataset)
{
	Data temp;
	for (uint8_t x : data) { // Range-based for loop (fun :)
		if (x > _maxFeature) _maxFeature = x;
		temp.In.push_back((float)x);
	}
    temp.Out = data[_nFeatures];
    dataset.push_back(temp);
    // int n = dataset.size();
    // Serial.printf("data %d : %f %f %d max %d\n", n-1,dataset[n-1].In[0],dataset[n-1].In[1],dataset[n-1].Out, _maxFeature);
}

// Predict a class for categorical data with fitting
uint8_t  NB::predictCat (std::vector<uint8_t> const &data, std::vector<Data> const &dataset)
{
	uint8_t bestClass = 255;
	float bestProba = 0.0f;
	//
	Serial.print("\nFeatures : {");
	for (uint8_t x : data) Serial.printf(" %d ", x);
	Serial.println("}");

	// Normalize data
	std::vector<float> dataNorm;
	for (int i = 0; i < _nFeatures; i++)
		dataNorm.push_back((data[i] - valMin[i]) / (valMax[i] - valMin[i]));

	// Number of samples in each class
	std::vector<uint16_t>nbClasses(_nClasses,0);
	for (Data x : dataset) ++nbClasses[x.Out];
	for (int k = 0; k < _nClasses; ++k)
		if (nbClasses[k] == 0) Serial.printf("Warning: class %d is empty!\n",k);

	// proba of each feature of the data for all classes
	std::vector<std::vector<uint16_t>> catMatrix(_nFeatures , std::vector<uint16_t>(_nClasses, 0));
	for (int i = 0; i < _nFeatures; ++i) {
		for (Data x : _dataset) {
			if (x.In[i] == dataNorm[i])
				++catMatrix[i][x.Out];
		}
	}

	// Look for empty matrix
	bool gauss = false;
	for (int i = 0; i < _nFeatures; ++i) 
		for (int k = 0; k < _nClasses; ++k) {
		if (catMatrix[i][k] == 0) {
			gauss = true;
			break;
		}
	}

	std::vector<std::vector<float>> meanMatrix(_nFeatures , std::vector<float>(_nClasses, 0));
	std::vector<std::vector<float>> varMatrix(_nFeatures , std::vector<float>(_nClasses, 0));
	if (gauss) {
// Mean and variance for each feature
		for (int i = 0; i < _nFeatures; i++)
			for (Data x : dataset)
				meanMatrix[i][x.Out] += x.In[i] / nbClasses[x.Out];

		for (int i = 0; i < _nFeatures; i++)
			for (Data x : dataset)
				varMatrix[i][x.Out] += pow(x.In[i] - meanMatrix[i][x.Out], 2) / nbClasses[x.Out];
	}

	// Compute denominator
	float sum = 0.0f;
	for (int j = 0; j < _nClasses; ++j) {
		float prod = (float)nbClasses[j] / _nData;
		for (int i = 0; i < _nFeatures; ++i) {
			if (catMatrix[i][j] == 0) {
				float prob = gaussProb(dataNorm[i], meanMatrix[i][j], varMatrix[i][j]);;
				prod *= prob;
			} else {
				prod *= (float)catMatrix[i][j] / (float)nbClasses[j];
			}
		}
		sum += prod;
	}

	// Compute probability for each class
	for (int j = 0; j < _nClasses; j++) {
		float prod = (float)nbClasses[j] / _nData;
		for (int i = 0; i < _nFeatures; i++) 
			if (catMatrix[i][j] == 0) {
				float prob = gaussProb(dataNorm[i], meanMatrix[i][j], varMatrix[i][j]);
				prod *= prob;
			} else {
				prod *= (float)catMatrix[i][j] / nbClasses[j];
			}
		float proba = prod / sum;
		Serial.printf("Class %d : probability %6.2f%%\n", j, proba * 100.0f);
		if (proba > bestProba) {
			bestProba = proba;
			bestClass = j;
		}
	}

	return bestClass;
}

// Predict a class for continuous data using Gaussian probability
uint8_t  NB::predictGau (std::vector<uint8_t> const &data, std::vector<Data> const &dataset)
{
	uint8_t bestClass = 255;
	float bestProba = 0.0f;
	//
	Serial.print("\nFeatures : {");
	for (uint8_t x : data) Serial.printf(" %d ", x);
	Serial.println("}");

	// Normalize data
	std::vector<float> dataNorm;
	for (int i = 0; i < _nFeatures; i++)
		dataNorm.push_back((data[i] - valMin[i]) / (valMax[i] - valMin[i]));

	// Number of data in each class
	std::vector<uint8_t>nbClasses(_nClasses,0);
	for (Data x : _dataset) ++nbClasses[x.Out];

	// Mean and variance for each feature
	std::vector<std::vector<float>> meanMatrix(_nFeatures , std::vector<float>(_nClasses, 0));
	for (int i = 0; i < _nFeatures; i++)
		for (Data x : _dataset)
			meanMatrix[i][x.Out] += x.In[i] / nbClasses[x.Out];

	std::vector<std::vector<float>> varMatrix(_nFeatures , std::vector<float>(_nClasses, 0));
	for (int i = 0; i < _nFeatures; i++)
		for (Data x : _dataset)
			varMatrix[i][x.Out] += pow(x.In[i] - meanMatrix[i][x.Out], 2) / nbClasses[x.Out];

	// Compute denominator
	float sum = 0.0f;
	for (int j = 0; j < _nClasses; j++) {
		float prod = (float)nbClasses[j] / _nData;
		for (int i = 0; i < _nFeatures; i++)
			prod *= gaussProb(dataNorm[i], meanMatrix[i][j], varMatrix[i][j]);
		sum += prod;
	}

	// Compute probability for each class
	for (int j = 0; j < _nClasses; j++) {
		float prod = (float)nbClasses[j] / _nData;
		for (int i = 0; i < _nFeatures; i++)
			prod *= gaussProb(dataNorm[i], meanMatrix[i][j], varMatrix[i][j]);
		float proba = prod / sum;
		Serial.printf("Class %d : probability %6.2f%%\n", j, proba * 100.0f);
		if (proba > bestProba) {
			bestProba = proba;
			bestClass = j;
		}
	}

	return bestClass;
}
