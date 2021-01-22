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
int NB::normalizeDataset (std::vector<Data> &dataset) {
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
	for (int i = 0; i < _nData; i++) {
		for (int j = 0; j < _nFeatures; j++)
			dataset[i].In[j] = (dataset[i].In[j] - valMin[j]) /
			(valMax[j] - valMin[j]);
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
uint8_t NB::findBestClass (std::vector<float> const &data, std::vector<Data> &dataset) {
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

		if (postProba > best) {
			best = postProba;
			bestClass = classe;
		}
	}
	return bestClass;
}

/*   
	PUBLIC METHODS   
*/

// Method to call after defining the dataset
void NB::fit (std::vector<Data> &dataset) {
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

// Add a data to the current dataset
void NB::addData (std::vector<float> const &data, uint8_t out, std::vector<Data> &dataset) {
	Data temp;
    temp.In = data;
    temp.Out = out;
    dataset.push_back(temp);
}

// Add a data to the current dataset
void NB::addDataCat (std::vector<uint8_t> const &data, std::vector<Data> &dataset) {
	Data temp;
	for (uint8_t x : data) { // Range-based for loop (fun :)
		if (x > _maxFeature) _maxFeature = x;
		temp.In.push_back((float)x);
	}
    temp.Out = data[_nFeatures];
    dataset.push_back(temp);
    int n = dataset.size();
    // Serial.printf("data %d : %f %f %d max %d\n", n-1,dataset[n-1].In[0],dataset[n-1].In[1],dataset[n-1].Out, _maxFeature);
}

//
uint8_t  NB::predictCat (std::vector<uint8_t> const &data, std::vector<Data> const &dataset) {
	uint8_t bestClass = 255;
	float bestProba = 0.0f;
	//
	Serial.print("\nFeatures : {");
	for (uint8_t x : data) Serial.printf(" %d ", x);
	Serial.println("}");

	// Proba of each class
	std::vector<uint8_t>nbClasses(_nClasses,0);
	for (Data x : dataset) ++nbClasses[x.Out];

	// proba of each feature of the data for all classes
	std::vector<std::vector<uint8_t>> catMatrix(_nFeatures , std::vector<uint8_t>(_nClasses, 0));
	for (int i = 0; i < _nFeatures; i++) {
		for (int k = 0; k < _nData; k++) {
			if ((uint8_t)dataset[k].In[i] == data[i]) ++catMatrix[i][dataset[k].Out];
		}
	}


	// Compute denominator
	float sum = 0.0f;
	for (int j = 0; j < _nClasses; j++) {
		float prod = (float)nbClasses[j] / _nData;
		for (int i = 0; i < _nFeatures; i++) prod *= (float)catMatrix[i][j] / nbClasses[j];
		sum += prod;
	}

	// Compute probability for each class
	for (int j = 0; j < _nClasses; j++) {
		float prod = (float)nbClasses[j] / _nData;
		for (int i = 0; i < _nFeatures; i++) prod *= (float)catMatrix[i][j] / nbClasses[j];
		float proba = prod / sum;
		Serial.printf("Class %d : probability %6.2f%%\n", j, proba * 100.0f);
		if (proba > bestProba) {
			bestProba = proba;
			bestClass = j;
		}
	}

	return bestClass;
}