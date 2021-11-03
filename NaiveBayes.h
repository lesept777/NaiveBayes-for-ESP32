/*
    Naive Bayes classification library for ESP32
    Inspired by:
    https://www.analyticsvidhya.com/blog/2021/01/a-guide-to-the-naive-bayes-algorithm/

	This library implements Naive Bayes classification for continuous data
	

    (c) 2021 Lesept
    contact: lesept777@gmail.com

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
    OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
    OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/
#ifndef NB_h
#define NB_h

#include <Arduino.h>

#define MIN_float      -HUGE_VAL
#define MAX_float      +HUGE_VAL

typedef struct
{
	std::vector<float> In; // vector of input data
	uint8_t           Out; // output (class)
} Data;

class NB 
{
public:
	NB  (int, int, int, bool);
	NB  (int, int, int);
	~NB ();
	void     addData (std::vector<float> const&, uint8_t, std::vector<Data> &);
	void     addData (std::vector<float> const&, std::vector<Data> &);
	void     addDataCat (std::vector<uint8_t> const&, std::vector<Data> &);
	void     fit     (std::vector<Data> &);
	uint8_t  predict (std::vector<float> &, std::vector<Data> &);
	uint8_t  predictCat (std::vector<uint8_t> &, std::vector<Data> const&);
	uint8_t  predictCatFit (std::vector<uint8_t> const&, std::vector<Data> const&);
	uint8_t  predictGau (std::vector<uint8_t> const&, std::vector<Data> const&);
	uint8_t  predictGauFit (std::vector<uint8_t> &, std::vector<Data> const&);
	void     destroyDataset (std::vector<Data> &);

private:
	std::vector<Data> _dataset;
	std::vector<float>valMin;
	std::vector<float>valMax;
	std::vector<int>  number;
	int      _nData, _nFeatures, _nClasses, _neighbours, _maxFeature;
	float    _radius;
	bool     _learn;
	int      createDataset    (std::vector<Data>);
	void     countDataset     (std::vector<Data> const&);
	int      normalizeDataset (std::vector<Data> &);
	inline float computeDistance  (std::vector<float> const&, std::vector<Data> const&, int);
	int      countNeighbours  (std::vector<float> const&, std::vector<Data> const&);
	uint8_t  findBestClass    (std::vector<float> const&, std::vector<Data> &);
	float    gaussProb        (float, float, float);
};

#endif