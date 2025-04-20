#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

class Point5D {
public:
	float x, y, l, a, b;
	Point5D();
	~Point5D();
	void PointLab();
	void PointRGB();
	void MSPoint5DAccum(Point5D);
	void MSPoint5DCopy(Point5D);
	float MSPoint5DColorDistance(Point5D);
	float MSPoint5DSpatialDistance(Point5D);
	void MSPoint5DScale(float);
	void MSPOint5DSet(float, float, float, float, float);
	void Print();
};

class MeanShift {
public:
	float hs, hr;
	vector<Mat> IMGChannels;
	MeanShift(float, float);
	void MSFiltering(Mat&);
	void MSSegmentation(Mat&);
};

#endif 

