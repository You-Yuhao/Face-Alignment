#pragma once

#include<vector>
#include<utility>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct Fern
{
	void ApplyMini(Mat features, Mat coeffs)const;

	void read(const FileNode &fn);

	vector<double> thresholds;
	vector<pair<int, int>> features_index;
	vector<vector<pair<int, double>>> outputs_mini;
};

void read(const FileNode& node, Fern& f, const Fern&);
