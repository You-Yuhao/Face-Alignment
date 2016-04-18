#pragma once

#include<vector>
#include<utility>
#include<string>
#include<opencv2/opencv.hpp>

#include "fern.h"
#include "utils.h"

using namespace std;
using namespace cv;

class Regressor
{
public:
	vector<Point2d> Apply(const Transform &t,
		Mat image, const vector<Point2d> &init_shape) const;

	void read(const FileNode &fn);

private:

	vector<pair<int, Point2d>> pixels_;
	vector<Fern> ferns_;
	Mat base_;
};

void read(const FileNode& node, Regressor& r, const Regressor&);
