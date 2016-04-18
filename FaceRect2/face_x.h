#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "regressor.h"

using namespace std;
using namespace cv;

class FaceX
{
public:
	// Construct the object and load model from file.
	//
	// filename: The file name of the model file.
	//
	// Throw runtime_error if the model file cannot be opened.
	FaceX(const string &filename);

	// Do face alignment.
	//
	// image: The image which contains face. Must be 8 bits gray image.
	// face_rect: Where the face locates.
	//
	// Return the landmarks. The number and positions of landmarks depends on
	// the model.
	vector<Point2d> Alignment(Mat image, Rect face_rect) const;

	// Return how many landmarks the model provides for a face.
	int landmarks_count() const
	{
		return mean_shape_.size();
	}

private:
	vector<Point2d> mean_shape_;
	vector<vector<Point2d>> test_init_shapes_;
	vector<Regressor> stage_regressors_;
};