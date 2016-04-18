#include "regressor.h"

#include <utility>
#include <algorithm>
#include <stdexcept>


using namespace std;
using namespace cv;


vector<Point2d> Regressor::Apply(const Transform &t,
	Mat image, const std::vector<Point2d> &init_shape) const
{
	Mat pixels_val(1, pixels_.size(), CV_64FC1);
	vector<Point2d> offsets(pixels_.size());
	for (size_t j = 0; j < pixels_.size(); ++j)
		offsets[j] = pixels_[j].second;
	t.Apply(&offsets, false);

	double *p = pixels_val.ptr<double>(0);
	for (size_t j = 0; j < pixels_.size(); ++j)
	{
		Point pixel_pos = init_shape[pixels_[j].first] + offsets[j];
		if (pixel_pos.inside(Rect(0, 0, image.cols, image.rows)))
			p[j] = image.at<uchar>(pixel_pos);
		else
			p[j] = 0;
	}

	Mat coeffs = Mat::zeros(base_.cols, 1, CV_64FC1);
	for (size_t i = 0; i < ferns_.size(); ++i)
		ferns_[i].ApplyMini(pixels_val, coeffs);

	Mat result_mat = base_ * coeffs;

	vector<Point2d> result(init_shape.size());
	for (size_t i = 0; i < result.size(); ++i)
	{
		result[i].x = result_mat.at<double>(i * 2);
		result[i].y = result_mat.at<double>(i * 2 + 1);
	}
	return result;
}

void Regressor::read(const FileNode &fn)
{
	pixels_.clear();
	ferns_.clear();
	FileNode pixels_node = fn["pixels"];
	for (auto it = pixels_node.begin(); it != pixels_node.end(); ++it)
	{
		pair<int, Point2d> pixel;
		(*it)["first"] >> pixel.first;
		(*it)["second"] >> pixel.second;
		pixels_.push_back(pixel);
	}
	FileNode ferns_node = fn["ferns"];
	for (auto it = ferns_node.begin(); it != ferns_node.end(); ++it)
	{
		Fern f;
		*it >> f;
		ferns_.push_back(f);
	}
	fn["base"] >> base_;
}

void read(const FileNode& node, Regressor& r, const Regressor&)
{
	if (node.empty())
		throw runtime_error("Model file is corrupt!");
	else
		r.read(node);
}