#include "face_x.h"
#include <algorithm>
#include <stdexcept>
#include "utils.h"

using namespace std;
using namespace cv;

FaceX::FaceX(const string & filename)
{
	FileStorage model_file;
	model_file.open(filename, FileStorage::READ);
	if (!model_file.isOpened())
		throw runtime_error("Cannot open model file \"" + filename + "\".");

	model_file["mean_shape"] >> mean_shape_;
	FileNode fn = model_file["test_init_shapes"];
	for (auto it = fn.begin(); it != fn.end(); ++it)
	{
		vector<Point2d> shape;
		*it >> shape;
		test_init_shapes_.push_back(shape);
	}
	fn = model_file["stage_regressors"];
	for (auto it = fn.begin(); it != fn.end(); ++it)
	{
		Regressor r;
		*it >> r;
		stage_regressors_.push_back(r);
	}
}

vector<Point2d> FaceX::Alignment(Mat image, Rect face_rect) const
{
	vector<vector<double>> all_results(test_init_shapes_[0].size() * 2);
	for (size_t i = 0; i < test_init_shapes_.size(); ++i)
	{
		vector<Point2d> init_shape = MapShape(Rect(0, 0, 1, 1),
			test_init_shapes_[i], face_rect);
		for (size_t j = 0; j < stage_regressors_.size(); ++j)
		{
			Transform t = Procrustes(init_shape, mean_shape_);
			vector<Point2d> offset =
				stage_regressors_[j].Apply(t, image, init_shape);
			t.Apply(&offset, false);
			init_shape = ShapeAdjustment(init_shape, offset);
		}

		for (size_t i = 0; i < init_shape.size(); ++i)
		{
			all_results[i * 2].push_back(init_shape[i].x);
			all_results[i * 2 + 1].push_back(init_shape[i].y);
		}
	}

	vector<Point2d> result(test_init_shapes_[0].size());
	for (size_t i = 0; i < result.size(); ++i)
	{
		nth_element(all_results[i * 2].begin(),
			all_results[i * 2].begin() + test_init_shapes_.size() / 2,
			all_results[i * 2].end());
		result[i].x = all_results[i * 2][test_init_shapes_.size() / 2];
		nth_element(all_results[i * 2 + 1].begin(),
			all_results[i * 2 + 1].begin() + test_init_shapes_.size() / 2,
			all_results[i * 2 + 1].end());
		result[i].y = all_results[i * 2 + 1][test_init_shapes_.size() / 2];
	}
	return result;
}
