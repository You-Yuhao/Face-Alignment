#pragma once

#include<vector>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct Transform
{
	Matx22d scale_rotation;
	Matx21d translation;

	void Apply(vector<Point2d> *x, bool need_translation = true) const;
};

template<typename T>
inline T Sqr(T a)
{
	return a * a;
}

// Find the transform from y to x
Transform Procrustes(const vector<Point2d> &x,
	const vector<Point2d> &y);

vector<Point2d> ShapeAdjustment(const vector<Point2d> &shape,
	const vector<Point2d> &offset);

vector<Point2d> MapShape(Rect original_face_rect,
	const vector<Point2d> original_landmarks, Rect new_face_rect);