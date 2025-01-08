#pragma once

#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
using namespace std;
using namespace cv;

void resize(const Mat& src, Mat& dst, double fx, double fy);
void resize_parallel(const Mat& src, Mat& dst, double fx, double fy);
void resize_parallel_const(const Mat& src, Mat& dst, const double fx, const double fy);
