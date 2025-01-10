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
void resize_bilinear(const Mat& src, Mat& dst, double fx, double fy);
void Opencv_resize_test_case(string img_path, string save_opencv_path, string save_resize_path, double fx, double fy);
bool areImagesEqual(const std::string& img1Path, const std::string& img2Path);
