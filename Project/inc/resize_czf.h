#pragma once

#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
using namespace std;
using namespace cv;

void resize_org(const Mat& src, Mat& dst, double fx, double fy,double &time);
void resize_parallel(const Mat& src, Mat& dst, double fx, double fy,double &time);
void resize_parallel_const(const Mat& src, Mat& dst, const double fx, const double fy);
void resize_org_avx2(const Mat& src, Mat& dst, double fx, double fy,double &time);
void resize_parallel_avx2(const Mat& src, Mat& dst, double fx, double fy,double &time);
void resize_bilinear(const Mat& src, Mat& dst, double fx, double fy,double &time);
// void resize_bilinear_avx2(const Mat& src, Mat& dst, double fx, double fy,double &time);
void resize_bilinear_avx2(const Mat& src, Mat& dst, double fx, double fy, double &time);
