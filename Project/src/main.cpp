#include <opencv2/opencv.hpp>
#include <iostream>
#include "resize.h"

int main() {
    // 读取图片
    Mat image = cv::imread("../SUSTech.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not read the image: " << std::endl;
        return 1;
    }

    // 参数定义
    double fx = 20;
    double fy = 20;
    int test_times = 1000000;

    // cv::Size
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_times; i++){
        Mat dst;
        cv::Size dsize(image.rows/fy, image.cols/fx);
        cv::resize(image,dst,dsize,fx,fy,0);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "cv::Size\n" << test_times << " Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    // resize_parallel
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_times; i++){
        Mat dst;
        resize_parallel(image,dst,fx,fy);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "resize_parallel\n" << test_times << " Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    // resize_parallel_const
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_times; i++){
        Mat dst;
        resize_parallel_const(image,dst,fx,fy);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "resize_parallel_const\n" << test_times << " Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    // 自定义——串行计时
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_times/100; i++){
        Mat dst;
        resize(image,dst,fx,fy);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "resize\n" << test_times/100 << " Elapsed time: " << elapsed.count() << " seconds" << std::endl;
























    // imwrite("../out.jpg", dst);
    // cout << "缩放因子: \t"      << fy           << "\t" << fx           << endl;
    // cout << "原文件尺寸: \t"    << image.rows   << "\t" << image.cols   << endl;
    // cout << "目标文件尺寸:\t"   << dst.rows     << "\t" << dst.cols     << endl;
    return 0;
}

