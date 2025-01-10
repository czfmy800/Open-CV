#include <cstddef>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "resize.h"

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    // 读取图片
    Mat image = cv::imread("../SUSTech.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not read the image: " << std::endl;
        return 1;
    }

    // 参数定义
    double fx = 20;
    double fy = 20;
    int test_times = 100000;

    // ---------------------------------------------------------
    // *                ** P1 最近邻插值实现 **                 *
    // ---------------------------------------------------------
    {
        Mat dst;
        resize(image,dst,fx,fy);
        cv::imwrite("../P1_resized_SUSTech.jpg", dst);
        
        cout << "P1 is down\n\n";
    }

    // ---------------------------------------------------------
    // *                ** P2 多通道支持 **                     *
    // ---------------------------------------------------------
    {
        // one channel
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        cv::imwrite("../P2_00_SUSTech_one_channel.jpg", channels[0]);
        Mat dst1;
        resize(channels[0],dst1,fx,fy);
        cv::imwrite("../P2_00_resized_SUSTech_one_channel.jpg", dst1);

        // three channels
        cv::imwrite("../P2_01_SUSTech_three_channel.jpg", image);
        Mat dst3;
        resize(image,dst3,fx,fy);
        cv::imwrite("../P2_01_resized_SUSTech_three_channel.jpg", dst3);

        cout << "P2 is down\n\n";
    }

    // ---------------------------------------------------------
    // *                ** P3 图像放大缩小支持 **               *
    // ---------------------------------------------------------
    {
        // enlarge
        fx = 0.25;
        fy = 0.25;
        Mat dst1;
        resize(image,dst1,fx,fy);
        cv::imwrite("../P3_00_enlarged_SUSTech.jpg", dst1);

        // minify
        fx = 15;
        fy = 15;
        Mat dst2;
        resize(image,dst2,fx,fy);
        cv::imwrite("../P3_01_minify_SUSTech.jpg", dst2);

        cout << "P3 is down\n\n";
    }

    // ---------------------------------------------------------
    // *                ** P4 多线程实现 **                     *
    // ---------------------------------------------------------
    {
        // save image
        Mat dst1;
        resize(image,dst1,fx,fy);
        cv::imwrite("../P4_00_normal_SUSTech.jpg", dst1);
        Mat dst2;
        resize_parallel(image,dst2,fx,fy);
        cv::imwrite("../P4_00_parallel_SUSTech.jpg", dst2);

        // normal one
        fx = 20;
        fy = 20;
        int new_times = test_times/1;
        auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < new_times; i++){
            Mat dst;
            resize(image,dst,fx,fy);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::chrono::duration<double> time_pre_test = elapsed/(new_times);
        std::cout << "normal resize\t" 
                    << new_times << "\tElapsed time: " << elapsed.count() << " seconds\t" 
                    << "average excuation time pre test: " << time_pre_test.count() << std::endl;

        // parallel one
        start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < test_times; i++){
            Mat dst;
            resize_parallel(image,dst,fx,fy);
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::chrono::duration<double> time_pre_test_2 = elapsed/(test_times);
        std::cout << "parallel resize\t" 
                    << test_times << "\tElapsed time: " << elapsed.count() << " seconds\t" 
                    << "average excuation time pre test: " << time_pre_test_2.count() << std::endl;

        // compare
        std::cout << "speed up: " << time_pre_test.count()/time_pre_test_2.count() << " times" << std::endl;

        cout << "P4 is down\n\n";
    }

    // ---------------------------------------------------------
    // *                ** P5 对比分析 **                       *
    // ---------------------------------------------------------
    {
        // save image
        Mat dst1;
        cv::Size dsize;
        cv::resize(image,dst1,cv::Size(),1/fx,1/fy,0);
        cv::imwrite("../P5_00_OpenCV_SUSTech.jpg", dst1);
        Mat dst2;
        resize_parallel(image,dst2,fx,fy);
        cv::imwrite("../P5_00_parallel_SUSTech.jpg", dst2);

        // OpenCV
        auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < test_times; i++){
            Mat dst;
            cv::resize(image,dst,cv::Size(),1/fx,1/fy,0);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::chrono::duration<double> time_pre_test_1 = elapsed/(test_times);
        std::cout << "OenCV\t\t" 
                    << test_times << " Elapsed time: " << elapsed.count() << " seconds\t"
                    << "average excuation time pre test: " << time_pre_test_1.count() << std::endl;

        // our best
        start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < test_times; i++){
            Mat dst;
            resize_parallel(image,dst,fx,fy);
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::chrono::duration<double> time_pre_test_2 = elapsed/(test_times);
        std::cout << "parallel resize\t" 
                    << test_times << " Elapsed time: " << elapsed.count() << " seconds\t" 
                    << "average excuation time pre test: " << time_pre_test_2.count() << std::endl;


        // compare
        std::cout << "OpenCV is speed up: " << time_pre_test_2.count()/time_pre_test_1.count() << " times" << std::endl;             
    
        cout << "P5 is down\n\n";
    }

    // ---------------------------------------------------------
    // *                ** B1 双线性插值实现 **                  *
    // ---------------------------------------------------------
    {
        // our function
        Mat dst;
        resize_bilinear(image,dst,fx,fy);
        cv::imwrite("../B1_resized_bilinear_SUSTech.jpg", dst);

        // OpenCV
        Mat dst1;
        cv::resize(image,dst1,cv::Size(),1/fx,1/fy);
        cv::imwrite("../B1_OpenCV_bilinear_SUSTech.jpg", dst1);
        
        cout << "B1 is down\n\n";
    }


    return 0;
}

