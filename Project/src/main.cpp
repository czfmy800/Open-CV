#include <cstddef>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "resize.h"

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    // 读取图片
    Mat image = cv::imread("../image/src/SUSTech.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not read the image: " << std::endl;
        return 1;
    }

    // 参数定义
    double fx = 10;
    double fy = 10;
    int test_times = 1000000;

    // ---------------------------------------------------------
    // *                ** P1 最近邻插值实现 **                 *
    // ---------------------------------------------------------
    {
        Mat dst;
        resize(image,dst,fx,fy);
        cv::imwrite("../image/P1_resized_SUSTech.jpg", dst);
        
        cout << "P1 is down\n\n";
    }

    // ---------------------------------------------------------
    // *                ** P2 多通道支持 **                     *
    // ---------------------------------------------------------
    {
        // one channel
        Mat image_one = cv::imread("../image/src/SUSTech_one_channel.jpg", cv::IMREAD_GRAYSCALE);
        Mat dst1;
        resize(image_one,dst1,fx,fy);
        cv::imwrite("../image/P2_00_resized_SUSTech_one_channel.jpg", dst1);

        // three channels
        cv::imwrite("../image/P2_01_SUSTech_three_channel.jpg", image);
        Mat dst3;
        resize(image,dst3,fx,fy);
        cv::imwrite("../image/P2_01_resized_SUSTech_three_channel.jpg", dst3);

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
        cv::imwrite("../image/P3_00_enlarged_SUSTech.jpg", dst1);

        // minify
        fx = 15;
        fy = 15;
        Mat dst2;
        resize(image,dst2,fx,fy);
        cv::imwrite("../image/P3_01_minify_SUSTech.jpg", dst2);

        cout << "P3 is down\n\n";
    }

    // ---------------------------------------------------------
    // *                ** P4 多线程实现 **                     *
    // ---------------------------------------------------------
    {
        // save image
        fx = 20;
        fy = 20;
        Mat dst1;
        resize(image,dst1,fx,fy);
        cv::imwrite("../image/P4_00_normal_SUSTech.jpg", dst1);
        Mat dst2;
        resize_parallel(image,dst2,fx,fy);
        cv::imwrite("../image/P4_00_parallel_SUSTech.jpg", dst2);

        // normal one
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
        // save image 1
        Opencv_resize_test_case("../image/src/SUSTech.jpg", 
                                "../image/P5_00_OpenCV_SUSTech.jpg", 
                                "../image/P5_00_parallel_SUSTech.jpg", 20, 20);
        Opencv_resize_test_case("../image/src/SUSTech.jpg", 
                                "../image/P5_01_OpenCV_SUSTech.jpg", 
                                "../image/P5_01_parallel_SUSTech.jpg", 0.5, 0.5);
        Opencv_resize_test_case("../image/src/SUSTech_one_channel.jpg",
                                "../image/P5_02_OpenCV_SUSTech.jpg",
                                "../image/P5_02_parallel_SUSTech.jpg", 10, 10);

        // save image 2
        Opencv_resize_test_case("../image/src/calendar.jpg",
                                "../image/P5_10_OpenCV_calendar.jpg",
                                "../image/P5_10_parallel_calendar.jpg", 10.8, 10.8);
        Opencv_resize_test_case("../image/src/calendar.jpg",
                                "../image/P5_11_OpenCV_calendar.jpg",
                                "../image/P5_11_parallel_calendar.jpg", 0.46, 0.46);
        Opencv_resize_test_case("../image/src/calendar_one_channel.jpg",
                                "../image/P5_12_OpenCV_calendar.jpg",
                                "../image/P5_12_parallel_calendar.jpg", 32.6, 57.8);

        // save image 3
        Opencv_resize_test_case("../image/src/test_image.jpg",
                                "../image/P5_20_OpenCV_test_image.jpg",
                                "../image/P5_20_parallel_test_image.jpg", 3.2, 5.9);
        Opencv_resize_test_case("../image/src/test_image.jpg",
                                "../image/P5_21_OpenCV_test_image.jpg",
                                "../image/P5_21_parallel_test_image.jpg", 4.4, 7.6);
        Opencv_resize_test_case("../image/src/test_image_one_channel.jpg",
                                "../image/P5_22_OpenCV_test_image.jpg",
                                "../image/P5_22_parallel_test_image.jpg", 222, 333);

        // OpenCV
        fx = 20;
        fy = 20;
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
        cv::imwrite("../image/B1_resized_bilinear_SUSTech.jpg", dst);

        // OpenCV
        Mat dst1;
        cv::resize(image,dst1,cv::Size(),1/fx,1/fy);
        cv::imwrite("../image/B1_OpenCV_bilinear_SUSTech.jpg", dst1);


        // OpenCV
        fx = 20;
        fy = 20;
        auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < test_times; i++){
            Mat dst;
            cv::resize(image,dst,cv::Size(),1/fx,1/fy);
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
            resize_bilinear(image,dst,fx,fy);
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::chrono::duration<double> time_pre_test_2 = elapsed/(test_times);
        std::cout << "parallel resize\t" 
                    << test_times << " Elapsed time: " << elapsed.count() << " seconds\t" 
                    << "average excuation time pre test: " << time_pre_test_2.count() << std::endl;

        // compare
        std::cout << "OpenCV is speed up: " << time_pre_test_2.count()/time_pre_test_1.count() << " times" << std::endl;             
    

        
        cout << "B1 is down\n\n";
    }


    return 0;
}

