#include <cstddef>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "resize.h"
#include <immintrin.h>
#include <random>
#include <string>

void compareMats(const Mat& mat1, const Mat& mat2);
using namespace std;
int main() {

    cout<<"--------下面进行resize算法是否使用simd算法的正确性测试--------"<<endl;
    Mat image = cv::imread("/root/CPP_Project/Project/SUSTech.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not read the image: " << std::endl;
        return 1;
    }

    // 参数定义
    double fx = 0.5;
    double fy = 0.5;
    int test_times = 100000;


{
    Mat *dst = new Mat();
    Mat *dst2 = new Mat();



    double time;

    resize_org(image, *dst, fx, fy,time);
    cv::imwrite("../P1_resized_SUSTech.jpg", *dst);
    resize_org_avx2(image, *dst2, fx, fy,time);
    cv::imwrite("../P1_resized_SUSTech_AVX2.jpg", *dst2); // 修正为写入dst2


    compareMats(*dst,*dst2);

    delete dst;
    delete dst2;

}

{
    cout<<"-----------下面进行resize算法加速比测试----------"<<endl;
    std::cout<<"下面进行resize算法是否使用simd加速的加速比测试。具体来说,我们随机生成100张图片,分别交由"<<
    "标准算法和simd加速算法处理"<<endl;
    std::cout<<"图片尺寸统一选取800*800,缩放比依次为10,5,1,0.5,0.2"<<endl;
    float array [5]={10,5,1,0.5,0.2};
 std::cout<<"----缩放比----"<<"标准算法处理时间(ms)----"<<"simd加速算法处理时间(ms)----"<<"加速比----"<<endl;
    // 参数定义
    for (int j = 0; j<5;j++){
    const int image_size = 800;
    const int num_images = 100;
    const double fx = array[j]; // 缩放因子，可以根据需要调整
    const double fy = array[j];

    // 生成随机图片并存储在vector中
    double time_stander = 0;
    double time_avx2 = 0;

    for(int i = 0; i < num_images; ++i) {
        Mat images (image_size, image_size, CV_8UC3);
        // 生成随机颜色值
        cv::randu(images, cv::Scalar::all(0), cv::Scalar::all(255));



    // 变量用于存储结果
       cv::Mat dst, dst_avx2;

    // 记录标准resize的总时间


        resize_org(images, dst, fx, fy,time_stander);

        resize_org_avx2(images, dst_avx2, fx, fy,time_avx2);




    // 输出时间对比
    // std::cout << "标准resize总时间: " << time_stander/1000000 << " ms." << std::endl;
    // std::cout << "AVX2优化resize总时间: " << time_avx2/1000000 << " ms." << std::endl;

    // 输出加速比

    // std::cout<<"加速比为："<<time_stander/time_avx2<<endl;

// delete images;


}
std::cout<<"      "<<array[j]<<"      "<<"      "<<time_stander/1000000<<"         "<<"        "<<time_avx2/1000000<< "       "<< "        "<<time_stander/time_avx2<<endl;


}

}








//     // ---------------------------------------------------------
//     // *                ** P2 多线程 **                 *
//     // ---------------------------------------------------------

{

    cout<<"----------接下来进行多线程正确性测试----------"<<endl;
    Mat dst;
    Mat dst2;

    double time = 0;
    resize_parallel(image, dst, fx, fy,time);

    cv::imwrite("../P1_resized_SUSTech_para.jpg", dst);


    resize_parallel_avx2(image, dst2, fx, fy,time);

    cv::imwrite("../P1_resized_SUSTech_AVX2_para.jpg", dst2); // 修正为写入dst2







    compareMats(dst,dst2);



}

{
     cout<<"----------下面进行多线程加速比测试------------"<<endl;



    std::cout<<"下面进行resize多线程算法是否使用simd加速的加速比测试。具体来说,我们随机生成100张图片,分别交由"<<
    "标准算法和simd加速算法处理"<<endl;
    std::cout<<"图片尺寸统一选取800*800,缩放比依次为10,5,1,0.5,0.2"<<endl;
    float array [5]={10,5,1,0.5,0.2};
 std::cout<<"----缩放比----"<<"标准算法处理时间(ms)----"<<"simd加速算法处理时间(ms)----"<<"加速比----"<<endl;
    // 参数定义
    for (int j = 0; j<5;j++){
    const int image_size = 800;
    const int num_images = 100;
    const double fx = array[j]; // 缩放因子，可以根据需要调整
    const double fy = array[j];

    // 生成随机图片并存储在vector中
    double time_stander = 0;
    double time_avx2 = 0;

    for(int i = 0; i < num_images; ++i) {
        Mat images_2(image_size, image_size, CV_8UC3);

        // 生成随机颜色值
        cv::randu(images_2, cv::Scalar::all(0), cv::Scalar::all(255));



    // 变量用于存储结果
    cv::Mat dst, dst_avx2;

    // 记录标准resize的总时间


        resize_parallel(images_2, dst, fx, fy,time_stander);

        resize_parallel_avx2(images_2, dst_avx2, fx, fy,time_avx2);




    // 输出时间对比
    // std::cout << "标准resize总时间: " << time_stander/1000000 << " ms." << std::endl;
    // std::cout << "AVX2优化resize总时间: " << time_avx2/1000000 << " ms." << std::endl;

    // 输出加速比

    // std::cout<<"加速比为："<<time_stander/time_avx2<<endl;

// delete images_2;


}
std::cout<<"      "<<array[j]<<"      "<<"      "<<time_stander/1000000<<"         "<<"        "<<time_avx2/1000000<< "       "<< "        "<<time_stander/time_avx2<<endl;


}

}

{
    cout << "----------接下来开始双线性插值正确性实验----------------------" << endl;

    Mat dst;
    Mat dst2;

    double time_bilinear = 0;  // 记录双线性插值的时间
    double time_avx2 = 0;      // 记录AVX2加速的双线性插值的时间

    Mat image = cv::imread("/root/CPP_Project/Project/SUSTech.jpg", cv::IMREAD_COLOR);
    Mat dst0;
    resize_org(image, dst0, 10, 10, time_bilinear); // 假设 resize_org 是原始实现

    // 测试双线性插值
    auto start_bilinear = chrono::high_resolution_clock::now();
    resize_bilinear(dst0, dst, fx, fy, time_bilinear);
    auto end_bilinear = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_bilinear = end_bilinear - start_bilinear;
    time_bilinear = elapsed_bilinear.count();

    cv::imwrite("../P1_resized_SUSTech_bilinear.jpg", dst);

    // 测试AVX2加速的双线性插值
    auto start_avx2 = chrono::high_resolution_clock::now();
    resize_bilinear_avx2(dst0, dst2, fx, fy, time_avx2);
    auto end_avx2 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_avx2 = end_avx2 - start_avx2;
    time_avx2 = elapsed_avx2.count();

    cv::imwrite("../P1_resized_SUSTech_AVX2_bilinear.jpg", dst2); // 修正为写入dst2

    // 比较两个结果矩阵
    compareMats(dst2, dst2);



}

{
     cout<<"----------下面进行多线程加速比测试------------"<<endl;



    std::cout<<"下面进行resize多线程算法是否使用simd加速的加速比测试。具体来说,我们随机生成100张图片,分别交由"<<
    "标准算法和simd加速算法处理"<<endl;
    std::cout<<"图片尺寸统一选取800*800,缩放比依次为10,5,1,0.5,0.2"<<endl;
    float array [5]={10,5,1,0.5,0.2};
 std::cout<<"----缩放比----"<<"标准算法处理时间(ms)----"<<"simd加速算法处理时间(ms)----"<<"加速比----"<<endl;
    // 参数定义
    for (int j = 0; j<5;j++){
    const int image_size = 800;
    const int num_images = 100;
    const double fx = array[j]; // 缩放因子，可以根据需要调整
    const double fy = array[j];

    // 生成随机图片并存储在vector中
    double time_stander = 0;
    double time_avx2 = 0;

    for(int i = 0; i < num_images; ++i) {
        Mat images_2(image_size, image_size, CV_8UC3);

        // 生成随机颜色值
        cv::randu(images_2, cv::Scalar::all(0), cv::Scalar::all(255));



    // 变量用于存储结果
    cv::Mat dst, dst_avx2;

    // 记录标准resize的总时间


        resize_bilinear(images_2, dst, fx, fy,time_stander);

        resize_bilinear_avx2(images_2, dst_avx2, fx, fy,time_avx2);




    // 输出时间对比
    // std::cout << "标准resize总时间: " << time_stander/1000000 << " ms." << std::endl;
    // std::cout << "AVX2优化resize总时间: " << time_avx2/1000000 << " ms." << std::endl;

    // 输出加速比

    // std::cout<<"加速比为："<<time_stander/time_avx2<<endl;

// delete images_2;


}
std::cout<<"      "<<array[j]<<"      "<<"      "<<time_stander/1000000<<"         "<<"        "<<time_avx2/1000000<< "       "<< "        "<<time_stander/time_avx2<<endl;


}

}


 return 0;
}

void compareMats(const Mat& mat1, const Mat& mat2) {
    // 首先检查尺寸和类型
    if (mat1.size() != mat2.size() || mat1.type() != mat2.type()) {
        cout << "不同" << endl;
        cout << "mat1 尺寸: " << mat1.size() << ", 类型: " << mat1.type() << endl;
        cout << "mat2 尺寸: " << mat2.size() << ", 类型: " << mat2.type() << endl;
        return;
    }

    // 使用 cv::norm 计算两个图像之间的差异
    double diff = norm(mat1, mat2, NORM_INF);
    if (diff == 0) {
        cout << "对比结果相同" << endl;
    } else {
        cout << "不同" << endl;
        cout << "最大差异值: " << diff << endl;

        // 遍历矩阵，找出不匹配的像素坐标
        for (int y = 0; y < mat1.rows; ++y) {
            for (int x = 0; x < mat1.cols; ++x) {
                if (mat1.channels() == 1) {
                    if (mat1.at<uchar>(y, x) != mat2.at<uchar>(y, x)) {
                        cout << "坐标 (" << y << ", " << x << ") 不同: "
                             << "mat1 = " << (int)mat1.at<uchar>(y, x) << ", "
                             << "mat2 = " << (int)mat2.at<uchar>(y, x) << endl;
                    }
                } else if (mat1.channels() == 3) {
                    Vec3b pixel1 = mat1.at<Vec3b>(y, x);
                    Vec3b pixel2 = mat2.at<Vec3b>(y, x);
                    if (pixel1 != pixel2) {
                        cout << "坐标 (" << y << ", " << x << ") 不同: "
                             << "mat1 = (" << (int)pixel1[0] << ", " << (int)pixel1[1] << ", " << (int)pixel1[2] << "), "
                             << "mat2 = (" << (int)pixel2[0] << ", " << (int)pixel2[1] << ", " << (int)pixel2[2] << ")" << endl;
                    }
                }
                // 根据需要添加更多通道类型
            }
        }
    }
}

