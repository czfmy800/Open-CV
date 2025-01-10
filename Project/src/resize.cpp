#include "resize.h"

/// @brief 最近邻插值
/// @param src 源图像
/// @param dst 目标图像
/// @param fx x 方向的缩放比例，i/o
/// @param fy y 方向的缩放比例，i/o
void resize(const Mat& src, Mat& dst, double fx, double fy) {
    // 获取源图像的尺寸和通道数
    int src_rows = src.rows;
    int src_cols = src.cols;
    int src_channels = src.channels();

    // 计算目标图像的尺寸
    int dst_rows = round(src_rows / fy);
    int dst_cols = round(src_cols / fx);

    // 更新缩放比例
    // fx = src_cols / static_cast<double>(dst_cols);
    // fy = src_rows / static_cast<double>(dst_rows);
    
    // 创建目标图像，类型与源图像相同
    dst.create(dst_rows, dst_cols, src.type());

    // 遍历目标图像中的每个像素
    for (int i = 0; i < dst_rows; ++i) {
        for (int j = 0; j < dst_cols; ++j) {
            // 计算在源图像中的对应位置
            int src_i = static_cast<int>(i * fy);
            int src_j = static_cast<int>(j * fx);

            // 边界检查，确保不越界
            src_i = min(max(src_i, 0), src_rows - 1);
            src_j = min(max(src_j, 0), src_cols - 1);

            // 处理单通道和多通道图像
            if (src_channels == 1) {
                // 单通道图像（灰度图像）
                dst.at<uchar>(i, j) = src.at<uchar>(src_i, src_j);
            } else {
                // 多通道图像（例如RGB图像）
                for (int c = 0; c < src_channels; ++c) {
                    dst.at<Vec3b>(i, j)[c] = src.at<Vec3b>(src_i, src_j)[c];
                }
            }
        }
    }
}

// ! 并行优化
void resize_parallel(const Mat& src, Mat& dst, double fx, double fy) {
    // 获取源图像的尺寸和通道数
    int src_rows = src.rows;
    int src_cols = src.cols;
    int src_channels = src.channels();

    // 计算目标图像的尺寸
    int dst_rows = round(src_rows / fy);
    int dst_cols = round(src_cols / fx);
    
    // 创建目标图像，类型与源图像相同
    dst.create(dst_rows, dst_cols, src.type());

    // 使用并行化处理大图像
    cv::parallel_for_(cv::Range(0, dst_rows), [&](const cv::Range& r) {
        // cout << "Thread processing rows: " << r.start << " to " << r.end - 1 << endl;
        for (int i = r.start; i < r.end; ++i) {
            for (int j = 0; j < dst_cols; ++j) {
                // 计算在源图像中的对应位置
                int src_i = static_cast<int>(i * fy);
                int src_j = static_cast<int>(j * fx);
                // 边界检查
                src_i = std::clamp(src_i, 0, src_rows - 1);
                src_j = std::clamp(src_j, 0, src_cols - 1);
                
                // 处理单通道和多通道图像
                if (src_channels == 1) {
                    dst.at<uchar>(i, j) = src.at<uchar>(src_i, src_j);
                } else {
                    dst.at<Vec3b>(i, j) = src.at<Vec3b>(src_i, src_j);
                }
            }
        }
    });
}

void resize_bilinear(const Mat& src, Mat& dst, double fx, double fy) {
    // 获取源图像的尺寸和通道数
    int src_rows = src.rows;
    int src_cols = src.cols;
    int src_channels = src.channels();

    // 计算目标图像的尺寸
    int dst_rows = static_cast<int>(src_rows / fy);
    int dst_cols = static_cast<int>(src_cols / fx);
    
    // 创建目标图像，类型与源图像相同
    dst.create(dst_rows, dst_cols, src.type());

    cv::parallel_for_(cv::Range(0, dst_rows), [&](const cv::Range& r) {
        // cout << "Thread processing rows: " << r.start << " to " << r.end - 1 << endl;
        for (int i = r.start; i < r.end; ++i) {
            for (int j = 0; j < dst_cols; ++j) {
            // 计算在源图像中的对应位置
            double src_i = (i+0.5)*fx-0.5;
            double src_j = (j+0.5)*fy-0.5;

            // 计算双线性插值的四个最近邻点
            int x1 = static_cast<int>(src_i);
            int y1 = static_cast<int>(src_j);
            int x2 = x1+1;
            int y2 = y1+1;

            // 边界检查，确保不越界
            x1 = min(max(x1, 0), src_rows - 1);
            y1 = min(max(y1, 0), src_cols - 1);
            x2 = min(max(x2, 0), src_rows - 1);
            y2 = min(max(y2, 0), src_cols - 1); 
            
            // 计算双线性插值权重
            double w_x2 = src_i - x1;
            double w_x1 = 1-w_x2;
            double w_y2 = src_j - y1;
            double w_y1 = 1-w_y2;
            
            // 处理单通道和多通道图像
            if (src_channels == 1) {
                // 单通道图像（灰度图像）
                double I_y1 = w_x1*src.at<uchar>(x1, y1) + w_x2*src.at<uchar>(x2, y1);
                double I_y2 = w_x1*src.at<uchar>(x1, y2) + w_x2*src.at<uchar>(x2, y2);
                dst.at<uchar>(i, j) = static_cast<uchar>(w_y1*I_y1 + w_y2*I_y2);
            } else {
                // 多通道图像（例如RGB图像）
                for (int c = 0; c < src_channels; ++c) {
                    double I_y1 = w_x1*src.at<Vec3b>(x1, y1)[c] + w_x2*src.at<Vec3b>(x2, y1)[c];
                    double I_y2 = w_x1*src.at<Vec3b>(x1, y2)[c] + w_x2*src.at<Vec3b>(x2, y2)[c];
                    dst.at<Vec3b>(i, j)[c] = static_cast<uchar>(w_y1*I_y1 + w_y2*I_y2);
                }
            }
            }
        }
    });
}

void Opencv_resize_test_case(string img_path, string save_opencv_path, string save_resize_path, double fx, double fy){
    Mat image = cv::imread(img_path);
    Mat dst1;
    cv::resize(image,dst1,cv::Size(),1/fx,1/fy,0);
    cv::imwrite(save_opencv_path, dst1);
    Mat dst2;
    resize_parallel(image,dst2,fx,fy);
    cv::imwrite(save_resize_path, dst2);


    // 检查尺寸是否一致
    if (dst1.size() != dst2.size()) {
        cerr << "ERROR: The sizes of the two images do not match." << endl;
        cerr << "dst1 size: " << dst1.size() << endl;
        cerr << "dst2 size: " << dst2.size() << endl;
        return;
    }

    // 检查类型是否一致
    if (dst1.type() != dst2.type()) {
        cerr << "ERROR: The types of the two images do not match." << endl;
        return;
    }

    // 使用cv::compare函数比较两张图片
    cv::Mat diff;
    cv::compare(dst1, dst2, diff, cv::CMP_NE);

   // 将差异矩阵转换为单通道
    cv::Mat diff_gray;
    cv::cvtColor(diff, diff_gray, cv::COLOR_BGR2GRAY);

    // 计算不同像素的数量
    int numDiffPixels = cv::countNonZero(diff_gray);

    if (numDiffPixels == 0)
    {
        cout << "MATCH: two images are same" << endl;
    } else {
        cout << "ERROR: two images are different" << endl;
    }
    
}
