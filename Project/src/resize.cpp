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
    int dst_rows = static_cast<int>(src_rows / fy);
    int dst_cols = static_cast<int>(src_cols / fx);
    
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
    int dst_rows = static_cast<int>(src_rows / fy);
    int dst_cols = static_cast<int>(src_cols / fx);
    
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

// ? const 意义不大
void resize_parallel_const(const Mat& src, Mat& dst, const double fx, const double fy) {
    const int src_rows = src.rows;
    const int src_cols = src.cols;
    const int src_channels = src.channels();

    const int dst_rows = static_cast<int>(src_rows / fy);
    const int dst_cols = static_cast<int>(src_cols / fx);
    
    dst.create(dst_rows, dst_cols, src.type());

    cv::parallel_for_(cv::Range(0, dst_rows), [&](const cv::Range& r) {
        for (int i = r.start; i < r.end; ++i) {
            for (int j = 0; j < dst_cols; ++j) {
                int src_i = static_cast<int>(i * fy);
                int src_j = static_cast<int>(j * fx);
                src_i = std::clamp(src_i, 0, src_rows - 1);
                src_j = std::clamp(src_j, 0, src_cols - 1);
                
                if (src_channels == 1) {
                    dst.at<uchar>(i, j) = src.at<uchar>(src_i, src_j);
                } else {
                    dst.at<Vec3b>(i, j) = src.at<Vec3b>(src_i, src_j);
                }
            }
        }
    });
}




