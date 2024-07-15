#ifndef DEEPVISION_BINOCULARCAMERA_HPP
#define DEEPVISION_BINOCULARCAMERA_HPP

#include "opencv2/opencv.hpp"
#include "DeepException.hpp"
#include <iostream>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;

class BinocularCamera {
public:
    Mat E, F, Q, R1, R2, P1, P2;
    Mat map1x, map1y, map2x, map2y;

private:
    /*** 参数
     * @param left_translation 左摄像头平移矩阵
     * @param right_translation 右摄像头平移矩阵
     * @param left_camera_matrix 左摄像头内参矩阵
     * @param right_camera_matrix 右摄像头内参矩阵
     * @param left_distortion_coefficients 左摄像头畸变矩阵
     * @param right_distortion_coefficients 右摄像头畸变矩阵
     * @param R 旋转矩阵
     * @param T 平移矩阵
     * @param E 本质矩阵
     * @param F 基础矩阵
     * @param projectionMatrix1 左投影矩阵
     * @param projectionMatrix2 右投影矩阵
     * @param left_camera 左摄像头图像
     * @param right_camera 右摄像头图像
     * @param camera 摄像头
     * @param image_size 图像大小
     * */
    Mat left_translation, right_translation;
    Mat left_camera_matrix, right_camera_matrix;
    Mat left_distortion_coefficients, right_distortion_coefficients;
    Mat R, T;
    Mat projectionMatrix1, projectionMatrix2;
    Mat left_camera, right_camera;
    VideoCapture camera;
    Size image_size;

    /*** 更新摄像头图像
     * @param frame 摄像头图像
     * @param left_camera 左摄像头图像
     * @param right_camera 右摄像头图像
     * */
    void _UpdateCameraPicture() {
        Mat frame;
        camera.read(frame);

        left_camera = frame(Rect(0, 0, frame.cols / 2, frame.rows));
        right_camera = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
    }


    /*** 计算参数
     * @param Q 重投影矩阵
     * @param R1 左立体校正映射矩阵
     * @param R2 右立体校正映射矩阵
     * @param P1 左投影矩阵
     * @param P2 右投影矩阵
     * @param map1x 左映射矩阵x
     * @param map1y 左映射矩阵y
     * @param map2x 右映射矩阵x
     * @param map2y 右映射矩阵y
     * */
    void _CalculateParam(){
        if (_IsInitParam()){
            // 计算重投影矩阵
            stereoRectify(left_camera_matrix, left_distortion_coefficients,
                          right_camera_matrix, right_distortion_coefficients,
                          image_size, R, T, R1, R2, P1, P2, Q);
            // 计算映射矩阵
            initUndistortRectifyMap(left_camera_matrix, left_distortion_coefficients, R1, P1, image_size, CV_32F, map1x, map1y);
            initUndistortRectifyMap(right_camera_matrix, right_distortion_coefficients, R2, P2, image_size, CV_32F, map2x, map2y);
        }
    }

private:
    // 摄像头是否打开
    bool _IsCameraOpened() {
        return camera.isOpened();
    }

    // 参数是否初始化
    bool _IsInitParam() {
        return !left_camera_matrix.empty() && !right_camera_matrix.empty() && !left_distortion_coefficients.empty() && !right_distortion_coefficients.empty() && !R.empty() && !T.empty();
    }

public:
    // 构造函数
    BinocularCamera() = default;

    // 构造函数
    BinocularCamera(int camera_id, int width, int height) {
        camera = VideoCapture(camera_id);
        camera.set(CAP_PROP_FRAME_WIDTH, width);
        camera.set(CAP_PROP_FRAME_HEIGHT, height);
        while (!camera.isOpened())
            std::cout << "video is ending!" << std::endl;
        image_size = Size(width, height);
    }

    // 构造函数
    BinocularCamera(Mat left_camera_matrix, Mat right_camera_matrix, Mat left_distortion_coefficients, Mat right_distortion_coefficients,
           Mat R, Mat T, int camera_id, int width, int height) : BinocularCamera(camera_id, width, height) {
        this->left_camera_matrix = left_camera_matrix;
        this->right_camera_matrix = right_camera_matrix;
        this->left_distortion_coefficients = left_distortion_coefficients;
        this->right_distortion_coefficients = right_distortion_coefficients;
        this->R = R;
        this->T = T;
    }

    // 从文件中读取摄像头参数
    BinocularCamera(String filename, int camera_id, int width, int height) : BinocularCamera(camera_id, width, height ){
        ifstream file(filename);
        string line;
        camera = VideoCapture(camera_id);
        camera.set(CAP_PROP_FRAME_WIDTH, width);
        camera.set(CAP_PROP_FRAME_HEIGHT, height);

        while (getline(file, line)) {
            stringstream ss(line);
            string key;
            ss >> key;
            if (key == "T") {
                vector<double> values(3);
                for (int i = 0; i < 3; i++) ss >> values[i];
                T = Mat(values).reshape(1, 3);
            } else if (key == "R") {
                vector<double> values(9);
                for (int i = 0; i < 9; i++) ss >> values[i];
                R = Mat(values).reshape(1, 3);
            } else if (key == "left_camera_matrix") {
                vector<double> values(9);
                for (int i = 0; i < 9; i++) ss >> values[i];
                left_camera_matrix = Mat(values).reshape(1, 3);
            } else if (key == "left_distortion_coefficients") {
                vector<double> values(5);
                for (int i = 0; i < 5; i++) ss >> values[i];
                left_distortion_coefficients = Mat(values).reshape(1, 1);
            } else if (key == "right_camera_matrix") {
                vector<double> values(9);
                for (int i = 0; i < 9; i++) ss >> values[i];
                right_camera_matrix = Mat(values).reshape(1, 3);
            } else if (key == "right_distortion_coefficients") {
                vector<double> values(5);
                for (int i = 0; i < 5; i++) ss >> values[i];
                right_distortion_coefficients = Mat(values).reshape(1, 1);
            }
            else if (key == "width") ss >> image_size.width;
            else if (key == "height") ss >> image_size.height;
        }

    }

    // 更新摄像头图像
    void UpdateCameraPicture() {
        _UpdateCameraPicture();
    }

    // 计算参数
    void CalculateParam() {
        _CalculateParam();
    }

    // 摄像头是否打开
    bool IsCameraOpened() {
        return _IsCameraOpened();
    }

    // 参数是否初始化
    bool IsInitParam() {
        return _IsInitParam();
    }

    // 获取左摄像头图像
    Mat GetLeftCamera() {
        return left_camera;
    }

    // 获取右摄像头图像
    Mat GetRightCamera() {
        return right_camera;
    }
};

#endif
