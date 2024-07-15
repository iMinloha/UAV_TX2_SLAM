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
    /*** ����
     * @param left_translation ������ͷƽ�ƾ���
     * @param right_translation ������ͷƽ�ƾ���
     * @param left_camera_matrix ������ͷ�ڲξ���
     * @param right_camera_matrix ������ͷ�ڲξ���
     * @param left_distortion_coefficients ������ͷ�������
     * @param right_distortion_coefficients ������ͷ�������
     * @param R ��ת����
     * @param T ƽ�ƾ���
     * @param E ���ʾ���
     * @param F ��������
     * @param projectionMatrix1 ��ͶӰ����
     * @param projectionMatrix2 ��ͶӰ����
     * @param left_camera ������ͷͼ��
     * @param right_camera ������ͷͼ��
     * @param camera ����ͷ
     * @param image_size ͼ���С
     * */
    Mat left_translation, right_translation;
    Mat left_camera_matrix, right_camera_matrix;
    Mat left_distortion_coefficients, right_distortion_coefficients;
    Mat R, T;
    Mat projectionMatrix1, projectionMatrix2;
    Mat left_camera, right_camera;
    VideoCapture camera;
    Size image_size;

    /*** ��������ͷͼ��
     * @param frame ����ͷͼ��
     * @param left_camera ������ͷͼ��
     * @param right_camera ������ͷͼ��
     * */
    void _UpdateCameraPicture() {
        Mat frame;
        camera.read(frame);

        left_camera = frame(Rect(0, 0, frame.cols / 2, frame.rows));
        right_camera = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
    }


    /*** �������
     * @param Q ��ͶӰ����
     * @param R1 ������У��ӳ�����
     * @param R2 ������У��ӳ�����
     * @param P1 ��ͶӰ����
     * @param P2 ��ͶӰ����
     * @param map1x ��ӳ�����x
     * @param map1y ��ӳ�����y
     * @param map2x ��ӳ�����x
     * @param map2y ��ӳ�����y
     * */
    void _CalculateParam(){
        if (_IsInitParam()){
            // ������ͶӰ����
            stereoRectify(left_camera_matrix, left_distortion_coefficients,
                          right_camera_matrix, right_distortion_coefficients,
                          image_size, R, T, R1, R2, P1, P2, Q);
            // ����ӳ�����
            initUndistortRectifyMap(left_camera_matrix, left_distortion_coefficients, R1, P1, image_size, CV_32F, map1x, map1y);
            initUndistortRectifyMap(right_camera_matrix, right_distortion_coefficients, R2, P2, image_size, CV_32F, map2x, map2y);
        }
    }

private:
    // ����ͷ�Ƿ��
    bool _IsCameraOpened() {
        return camera.isOpened();
    }

    // �����Ƿ��ʼ��
    bool _IsInitParam() {
        return !left_camera_matrix.empty() && !right_camera_matrix.empty() && !left_distortion_coefficients.empty() && !right_distortion_coefficients.empty() && !R.empty() && !T.empty();
    }

public:
    // ���캯��
    BinocularCamera() = default;

    // ���캯��
    BinocularCamera(int camera_id, int width, int height) {
        camera = VideoCapture(camera_id);
        camera.set(CAP_PROP_FRAME_WIDTH, width);
        camera.set(CAP_PROP_FRAME_HEIGHT, height);
        while (!camera.isOpened())
            std::cout << "video is ending!" << std::endl;
        image_size = Size(width, height);
    }

    // ���캯��
    BinocularCamera(Mat left_camera_matrix, Mat right_camera_matrix, Mat left_distortion_coefficients, Mat right_distortion_coefficients,
           Mat R, Mat T, int camera_id, int width, int height) : BinocularCamera(camera_id, width, height) {
        this->left_camera_matrix = left_camera_matrix;
        this->right_camera_matrix = right_camera_matrix;
        this->left_distortion_coefficients = left_distortion_coefficients;
        this->right_distortion_coefficients = right_distortion_coefficients;
        this->R = R;
        this->T = T;
    }

    // ���ļ��ж�ȡ����ͷ����
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

    // ��������ͷͼ��
    void UpdateCameraPicture() {
        _UpdateCameraPicture();
    }

    // �������
    void CalculateParam() {
        _CalculateParam();
    }

    // ����ͷ�Ƿ��
    bool IsCameraOpened() {
        return _IsCameraOpened();
    }

    // �����Ƿ��ʼ��
    bool IsInitParam() {
        return _IsInitParam();
    }

    // ��ȡ������ͷͼ��
    Mat GetLeftCamera() {
        return left_camera;
    }

    // ��ȡ������ͷͼ��
    Mat GetRightCamera() {
        return right_camera;
    }
};

#endif
