#ifndef DEEPVISION_MONOCULARCAMERA_HPP
#define DEEPVISION_MONOCULARCAMERA_HPP

#include <fstream>
#include "opencv2/opencv.hpp"
#include "DeepException.hpp"

using namespace std;
using namespace cv;
using namespace DeepVision;


// ****************************************************
/*** 相机类Camera
 * @param cameraID: 相机ID
 * @param params: 相机内参
 * @param distortion: 相机畸变参数
 * @param imageSize: 图像尺寸
 * @param map1, map2: 畸变矫正映射
 * @param R, T: 旋转矩阵和平移矩阵
 * @param frame: 当前帧
 * @param cap: VideoCapture对象
 * @param isOpened: 相机是否打开
 * @param status: 获取相机状态
 * @param getFrame: 获取帧(包含畸变矫正)
 * @param showFrame: 显示帧
 * @param release: 释放相机
 * @note: 1. 无参构造函数, 2. 有参构造函数, 3. 从csv文件中读取参数
 * @api: 1. Camera(), 2. Camera(int cameraID, Mat params, Mat distortion, Size imageSize), 3. Camera(int CameraID, String ParamsFile)
 * @api: 1. status(), 2. getFrame(Mat &dst), 3. getFrame(), 4. showFrame(), 5. release()
 * @example:
 *     Camera cam(0, params, distortion, imageSize);
 *     while(cam.status()){
 *          cam.getFrame();
 *          cam.showFrame();
 *     }
 * */
// ****************************************************
class MonocularCamera{
private:
    VideoCapture cap;
    Mat frame;
    int cameraID;
    bool isOpened;

private:
    // 相机参数
    /*** 注意
     * @brief params: 相机内参, 3x3矩阵
     * @brief distortion: 畸变参数, 1x5矩阵
     * @brief Projection: 投影矩阵, 3x4矩阵(需要确保相机内参和畸变参数已知)
     * @brief map1, map2: 畸变矫正映射(需要确保相机内参和畸变参数已知)
     * @brief R, T: 旋转矩阵和平移矩阵, 单目相机获取R与t需要获取连续两帧的本质矩阵计算位姿(标定获得的R与t是相对于第一帧的)
     * @brief imageSize: 图像尺寸
     * */
    Mat params, distortion, Projection;
    Mat map1, map2, R, T;
    Size imageSize;

private:
    void _UpdateFrame(){
        cap >> frame;
        if(frame.empty()) throw DeepException("Camera frame is empty!");
    }

    void _Undistort(Mat &src, Mat &dst){
        remap(src, dst, map1, map2, INTER_LINEAR);
    }

    void _getFrame(Mat &dst){
        _UpdateFrame();
        _Undistort(frame, dst);
    }

public:
    // 无参构造
    MonocularCamera();

    // 有参构造, 包含相机ID以及相机参数(params, distortion, imageSize)就足够了
    MonocularCamera(int cameraID, Mat params, Mat distortion, Size imageSize) : cameraID(cameraID), params(params), distortion(distortion), imageSize(imageSize){
        cap.open(cameraID);
        if(!cap.isOpened()) throw DeepException("Camera open failed!");
        isOpened = true;
        initUndistortRectifyMap(params, distortion, R, params, imageSize, CV_32FC1, map1, map2);
    }

    // 从csv文件中读取参数
    MonocularCamera(int CameraID, String ParamsFile){
        cameraID = CameraID;
        cap.open(cameraID);
        if(!cap.isOpened()) throw DeepException("Camera open failed!");
        isOpened = true;
        ifstream file(ParamsFile);
        string line;
        while (getline(file, line)) {
            stringstream ss(line);
            string key;
            ss >> key;
            if (key == "params") {
                vector<double> values(9);
                for (int i = 0; i < 9; i++) ss >> values[i];
                params = Mat(values).reshape(1, 3);
            } else if (key == "distortion") {
                vector<double> values(5);
                for (int i = 0; i < 5; i++) ss >> values[i];
                distortion = Mat(values).reshape(1, 1);
            } else if (key == "imageSize") {
                ss >> imageSize.width >> imageSize.height;
            }
        }
        initUndistortRectifyMap(params, distortion, R, params, imageSize, CV_32FC1, map1, map2);
    }

    // 状态函数
    bool status(){
        return isOpened;
    }

    void getFrame(Mat &dst){
        _getFrame(dst);
    }

    void getFrame(){
        _getFrame(frame);
    }

    void showFrame(){
        imshow("Camera", frame);
        waitKey(1);
    }

    void release(){
        cap.release();
        isOpened = false;
    }

    ~MonocularCamera(){
        if(isOpened) release();
    }
};

#endif
