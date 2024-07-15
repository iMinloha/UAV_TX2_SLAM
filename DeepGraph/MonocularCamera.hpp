#ifndef DEEPVISION_MONOCULARCAMERA_HPP
#define DEEPVISION_MONOCULARCAMERA_HPP

#include <fstream>
#include "opencv2/opencv.hpp"
#include "DeepException.hpp"

using namespace std;
using namespace cv;
using namespace DeepVision;


// ****************************************************
/*** �����Camera
 * @param cameraID: ���ID
 * @param params: ����ڲ�
 * @param distortion: ����������
 * @param imageSize: ͼ��ߴ�
 * @param map1, map2: �������ӳ��
 * @param R, T: ��ת�����ƽ�ƾ���
 * @param frame: ��ǰ֡
 * @param cap: VideoCapture����
 * @param isOpened: ����Ƿ��
 * @param status: ��ȡ���״̬
 * @param getFrame: ��ȡ֡(�����������)
 * @param showFrame: ��ʾ֡
 * @param release: �ͷ����
 * @note: 1. �޲ι��캯��, 2. �вι��캯��, 3. ��csv�ļ��ж�ȡ����
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
    // �������
    /*** ע��
     * @brief params: ����ڲ�, 3x3����
     * @brief distortion: �������, 1x5����
     * @brief Projection: ͶӰ����, 3x4����(��Ҫȷ������ڲκͻ��������֪)
     * @brief map1, map2: �������ӳ��(��Ҫȷ������ڲκͻ��������֪)
     * @brief R, T: ��ת�����ƽ�ƾ���, ��Ŀ�����ȡR��t��Ҫ��ȡ������֡�ı��ʾ������λ��(�궨��õ�R��t������ڵ�һ֡��)
     * @brief imageSize: ͼ��ߴ�
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
    // �޲ι���
    MonocularCamera();

    // �вι���, �������ID�Լ��������(params, distortion, imageSize)���㹻��
    MonocularCamera(int cameraID, Mat params, Mat distortion, Size imageSize) : cameraID(cameraID), params(params), distortion(distortion), imageSize(imageSize){
        cap.open(cameraID);
        if(!cap.isOpened()) throw DeepException("Camera open failed!");
        isOpened = true;
        initUndistortRectifyMap(params, distortion, R, params, imageSize, CV_32FC1, map1, map2);
    }

    // ��csv�ļ��ж�ȡ����
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

    // ״̬����
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
