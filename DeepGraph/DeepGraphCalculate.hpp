#ifndef DEEPVISION_DEEPGRAPHCALCULATE_HPP
#define DEEPVISION_DEEPGRAPHCALCULATE_HPP

#include "opencv2/opencv.hpp"
#include "BinocularCamera.hpp"

// 双目相机获取深度图
/*** BinDepthMap
 * @param camera 摄像头
 * @param depth_map 深度图
 * @Methods
 *      CalculateDepthMap 计算深度图
 *      FillHole 空洞值掩膜
 *      CalculatePointCloud 计算点云
 * */
class BinDepthMap : public BinocularCamera {
private:
    BinocularCamera camera;

public:
    // 深度图
    Mat depth_map;
    // 彩色深度图
    Mat color_depth_map;
    // 灰度深度图
    Mat gray_depth_map;

    // 构造函数
    BinDepthMap() = default;

    // 构造函数
    BinDepthMap(BinocularCamera camera) : camera(std::move(camera)) {}

    void CalculateDepthMap() {
        if (camera.IsInitParam()) {
            Mat left_camera = camera.GetLeftCamera();
            Mat right_camera = camera.GetRightCamera();
            // 图像缩小一倍, 加快计算速度(对于SGBM算法)
            resize(left_camera, left_camera, Size(left_camera.cols / 2, left_camera.rows / 2));
            resize(right_camera, right_camera, Size(right_camera.cols / 2, right_camera.rows / 2));

            Mat left_camera_gray, right_camera_gray;
            cvtColor(left_camera, left_camera_gray, COLOR_BGR2GRAY);
            cvtColor(right_camera, right_camera_gray, COLOR_BGR2GRAY);
            Mat disparity_map;
            // 使用SGBM算法, 相机基线为60mm, 视差范围为64

            int minDisparity = 0;
            int numDisparities = 60;  //max disparity - min disparity
            int blockSize = 3;
            int P1 = 8 * blockSize*blockSize;
            int P2 = 32 *blockSize*blockSize;
            int disp12MaxDiff = 1;
            int preFilterCap = 63;
            int uniquenessRatio = 10;
            int speckleWindowSize = 200;
            int speckleRange = 32;
            int mode = StereoSGBM::MODE_SGBM;

            Ptr<StereoSGBM> stereo = StereoSGBM::create(minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, mode);
            stereo->compute(left_camera_gray, right_camera_gray, disparity_map);
            disparity_map.convertTo(depth_map, CV_32F, 1.0 / 16);

            // 空洞值掩膜
            FillHole();

            Mat imgDispMap8U = Mat(depth_map.rows, depth_map.cols, CV_8U);
            double max, min;
            Point minLoc, maxLoc;
            minMaxLoc(depth_map, &min, &max, &maxLoc, &minLoc);
            double alpha = 255.0 / (max - min);
            depth_map.convertTo(imgDispMap8U, CV_8U, alpha, -alpha * min);
            cv::Mat colorisedDispMap;
            cv::applyColorMap(imgDispMap8U, color_depth_map, cv::COLORMAP_JET);

            // 灰度深度图(每个人的喜好不同, 可以不要颜色的depthMap)
            cvtColor(color_depth_map, gray_depth_map, COLOR_BGR2GRAY);
        }else{
            throw DeepVision::DeepException("Camera parameter is not initialized!");
        }
    }

    // 空洞值掩膜
    void FillHole() {
        // SGBM空洞填充
        Mat hole_mask;
        // 通过阈值处理得到空洞掩膜
        threshold(depth_map, hole_mask, 0, 185, THRESH_BINARY);
        // 覆盖后转为8位图(2 ^ 8 - 1 = 255)也就是正常色彩
        hole_mask.convertTo(hole_mask, CV_8U);
        // 闭运算填充空洞
        Mat kernel = getStructuringElement(MORPH_RECT, Size(8, 8));
        morphologyEx(hole_mask, hole_mask, MORPH_CLOSE, kernel);
        // 反转掩膜(空洞为0, 非空洞为255)
        hole_mask = 255 - hole_mask;
        // 修复空洞, 使用TELEA算法
        inpaint(depth_map, hole_mask, depth_map, 6, INPAINT_TELEA);
    }
};


// TODO: 未完成, 未测试
class CameraCalibration : public BinocularCamera {
private:
    BinocularCamera camera;

public:
    CameraCalibration() = default;

    CameraCalibration(BinocularCamera camera) : camera(std::move(camera)) {}

    void CalibrateCamera(string param_path, int camera_id){
        if(camera.IsCameraOpened()){
            // 双目标定 25mm小方格纸
            Size board_size = Size(9, 6);
            vector<vector<Point2f>> left_image_points, right_image_points;
            vector<vector<Point3f>> object_points;
            vector<Point2f> left_corners, right_corners;
            vector<Point3f> object_point;
            for (int i = 0; i < board_size.height; i++) {
                for (int j = 0; j < board_size.width; j++) {
                    object_point.push_back(Point3f(j * 25, i * 25, 0));
                }
            }
            Mat left_camera_matrix, right_camera_matrix, left_distortion_coefficients, right_distortion_coefficients, R, T;
            Mat R1, R2, P1, P2, Q;
            Mat map1x, map1y, map2x, map2y;
            Size image_size;
            while (true) {
                camera.UpdateCameraPicture();
                Mat left_camera = camera.GetLeftCamera();
                Mat right_camera = camera.GetRightCamera();
                Mat left_camera_gray, right_camera_gray;
                cvtColor(left_camera, left_camera_gray, COLOR_BGR2GRAY);
                cvtColor(right_camera, right_camera_gray, COLOR_BGR2GRAY);
                bool left_found = findChessboardCorners(left_camera_gray, board_size, left_corners);
                bool right_found = findChessboardCorners(right_camera_gray, board_size, right_corners);
                if (left_found && right_found) {
                    left_image_points.push_back(left_corners);
                    right_image_points.push_back(right_corners);
                    object_points.push_back(object_point);
                    drawChessboardCorners(left_camera, board_size, left_corners, left_found);
                    drawChessboardCorners(right_camera, board_size, right_corners, right_found);
                    imshow("left_camera", left_camera);
                    imshow("right_camera", right_camera);
                    waitKey(100);
                }
                if (left_image_points.size() == 10) {
                    calibrateCamera(object_points, left_image_points, left_camera.size(), left_camera_matrix, left_distortion_coefficients, R, T);
                    calibrateCamera(object_points, right_image_points, right_camera.size(), right_camera_matrix, right_distortion_coefficients, R, T);
                    stereoCalibrate(object_points, left_image_points, right_image_points, left_camera_matrix, left_distortion_coefficients, right_camera_matrix, right_distortion_coefficients, left_camera.size(), R, T, E, F);
                    stereoRectify(left_camera_matrix, left_distortion_coefficients, right_camera_matrix, right_distortion_coefficients, left_camera.size(), R, T, R1, R2, P1, P2, Q);
                    initUndistortRectifyMap(left_camera_matrix, left_distortion_coefficients, R1, P1, left_camera.size(), CV_32F, map1x, map1y);
                    initUndistortRectifyMap(right_camera_matrix, right_distortion_coefficients, R2, P2, right_camera.size(), CV_32F, map2x, map2y);
                    image_size = left_camera.size();
                    break;
                }
            }
            ofstream file(param_path);
            file << "left_camera " << left_camera_matrix.reshape(1, 1) << endl;
            file << "left_distort " << left_distortion_coefficients.reshape(1, 1) << endl;
            file << "right_camera " << right_camera_matrix.reshape(1, 1) << endl;
            file << "right_distort " << right_distortion_coefficients.reshape(1, 1) << endl;
            file << "R " << R.reshape(1, 1) << endl;
            file << "T " << T.reshape(1, 1) << endl;
            file << "width " << image_size.width << endl;
            file << "height " << image_size.height << endl;
            file.close();
        }
    }
};

// 单目相机获取深度图
class MonoDepthMap : public MonocularCamera {
private:
    MonocularCamera camera;

public:
    MonoDepthMap() = default;
    MonoDepthMap(MonocularCamera camera) : camera(std::move(camera)) {}

private:
    Mat E;
    Mat DepthMap, ColorMap;

public:
    void CalculateDepthMap(Mat lastFrame) {
        // 计算本质矩阵E
        Mat frame;
        camera.getFrame(frame);
        Mat frame_gray;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        Mat mask = Mat::zeros(frame.size(), CV_8U);
        mask(Rect(0, 0, frame.cols, frame.rows)) = 255;

        // 本质矩阵E
        E = findEssentialMat(frame_gray, lastFrame, camera.params, RANSAC, 0.999, 1.0, mask);
        Mat R, T;
        recoverPose(E, frame_gray, lastFrame, camera.params, R, T);

        // 计算图片间的orb特征点并进行三角测量
        vector<KeyPoint> keyPoints1, keyPoints2;
        Mat descriptors1, descriptors2;
        Ptr<ORB> orb = ORB::create();
        orb->detectAndCompute(lastFrame, mask, keyPoints1, descriptors1);
        orb->detectAndCompute(frame, mask, keyPoints2, descriptors2);
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        vector<DMatch> matches;
        matcher->match(descriptors1, descriptors2, matches);
        vector<Point2f> points1, points2;
        for (int i = 0; i < matches.size(); i++) {
            points1.push_back(keyPoints1[matches[i].queryIdx].pt);
            points2.push_back(keyPoints2[matches[i].trainIdx].pt);
        }
        Mat points4D;
        triangulatePoints(camera.params * Mat::eye(3, 4, CV_64F), Mat::eye(3, 4, CV_64F), points1, points2, points4D);
        Mat points3D;
        convertPointsFromHomogeneous(points4D.t(), points3D);

        // 计算深度图
        DepthMap = Mat::zeros(frame.size(), CV_32F);
        for (int i = 0; i < points3D.rows; i++) {
            Point3f point = points3D.at<Point3f>(i);
            DepthMap.at<float>(point.y, point.x) = point.z;
        }

        // 空洞值掩膜
        FillHole();

        // 彩色深度图
        ColorMap = Mat::zeros(frame.size(), CV_8UC3);
        double min, max;
        minMaxLoc(DepthMap, &min, &max);
        for (int i = 0; i < DepthMap.rows; i++) {
            for (int j = 0; j < DepthMap.cols; j++) {
                ColorMap.at<Vec3b>(i, j) = Vec3b(255 * (DepthMap.at<float>(i, j) - min) / (max - min), 0, 0);
            }
        }
    }

    void FillHole(){
        // 空洞值掩膜
        Mat hole_mask;
        // 通过阈值处理得到空洞掩膜
        threshold(DepthMap, hole_mask, 0, 185, THRESH_BINARY);
        // 覆盖后转为8位图(2 ^ 8 - 1 = 255)也就是正常色彩
        hole_mask.convertTo(hole_mask, CV_8U);
        // 闭运算填充空洞
        Mat kernel = getStructuringElement(MORPH_RECT, Size(8, 8));
        morphologyEx(hole_mask, hole_mask, MORPH_CLOSE, kernel);
        // 反转掩膜(空洞为0, 非空洞为255)
        hole_mask = 255 - hole_mask;
        // 修复空洞, 使用TELEA算法
        inpaint(DepthMap, hole_mask, DepthMap, 6, INPAINT_TELEA); }
};

#endif
