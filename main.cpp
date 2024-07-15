#include "opencv2/opencv.hpp"
#include "DeepGraph/DeepGraph.h"

using namespace cv;
using namespace std;

int main() {
    BinocularCamera camera("camera.csv", 0, 3840, 1080);
    while (true){
        camera.UpdateCameraPicture();

        BinDepthMap depthMap(camera);
        // StereoImageFusion stereoImageFusion(camera);
        depthMap.CalculateDepthMap();
        imshow("Depth Map", depthMap.gray_depth_map);
        // stereoImageFusion.FusionImage();

        int key = waitKey(30);
        if (key == 27) break;
    }
}
