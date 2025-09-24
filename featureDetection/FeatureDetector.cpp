#include "FeatureDetector.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"

using namespace FeatureDetection;
using namespace cv;
using namespace std;

void feature_detector::run_feature_detection() {
    Ptr<FeatureDetector> orb = ORB::create(50000);
    Ptr<FeatureDetector> akaze = AKAZE::create();

    vector<string> imagePaths =
    {
        "../images/brick1.jpg",
        "../images/brick2.jpg",
        "../images/car1.jpg",
        "../images/car2.jpg",
        "../images/stair1.jpg",
        "../images/stair2.jpg"
    };
    for (const string& path : imagePaths)
    {
        Mat img = imread(path, IMREAD_COLOR);
        if (img.empty())
        {
            cerr << "Could not read image: " << path << endl;
            continue;
        }

        vector<KeyPoint> keypoints_orb, keypoints_akaze;

        orb->detect(img, keypoints_orb);
        akaze->detect(img, keypoints_akaze);

        Mat img_orb, img_akaze;
        drawKeypoints(img, keypoints_orb, img_orb, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);
        drawKeypoints(img, keypoints_akaze, img_akaze, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);

        imshow("ORB - " + path, img_orb);
        imshow("AKAZE - " + path, img_akaze);

        cout << "Image " << path << ": ORB keypoints = " << keypoints_orb.size()
             << ", AKAZE keypoints = " << keypoints_akaze.size() << endl;
    }
    cout << "Press any key to close windows..." << endl;
    waitKey(0);
    destroyAllWindows();
}