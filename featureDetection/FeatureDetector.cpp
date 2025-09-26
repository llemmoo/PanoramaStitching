#include "FeatureDetector.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"

using namespace FeatureDetection;
using namespace cv;
using namespace std;

void feature_detector::run_feature_detection() {
    //When nfeatures is not set for ORB it caps at 500, which akaze does not. Therefore, I'm setting orb at 50000, since matches stopped around 40000
    Ptr<FeatureDetector> orb = ORB::create(50000);
    Ptr<FeatureDetector> akaze = AKAZE::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);

    chrono::microseconds akazeBrickMatchDuration, akazeCarMatchDuration, akazeStairMatchDuration;
    chrono::microseconds orbBrickMatchDuration, orbCarMatchDuration, orbStairMatchDuration;

    vector<string> imagePaths =
    {
        "../images/brick1.jpg",
        "../images/brick2.jpg",
        "../images/car1.jpg",
        "../images/car2.jpg",
        "../images/stair1.jpg",
        "../images/stair2.jpg"
    };

    // ORB
    cout << "ORB Experiments \n\n" << endl;
        for (int i = 0; i < imagePaths.size()-1; i+=2)
        {
            Mat img1 = imread(imagePaths[i], IMREAD_COLOR);
            Mat img2 = imread(imagePaths[i+1], IMREAD_COLOR);

            if (img1.empty() || img2.empty())
            {
                cerr << "Could not read images: " << endl;
                continue;
            }
            //Feature Matching
            vector<KeyPoint> keypoints_orb1, keypoints_orb2;
            Mat descriptors_orb1, descriptors_orb2;

            auto startOrb1 = std::chrono::system_clock::now();
            orb->detectAndCompute(img1, noArray(), keypoints_orb1, descriptors_orb1);
            auto stopOrb1 = std::chrono::system_clock::now();

            auto startOrb2 = std::chrono::system_clock::now();
            orb->detectAndCompute(img2, noArray(), keypoints_orb2, descriptors_orb2);
            auto stopOrb2 = std::chrono::system_clock::now();

            auto durationOrb1 = duration_cast<chrono::milliseconds>(stopOrb1 - startOrb1);
            auto durationOrb2 = duration_cast<chrono::milliseconds>(stopOrb2 - startOrb2);

            Mat img1_orb;
            Mat img2_orb;
            drawKeypoints(img1, keypoints_orb1, img1_orb, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);
            drawKeypoints(img2, keypoints_orb2, img2_orb, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);

            imshow("ORB - " + imagePaths[i], img1_orb);
            imshow("ORB - " + imagePaths[i+1], img2_orb);

            cout << imagePaths[i] <<": \n" <<
                "\t ORB keypoints = " << keypoints_orb1.size() << " ORB Feature Detection Execution Time = " << durationOrb1 << endl;
            cout << imagePaths[i+1] <<": \n" <<
                "\t ORB keypoints = " << keypoints_orb2.size() << " ORB Feature Detection Execution Time = " << durationOrb2 << endl;

            if (descriptors_orb1.empty() || descriptors_orb2.empty()) {
                cerr << "No descriptors found for one of the images: "
                     << imagePaths[i] << " or " << imagePaths[i+1] << endl;
                continue;
            }
            std::vector< DMatch > descriptorMatches;
            auto orbMatchStart = std::chrono::system_clock::now();
            matcher->match( descriptors_orb1, descriptors_orb2, descriptorMatches );
            auto orbMatchStop = std::chrono::system_clock::now();

            if(i == 0)
            {
                orbBrickMatchDuration = duration_cast<chrono::microseconds>(orbMatchStop - orbMatchStart);
            }
            if (i == 2)
            {
                orbCarMatchDuration = duration_cast<chrono::microseconds>(orbMatchStop - orbMatchStart);
            }
            if (i == 4)
            {
                orbStairMatchDuration = duration_cast<chrono::microseconds>(orbMatchStop - orbMatchStart);
            }

            Mat img_matches;
            drawMatches( img1, keypoints_orb1, img2, keypoints_orb2, descriptorMatches, img_matches );

            imshow("Orb Matches of" + imagePaths[i] + " & " + imagePaths[i+1], img_matches);
        }

        cout << "\n AKAZE Experiments \n\n" << endl;
        // AKAZE
        for (int i = 0; i < imagePaths.size()-1; i+=2)
        {
            /**
             *
             *  FEATURE DESCRIPTION AND DESCRIPTOR MATCHING
             *
             * */
            Mat img1 = imread(imagePaths[i], IMREAD_COLOR);
            Mat img2 = imread(imagePaths[i+1], IMREAD_COLOR);

            if (img1.empty() || img2.empty())
            {
                cerr << "Could not read images: " << endl;
                continue;
            }

            vector<KeyPoint> keypoints_akaze1, keypoints_akaze2;
            Mat descriptors_akaze1, descriptors_akaze2;

            auto startAkaze1 = std::chrono::system_clock::now();
            akaze->detectAndCompute(img1, noArray(), keypoints_akaze1, descriptors_akaze1);
            auto stopAkaze1 = std::chrono::system_clock::now();

            auto startAkaze2 = std::chrono::system_clock::now();
            akaze->detectAndCompute(img2, noArray(), keypoints_akaze2, descriptors_akaze2);
            auto stopAkaze2 = std::chrono::system_clock::now();

            auto durationAkaze1 = duration_cast<chrono::milliseconds>(stopAkaze1 - startAkaze1);
            auto durationAkaze2 = duration_cast<chrono::milliseconds>(stopAkaze2 - startAkaze2);

            Mat img1_akaze;
            Mat img2_akaze;
            drawKeypoints(img1, keypoints_akaze1, img1_akaze, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);
            drawKeypoints(img2, keypoints_akaze2, img2_akaze, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);

            imshow("AKAZE - " + imagePaths[i], img1_akaze);
            imshow("AKAZE - " + imagePaths[i+1], img2_akaze);

            cout << imagePaths[i] <<": \n" <<
                "\t AKAZE keypoints = " << keypoints_akaze1.size() << " AKAZE Feature Detection Execution Time = " << durationAkaze1 << endl;
            cout << imagePaths[i+1] <<": \n" <<
                "\t AKAZE keypoints = " << keypoints_akaze2.size() << " AKAZE Feature Detection Execution Time = " << durationAkaze2 << endl;

            //Histogram plotting of distances


            std::vector< std::vector<DMatch> > descriptorKnnMatches;
            auto akazeMatchStart = std::chrono::system_clock::now();
            matcher->knnMatch( descriptors_akaze1, descriptors_akaze2, descriptorKnnMatches, 2 );
            auto akazeMatchStop = std::chrono::system_clock::now();

            vector<float> distances;
            for (auto &m : descriptorKnnMatches) {
                distances.push_back(m.data()->distance);
            }

            //OpenCV conversion

            // Find max distance to set range
            double maxVal = *max_element(distances.begin(), distances.end());

            // Histogram parameters
            int histSize = 50; // number of bins
            vector<int> bins(histSize, 0);

            // Fill bins manually
            for (auto d : distances) {
                int bin = cvFloor((d / maxVal) * (histSize - 1));
                bins[bin]++;
            }

            //we draw an image the size of 512 by 400 to show our histogram
            int hist_w = 512, hist_h = 400;
            int bin_w = cvRound((double) hist_w / histSize);
            Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0,0,0));

            int maxCount = *max_element(bins.begin(), bins.end());
            for (int i = 1; i < histSize; i++) {
                line(histImage,
                     Point(bin_w*(i-1), hist_h - cvRound(((double)bins[i-1]/maxCount)*hist_h)),
                     Point(bin_w*(i),   hist_h - cvRound(((double)bins[i]/maxCount)*hist_h)),
                     Scalar(255, 0, 0), 2, 8, 0);
            }

            imshow("Histogram of Match Distances for: " + imagePaths[i] + " & " + imagePaths[i+1], histImage);

            Mat img_matches;
            drawMatches( img1, keypoints_akaze1, img2, keypoints_akaze2, descriptorKnnMatches, img_matches );

            imshow("Akaze Matches of" + imagePaths[i] + " & " + imagePaths[i+1], img_matches);

            if(i == 0)
            {
                akazeBrickMatchDuration = duration_cast<chrono::microseconds>(akazeMatchStop - akazeMatchStart);
            }
            if (i == 2)
            {
                akazeCarMatchDuration = duration_cast<chrono::microseconds>(akazeMatchStop - akazeMatchStart);
            }
            if (i == 4)
            {
                akazeStairMatchDuration = duration_cast<chrono::microseconds>(akazeMatchStop - akazeMatchStart);
            }

            /**
             *
             *  HOMOGRAPHY ESTIMATION
             *
             * */

            //Lowe's ratio to find better matches
            const float ratio_thresh = 0.75f;
            std::vector<DMatch> good_matches;
            for (size_t i = 0; i < descriptorKnnMatches.size(); i++)
            {
                if (descriptorKnnMatches[i][0].distance < ratio_thresh * descriptorKnnMatches[i][1].distance)
                {
                    good_matches.push_back(descriptorKnnMatches[i][0]);
                }
            }

            std::vector<Point2f> img1akaze;
            std::vector<Point2f> img2akaze;
            for( size_t i = 0; i < good_matches.size(); i++ )
            {
                //Get the keypoints from the good matches
                img1akaze.push_back( keypoints_akaze1[ good_matches[i].queryIdx ].pt );
                img2akaze.push_back( keypoints_akaze2[ good_matches[i].trainIdx ].pt );
            }

            //We output a inlier mask to find the number of inliers and experiment with the ransacReprojThreshhold
            Mat inlierMask1;
            Mat inlierMask3;
            Mat inlierMask5;
            Mat inlierMask10;

            Mat H1 = findHomography( img1akaze, img2akaze, RANSAC, 1, inlierMask1);
            Mat H3 = findHomography( img1akaze, img2akaze, RANSAC, 3, inlierMask3);
            Mat H5 = findHomography( img1akaze, img2akaze, RANSAC, 5, inlierMask5);
            Mat H10 = findHomography( img1akaze, img2akaze, RANSAC, 10, inlierMask10);

            int numberOfInliers1 = countNonZero(inlierMask1);
            int numberOfInliers3 = countNonZero(inlierMask3);
            int numberOfInliers5 = countNonZero(inlierMask5);
            int numberOfInliers10 = countNonZero(inlierMask10);

            cout << "Number of inliers for threshold 1: " << numberOfInliers1 << endl;
            cout << "Number of inliers for threshold 3: " << numberOfInliers3 << endl;
            cout << "Number of inliers for threshold 5: " << numberOfInliers5 << endl;
            cout << "Number of inliers for threshold 10: " << numberOfInliers10 << endl;

            Mat result1;
            Mat result3;
            Mat result5;
            Mat result10;
            warpPerspective(img1, result1, H1, Size(img2.cols, img2.rows));
            warpPerspective(img1, result3, H3, Size(img2.cols, img2.rows));
            warpPerspective(img1, result5, H5, Size(img2.cols, img2.rows));
            warpPerspective(img1, result10, H10, Size(img2.cols, img2.rows));

            Mat overlay1;
            Mat overlay3;
            Mat overlay5;
            Mat overlay10;
            addWeighted(result1, 0.5, img2, 0.5, 0.0, overlay1);
            addWeighted(result3, 0.5, img2, 0.5, 0.0, overlay3);
            addWeighted(result5, 0.5, img2, 0.5, 0.0, overlay5);
            addWeighted(result10, 0.5, img2, 0.5, 0.0, overlay10);
            imshow("Overlay of: " + imagePaths[i] + " & " + imagePaths[i+1] + " ransac 1", overlay1);
            imshow("Overlay of: " + imagePaths[i] + " & " + imagePaths[i+1] + " ransac 3", overlay3);
            imshow("Overlay of: " + imagePaths[i] + " & " + imagePaths[i+1] + " ransac 5", overlay5);
            imshow("Overlay of: " + imagePaths[i] + " & " + imagePaths[i+1] + " ransac 10", overlay10);


            /**
             *
             *  PANORAMA STITCHING
             *
             * */

        }

    cout << "Brick Descriptor Matching times: " << "AKAZE = " << akazeBrickMatchDuration << ", ORB = " << orbBrickMatchDuration << endl;
    cout << "Car Descriptor Matching times: " << "AKAZE = " << akazeCarMatchDuration << ", ORB = " << orbCarMatchDuration << endl;
    cout << "Stair Descriptor Matching times: " << "AKAZE = " << akazeStairMatchDuration << ", ORB = " << orbStairMatchDuration << endl;

    cout << "Press any key to close windows..." << endl;
    waitKey(0);
    destroyAllWindows();
}