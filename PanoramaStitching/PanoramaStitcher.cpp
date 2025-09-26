#include "PanoramaStitcher.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"

using namespace PanoramaStitching;
using namespace cv;
using namespace std;

void PanoramaStitcher::run_panorama_stitcher(Detector chosenDetector, ImagePair imagePair) {
    //bruteforce can be used for both ORB and AKAZE
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);

    chrono::microseconds brickMatchDuration, carMatchDuration, stairMatchDuration;

    vector<string> imagePaths;

    Ptr<Feature2D> detector;

    switch (chosenDetector)
    {
        case ORB:
            detector = ORB::create(5000);
            break;
        case AKAZE:
            detector = AKAZE::create();
            break;
    }

    switch (imagePair)
    {
        case brick:
            imagePaths = vector<string>{"../images/brick1.jpg", "../images/brick2.jpg"};
            break;
        case car:
            imagePaths = vector<string>{"../images/car1.jpg", "../images/car2.jpg"};
            break;
        case stair:
            imagePaths = vector<string>{"../images/stair1.jpg", "../images/stair2.jpg"};
            break;
    }

    for (int i = 0; i < imagePaths.size()-1; i+=2) {
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

        vector<KeyPoint> keypointsFirstImage, keypointsSecondImage;
        Mat descriptorFirstImage, descriptorsSecondImage;

        auto startDetectAndComputeFirstImage = std::chrono::system_clock::now();
        detector->detectAndCompute(img1, noArray(), keypointsFirstImage, descriptorFirstImage);
        auto stopDetectAndComputeFirstImage = std::chrono::system_clock::now();


        auto startDetectAndComputeSecondImage = std::chrono::system_clock::now();
        detector->detectAndCompute(img2, noArray(), keypointsSecondImage, descriptorsSecondImage);
        auto stopDetectAndComputeSecondImage = std::chrono::system_clock::now();

        auto detectAndComputeDurationFirstImage = duration_cast<chrono::milliseconds>(stopDetectAndComputeFirstImage - startDetectAndComputeFirstImage);
        auto detectAndComputeDurationSecondImage = duration_cast<chrono::milliseconds>(stopDetectAndComputeSecondImage - startDetectAndComputeSecondImage);

        Mat detectionImage1;
        Mat detectionImage2;
        drawKeypoints(img1, keypointsFirstImage, detectionImage1, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);
        drawKeypoints(img2, keypointsSecondImage, detectionImage2, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);

        imshow("First image keypoints" + imagePaths[i], detectionImage1);
        imshow("Second Image Keypoints" + imagePaths[i+1], detectionImage2);

        cout << imagePaths[i] <<": \n" <<
            "\t Detected keypoints = " << keypointsFirstImage.size() << ", compute time = " << detectAndComputeDurationFirstImage << endl;
        cout << imagePaths[i+1] <<": \n" <<
            "\t Detected keypoint = " << keypointsSecondImage.size() << ", compute time = " << detectAndComputeDurationSecondImage << endl;

        //Histogram plotting of distances
        std::vector< std::vector<DMatch> > descriptorKnnMatches;
        auto descriptorMatchingStart = std::chrono::system_clock::now();
        matcher->knnMatch( descriptorFirstImage, descriptorsSecondImage, descriptorKnnMatches, 2 );
        auto descriptorMatchingStop = std::chrono::system_clock::now();

        vector<float> distances;
        for (auto &m : descriptorKnnMatches) {
            distances.push_back(m.data()->distance);
        }

        //Histogram plotting
        double maxVal = *max_element(distances.begin(), distances.end());
        int histSize = 50; // number of bins
        vector<int> bins(histSize, 0);
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
        drawMatches( img1, keypointsFirstImage, img2, keypointsSecondImage, descriptorKnnMatches, img_matches );

        imshow("Matches of " + imagePaths[i] + " & " + imagePaths[i+1], img_matches);

        if(imagePair == brick)
        {
            brickMatchDuration = duration_cast<chrono::microseconds>(descriptorMatchingStop - descriptorMatchingStart);
            cout << "Brick Match Runtime = " << brickMatchDuration << endl;
        }
        if(imagePair == car)
        {
            carMatchDuration = duration_cast<chrono::microseconds>(descriptorMatchingStop - descriptorMatchingStart);
            cout << "Car Match Runtime = " << carMatchDuration << endl;
        }
        if(imagePair == stair)
        {
            stairMatchDuration = duration_cast<chrono::microseconds>(descriptorMatchingStop - descriptorMatchingStart);
            cout << "Stair Match Runtime = " << stairMatchDuration << endl;
        }

        /**
         *
         *  HOMOGRAPHY ESTIMATION
         *
         * */

        //Lowe's ratio to find better matches set to 0.8 like in slides
        const float ratio_thresh = 0.8f;
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < descriptorKnnMatches.size(); i++)
        {
            if (descriptorKnnMatches[i][0].distance < ratio_thresh * descriptorKnnMatches[i][1].distance)
            {
                good_matches.push_back(descriptorKnnMatches[i][0]);
            }
        }

        std::vector<Point2f> goodKeypointsFirstImage;
        std::vector<Point2f> goodKeypointsSecondImage;
        for( size_t i = 0; i < good_matches.size(); i++ )
        {
            //Get the keypoints from the good matches
            goodKeypointsFirstImage.push_back( keypointsFirstImage[ good_matches[i].queryIdx ].pt );
            goodKeypointsSecondImage.push_back( keypointsSecondImage[ good_matches[i].trainIdx ].pt );
        }

        //We output a inlier mask to find the number of inliers and experiment with the ransacReprojThreshold
        Mat inlierMask1;
        Mat inlierMask3;
        Mat inlierMask5;
        Mat inlierMask10;

        auto startThreshold1Runtime = std::chrono::system_clock::now();
        Mat H1 = findHomography( goodKeypointsFirstImage, goodKeypointsSecondImage, RANSAC, 1, inlierMask1);
        auto stopThreshold1Runtime = std::chrono::system_clock::now();

        auto startThreshold3Runtime = std::chrono::system_clock::now();
        Mat H3 = findHomography( goodKeypointsFirstImage, goodKeypointsSecondImage, RANSAC, 3, inlierMask3);
        auto stopThreshold3Runtime = std::chrono::system_clock::now();

        auto startThreshold5Runtime = std::chrono::system_clock::now();
        Mat H5 = findHomography( goodKeypointsFirstImage, goodKeypointsSecondImage, RANSAC, 5, inlierMask5);
        auto stopThreshold5Runtime = std::chrono::system_clock::now();

        auto startThreshold10Runtime = std::chrono::system_clock::now();
        Mat H10 = findHomography( goodKeypointsFirstImage, goodKeypointsSecondImage, RANSAC, 10, inlierMask10);
        auto stopThreshold10Runtime = std::chrono::system_clock::now();


        int numberOfInliers1 = countNonZero(inlierMask1);
        int numberOfInliers3 = countNonZero(inlierMask3);
        int numberOfInliers5 = countNonZero(inlierMask5);
        int numberOfInliers10 = countNonZero(inlierMask10);

        auto threshold1Runtime = duration_cast<chrono::microseconds>(stopThreshold1Runtime - startThreshold1Runtime);
        auto threshold3Runtime = duration_cast<chrono::microseconds>(stopThreshold3Runtime - startThreshold3Runtime);
        auto threshold5Runtime = duration_cast<chrono::microseconds>(stopThreshold5Runtime - startThreshold5Runtime);
        auto threshold10Runtime = duration_cast<chrono::microseconds>(stopThreshold10Runtime - startThreshold10Runtime);

        cout << "Number of inliers for threshold 1: " << numberOfInliers1 << "| Runtime: " << threshold1Runtime << endl;
        cout << "Number of inliers for threshold 3: " << numberOfInliers3 << "| Runtime: " << threshold3Runtime <<  endl;
        cout << "Number of inliers for threshold 5: " << numberOfInliers5 << "| Runtime: " << threshold5Runtime << endl;
        cout << "Number of inliers for threshold 10: " << numberOfInliers10 << "| Runtime: " << threshold10Runtime << endl;

        Mat result1;
        Mat result3;
        Mat result5;
        Mat result10;
        warpPerspective(img1, result1, H1, Size(img2.cols, img2.rows));
        warpPerspective(img1, result3, H3, Size(img2.cols, img2.rows));
        warpPerspective(img1, result5, H5, Size(img2.cols, img2.rows));
        warpPerspective(img1, result10, H10, Size(img2.cols, img2.rows));

        // Creating overlay to map the warped perspective onto the second image, letting us evaluate the stitch quality of the homography
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
        vector<Point2f> corners1 =
        {
            Point2f(0,0), Point2f(img1.cols,0),
            Point2f(img1.cols,img1.rows), Point2f(0,img1.rows)
        };

        vector<Point2f> warpedCorners1;
        perspectiveTransform(corners1, warpedCorners1, H5);

        vector<Point2f> corners2 =
        {
            Point2f(0,0), Point2f(img2.cols,0),
            Point2f(img2.cols,img2.rows), Point2f(0,img2.rows)
        };

        vector<Point2f> allCorners = warpedCorners1;
        allCorners.insert(allCorners.end(), corners2.begin(), corners2.end());

        Rect bbox = boundingRect(allCorners);

        Mat translation =
        (Mat_<double>(3,3) << 1, 0, -bbox.x,
                                        0, 1, -bbox.y,
                                        0, 0, 1);

        Mat panorama(bbox.height, bbox.width, img1.type());
        warpPerspective(img1, panorama, translation * H5, panorama.size());

        Mat roi(panorama, Rect(-bbox.x, -bbox.y, img2.cols, img2.rows));
        img2.copyTo(roi);

        imshow("Panorama of " + imagePaths[i] + " & " + imagePaths[i+1], panorama);
    }
        cout << "Press any key to close windows..." << endl;
        waitKey(0);
        destroyAllWindows();
}