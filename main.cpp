#include "FeatureDetection/FeatureDetector.h"
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/utility.hpp>

int main() {
    std::cout << "Runtime OpenCV: " << cv::getVersionString() << std::endl;
    FeatureDetection::feature_detector detector;

    while (true) {
        std::cout << "1. Feature Detection\n";
        std::cout << "2. Image Stitching\n";
        std::cout << "0. Exit\n";
        std::cout << "Select an option: ";

        int choice;
        std::cin >> choice;

        if (choice == 0) {
            std::cout << "Exiting...\n";
            break;
        }
        else if (choice == 1) {
            std::cout << "Running feature detection on images\n";
            detector.run_feature_detection();
        }
        else if (choice == 2) {
            int numImages;

        }
        else {
            std::cout << "Invalid choice, try again.\n";
        }
    }

    return 0;
}
