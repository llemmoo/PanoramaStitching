#include "FeatureDetection/FeatureDetector.h"
#include "ImageStitching/ImageStitcher.h"
#include <iostream>
#include <string>
#include <vector>

int main() {
    while (true) {
        std::cout << "\n=== Image Tool ===\n";
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
            std::string filename;
            std::cout << "Enter image filename: ";
            std::cin >> filename;

            // Example call
            std::cout << "Running feature detection on images";
            // auto features = detector.detectFeatures(loadImage(filename));
            // std::cout << "Found " << features.size() << " features.\n";
        }
        else if (choice == 2) {
            int numImages;

            // auto panorama = stitcher.stitch(loadImages(files));
            // saveImage("result.jpg", panorama);
            // std::cout << "Saved result to result.jpg\n";
        }
        else {
            std::cout << "Invalid choice, try again.\n";
        }
    }

    return 0;
}