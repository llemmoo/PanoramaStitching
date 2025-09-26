#include "PanoramaStitching/PanoramaStitcher.h"
#include <iostream>
#include <opencv2/core/utility.hpp>

using namespace std;

void showMainMenu()
{
    cout << "\n===== Panorama Stitching Application =====\n";
    cout << "1. Run program\n";
    cout << "0. Exit\n";
    cout << "Select an option: ";
}

void showDetectorMenu()
{
    cout << "\n--- Select Feature Detector ---\n";
    cout << "1. ORB\n";
    cout << "2. AKAZE\n";
    cout << "Select detector: ";
}

void showImageSetMenu()
{
    cout << "\n--- Select Image Set ---\n";
    cout << "1. Brick\n";
    cout << "2. Car\n";
    cout << "3. Stairs\n";
    cout << "Select image set: ";
}

int main()
{
    cout << "Assignment 1: Feature Detection, Matching, and Panorama Stitching with Experimental Evaluation \n\n" << endl;
    string chosenDetector;
    string chosenImagePair;
    while (true)
    {
        showMainMenu();
        int choice;
        cin >> choice;

        if (choice == 0)
        {
            cout << "Exiting...\n";
            break;
        }

        if (choice >= 1 && choice <= 3)
        {
            // Select detector
            showDetectorMenu();
            int detector_choice;
            cin >> detector_choice;

            PanoramaStitching::Detector detector;
            if (detector_choice == 1)
            {
                detector = PanoramaStitching::ORB;
                chosenDetector = "ORB";
            }
            else if (detector_choice == 2)
            {
                detector = PanoramaStitching::AKAZE;
                chosenDetector = "AKAZE";
            }
            else {
                cout << "Invalid detector choice.\n";
                continue;
            }

            // Choose an image pair
            showImageSetMenu();
            int dataset_choice;
            cin >> dataset_choice;

            PanoramaStitching::ImagePair dataset;
            if (dataset_choice == 1)
            {
                dataset = PanoramaStitching::brick;
                chosenImagePair = "brick";
            }
            if (dataset_choice == 2)
            {
                dataset = PanoramaStitching::car;
                chosenImagePair = "car";
            }
            if(dataset_choice == 3)
            {
                dataset = PanoramaStitching::stair;
                chosenImagePair = "stairs";
            }
            else
            {
                cout << "Invalid image set choice.\n";
                continue;
            }

            cout << "\n[Feature Detection] using " << chosenDetector
                 << " on dataset: " << chosenImagePair << endl;
            PanoramaStitching::PanoramaStitcher::run_panorama_stitcher(detector, dataset);
        }
        else
        {
            cout << "Invalid choice, try again.\n";
        }
    }
    return 0;
}
