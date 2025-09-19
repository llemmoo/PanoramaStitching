#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("C:/Users/Oliver/CLionProjects/PanoramaStitching/test.jpg");

    if (image.empty()) {

    std::cerr << "Error: Image could not be loaded." << std::endl;
        return -1;
    }
    // Create a window for image display.
    cv::namedWindow("Image Display", cv::WINDOW_AUTOSIZE);
    // Render the image in the created window.
    cv::imshow("Image Display", image);
    // Wait indefinitely for a user key press.
    cv::waitKey(0);
    // Release all resources.
    cv::destroyAllWindows();
    return 0;
}
