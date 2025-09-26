#ifndef FEATUREDETECTOR_H
#define FEATUREDETECTOR_H
namespace PanoramaStitching
{
    enum Detector
    {
        ORB,
        AKAZE
    };
    enum ImagePair
    {
        brick,
        car,
        stair
    };
    class PanoramaStitcher
    {
        public:
        static void run_panorama_stitcher(Detector detector, ImagePair imagePair);
    };
}
#endif
