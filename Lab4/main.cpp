#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <sys/stat.h>

#include <cv.h>
#include <highgui.h>
#include <vector>
#include <cxcore.h>
#include <time.h>

using namespace cv;
using namespace std;
// define mode for stitching as panorama
Stitcher::Mode mode = Stitcher::PANORAMA;

void vector<Mat> load_Images(const String &dirname, vector<Mat> &img_lst, bool showImages = true);
void show_Images(vector<Mat> &img_lst);

int main(int argc, char const *argc[])
{
    load_Images("/Users/taomenglu/Unipd2022/CV/LAB/Lab4/data/T1", img_lst);
    doStitching(img_lst);
    return 0;
}

// implementation of the function, which will be called by the main function

// 1.load all the images into a vector
void vector<Mat> load_Images(const String &dirname, vector<Mat> &img_lst, bool showImages = true)
{
    vector<String> files;
    glob(dirname, files);
    for (size_t i = 0; i < files.size(); i++)
    {
        Mat img = imread(files[i], IMREAD_UNCHANGED);
        if (img.empty())
        {
            cout << "Error loading image " << files[i] << endl;
            continue;
        }
        if (showImages)
        {
            imshow("image", img);
            waitKey(1);
        }
        img_lst.push_back(img);
    }
    return img_lst;
}

void doStitching(vector<Mat> &img_lst)
{
    Mat pano;
    ptr<Stitcher> stitcher = Stitcher::create(mode, false);
    Stitcher::Status status = stitcher->stitch(img_lst, pano);

    if (status != Stitcher::OK)
    {
        cout << "Can't stitch images, error code = " << int(status) << endl;
        return;
    }
    imwrite("result.jpg", pano);
    imshow("Result", pano);
    waitKey(0);
    return 0;
}

void doSIFT(vector<Mat> &img_lst, Mat &result)
{
    // 2.define paramenters of SIFT alogrithm
    int numFeatures = 500; // defind the number of features

    // 3.create detector put into Keypoint
    Ptr<SIFT> detector = SIFT::create(numFeatures);
    int i = 0; // index of keypoint
    for (i; i < 10; i++)

    {
        vector<KeyPoint> keypoints[i]; // define the keypoint
        detector->detect(img_lst[i], keypoints[i]);
        // print the keypoints
        cout << "the number of keypoints is " << keypoints[i].size() << endl;
        // draw the keypoints
        Mat drawsrc[i];
        drawKeypoints(img_lst[i], keypoints[i], drawsrc[i]);
        CvUtils::SetShowWindow(drawsrc[i], "drawsrc", 10, 20);
        imshow("drawsrc", drawsrc[i]); // show the keypoints
        waitKey(1);

        // calculate the feature descriptors needed for matching
        Mat dst[i];
        Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create(); // create the extractor
        descriptor->compute(img_lst[i], keypoints[i], dst[i]);
        cout << "the number of descriptors is " << dst[i].size() << endl;

        for (int j = i + 1; j < 10; j++)
        {
            // 4.match the descriptors
            BFMatcher matcher(NORM_L2);
            vector<DMatch> matches;
            matcher.match(dst[i], dst[j], matches);
            cout << "the number of matches is " << matches.size() << endl;

            // 5.draw the matches
            Mat drawmatch[i];
            drawMatches(img_lst[i], keypoints[i], img_lst[j], keypoints[j], matches, drawmatch[i]);
            CvUtils::SetShowWindow(drawmatch[i], "drawmatch", 10, 20);
            imshow("drawmatch", drawmatch[i]); // show the matches
            waitKey(1);

            // 6.find the minimum distance
            double min_dist = 10000, max_dist = 0;
            for (int i = 0; i < dst[i].rows; i++) // find the minimum distance
            {
                double dist = matches[i].distance;
                if (dist < min_dist)
                    min_dist = dist;
                if (dist > max_dist)
                    max_dist = dist;
            }
            cout << "the minimum distance is " << min_dist << endl;
            cout << "the maximum distance is " << max_dist << endl;

            // 7.find the good matches
            vector<DMatch> good_matches;
            for (int i = 0; i < matches.size(); ++i)
            {
                double dist = matches[i].distance;
                if (dist < 2 * min_dist)
                    goodMatches.push_back(matches[i]);
            }
            cout << "the number of good matches is " << good_matches.size() << endl;
            Mat result;

            // 8.draw the result. match points are blue, and other points are random color
            drawMatches(img_lst[i], keypoints[i], img_lst[j], keypoints[j], good_matches, result, Scalar(255, 255, 0), Scalar::all(-1));
            CvUtils::SetShowWindow(result, "result", 100, 20);
            imshow("result", result);
            waitKey(1);

            // 9.find the homography matrix
            vector<Point2f> p01, p02;
        }
    }
}