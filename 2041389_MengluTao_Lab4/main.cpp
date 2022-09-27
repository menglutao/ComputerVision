// ref: https://pythonmana.com/2021/08/20210803083628400F.html

#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <sstream>
#include <sys/stat.h>
#include <tuple>
#include <vector>
#include <time.h>
#include <algorithm>
#include <cstdio>
#include <random>

#define DO_RESIZE 0
#define DEFAULT_RATIO 0.5 // if the user does not provide ratio
#define NUM_FEATURES 20000

using namespace cv;
using namespace std;

vector<Mat> loadImages(const String &dirname);
void doInpaint(Mat &img);
string type2str(int type);
Mat doSIFT(vector<Mat> &imgList, string datasetName);
void stitchImages(Mat &img1, Mat &img2, vector<DMatch> &goodMatches, vector<KeyPoint> &keypoints_1, vector<KeyPoint> &keypoints_2, Mat &out, bool alreadyCalledRecursive, bool onlyHorizontalImages, vector<Mat> &horizontalImages, vector<Mat> &verticalImages);

double matchesRatio = DEFAULT_RATIO; // global variable with default value - it receives from the user if provided

int main(int argc, char const *argv[])
{
    // srand(time(NULL));
    if (argc > 1)
    {
        matchesRatio = stod(argv[1]); // stod = String TO Double
    }
    string datasetNames[] = {"T1", "T2", "T3", "RT1"}; // test files
    mkdir("./test_outputs", 0777);
    for (string datasetName : datasetNames)
    {
        string outputDir = "./test_outputs/" + datasetName;
        mkdir(outputDir.c_str(), 0777);
    }
    for (string datasetName : datasetNames)
    {
        vector<Mat> imgList = loadImages("data/" + datasetName);
        auto rng = default_random_engine{};
        shuffle(begin(imgList), end(imgList), rng); // shuffle the images
        try
        {
            Mat res = doSIFT(imgList, datasetName);
            string fileName = "./test_outputs/" + datasetName + "/final_stitched_image.png";
            cout << "\n\n---> Writting to file: " << fileName << "\n\n";
            imwrite(fileName, res); // write the final stitched image
        }
        catch (bool v)
        {
            cout << "\nFinal result not generated for dataset " << datasetName << "\n";
        }
    }

    return 0;
}

// 1.load all the images into a vector
vector<Mat> loadImages(const String &dirname)
{
    vector<String> files;
    vector<Mat> imgList;
    glob(dirname, files);
    int numImages = files.size(); // 3;//files.size();
    for (size_t i = 1; i < numImages; i++)
    {
        cout << "-> Loading image " << files[i] << " in position " << (i - 1) << "\n";
        Mat img = imread(files[i], IMREAD_UNCHANGED);
        if (img.empty())
        {
            cout << "Error loading image " << files[i] << endl;
            continue;
        }
        imgList.push_back(img);
    }
    if (DO_RESIZE) // resize the images if needed
    {
        Mat img = imgList[0];
        double width = img.cols, height = img.rows;
        for (size_t i = 1; i < imgList.size(); i++)
        {
            if (imgList[i].cols < width)
                width = imgList[i].cols;
            if (imgList[i].rows > height)
                height = imgList[i].rows;
        }

        for (size_t i = 0; i < imgList.size(); i++)
        {
            resize(imgList[i], imgList[i], Size(width, height), INTER_LINEAR);
        }
    }

    cout << "image list size: " << imgList.size() << endl;
    return imgList;
}

void computeKeypointsAndDescriptors(vector<Mat> &imgList, vector<KeyPoint> *keypoints, Mat *descriptors)
{
    Ptr<SIFT> detector = SIFT::create(NUM_FEATURES);
    Ptr<SiftDescriptorExtractor> descriptor = SiftDescriptorExtractor::create();

    int numImages = imgList.size();
    for (int i = 0; i < numImages; i++)
    {
        // detect keypoints and compute descriptors
        detector->detect(imgList[i], keypoints[i]);
        descriptor->compute(imgList[i], keypoints[i], descriptors[i]);
    }
}

void computeAllGoodMatches(int numImages, vector<KeyPoint> *keypoints, Mat *descriptors, vector<tuple<vector<DMatch>, int, int>> &allGoodMatches)
{
    for (int i = 0; i < numImages - 1; i++)
    {
        for (int j = i + 1; j < numImages; j++)
        {

            BFMatcher matcher(NORM_L2);
            vector<vector<DMatch>> matches;
            matcher.knnMatch(descriptors[i], descriptors[j], matches, 2);

            vector<DMatch> goodMatches;
            for (vector<DMatch> match : matches)
            {
                if (match[0].distance < matchesRatio * match[1].distance)
                {
                    goodMatches.push_back(match[0]);
                }
            }

            if (goodMatches.size() > 10) // if there are enough good matches
            {
                allGoodMatches.push_back(make_tuple(goodMatches, i, j));
            }
        }
    }

    sort(allGoodMatches.begin(), allGoodMatches.end(),
         [](const tuple<vector<DMatch>, int, int> &lhs, const tuple<vector<DMatch>, int, int> &rhs)
         {
             return get<0>(lhs).size() > get<0>(rhs).size(); // if it has more good matches, then it comes first
         });

    cout << "\n------\n";
    for (tuple<vector<DMatch>, int, int> v : allGoodMatches)
    { // for debbuging
        cout << "(" << get<1>(v);
        cout << ", " << get<2>(v) << ") ";
        cout << get<0>(v).size();
        if (get<0>(v).size() > 0)
        {
            cout << " --- queryIdx: " << get<0>(v).at(0).queryIdx;
            cout << " --- trainIdx: " << get<0>(v).at(0).trainIdx << "\n";
        }
    }
}

/*
 * @brief
 * This function is called when it is needed to swap images because findHomography method from OpenCV depends on the order of
 * the images. So, when its output (H) has negative values, then we call this function. However,
 * since it still can have negative values in a second call for images that are not supposed to be stitched,
 * then it throw an error to skip the computation for these two images, which is the correct behavior to do.
 */

void swapImagesAndStitch(Mat &img1, Mat &img2, Mat &out, vector<Mat> &horizontalImages, vector<Mat> &verticalImages, bool alreadyCalledRecursive, bool onlyHorizontalImages = false)
{

    vector<Mat> imgList;
    imgList.push_back(img2);
    imgList.push_back(img1);
    int numImages = imgList.size();
    vector<KeyPoint> keypoints[numImages];
    Mat descriptors[numImages];
    computeKeypointsAndDescriptors(imgList, keypoints, descriptors);

    vector<tuple<vector<DMatch>, int, int>> allGoodMatches;
    computeAllGoodMatches(numImages, keypoints, descriptors, allGoodMatches);
    vector<DMatch> &goodMatches_2 = get<0>(allGoodMatches[0]);
    stitchImages(img2, img1, goodMatches_2, keypoints[0], keypoints[1], out, alreadyCalledRecursive, onlyHorizontalImages, horizontalImages, verticalImages);
}

void stitchImages(Mat &img1, Mat &img2, vector<DMatch> &goodMatches, vector<KeyPoint> &keypoints_1, vector<KeyPoint> &keypoints_2, Mat &out, bool alreadyCalledRecursive, bool onlyHorizontalImages, vector<Mat> &horizontalImages, vector<Mat> &verticalImages)
{
    vector<Point2f> pointsImg_1;
    vector<Point2f> pointsImg_2;
    for (int j = 0; j < goodMatches.size(); j++)
    {
        pointsImg_1.push_back(keypoints_1[goodMatches[j].queryIdx].pt);
        pointsImg_2.push_back(keypoints_2[goodMatches[j].trainIdx].pt);
    }
    Mat mask;
    Mat H = findHomography(pointsImg_1, pointsImg_2, RANSAC, 3, mask);
    // cout << "\nMask: " << mask << "\n";
    cout << "\nH: " << H << "\n";
    if (H.cols == 0)
    {
        throw(true);
    }
    double width = H.at<double>(0, 2);
    double height = H.at<double>(1, 2);

    if (width < -30) // if the width is bigger than 30, then the images should be swapped.
    {
        if (alreadyCalledRecursive) // if already swapped, and still negative, then means this image shouldn't be stitched
        {
            throw(true);
        }
        {
            throw(alreadyCalledRecursive);
        }
        swapImagesAndStitch(img1, img2, out, horizontalImages, verticalImages, true, onlyHorizontalImages);
        return;
    }

    if (height < -30) // if the height is bigger than 30, then the images should be swapped.
    {
        if (alreadyCalledRecursive)
        {
            throw(alreadyCalledRecursive);
        }
        swapImagesAndStitch(img1, img2, out, horizontalImages, verticalImages, true, onlyHorizontalImages);
        return;
    }

    if (abs(width) > abs(height))
    { // horizontal image
        if (abs(height) > 100)
        {
            throw(true);
        }
        cout << "(horizontal) Dimensions width > height: ";
        cout << "\nimg1.cols: " << img1.cols;
        cout << "\nabs(width): " << abs(width);
        cout << "\nmax(img1.rows, img2.rows): " << max(img1.rows, img2.rows) << "\n";

        double rows = max(img1.rows, img2.rows);

        cout << "\nH: " << H << "\n";

        double cols_2 = img1.cols + abs(width);
        double cols = cols_2 < img2.cols ? img2.cols : cols_2;
        cout << "SIZE: " << cols << ", " << rows << "\n";
        cout << "roi1: " << img2.cols << ", " << img2.rows << "\n";
        warpPerspective(img1, out, H, Size(cols, rows)); // ref:

        Mat roi1(out, Rect(0, 0, img2.cols, img2.rows));
        img2.copyTo(roi1);

        // Mat roi2(out, Rect(width, 0, img1.cols, img1.rows));
        // img1.copyTo(roi2);
        //////// resize(out, out, Size(), 0.5, 0.5);
        horizontalImages.push_back(out);
    }
    else
    { // vertical image
        if (onlyHorizontalImages)
        {
            throw(onlyHorizontalImages);
        }
        if (abs(width) > 100)
        {
            throw(true);
        }

        cout << "(vertical) Dimensions width <= height: ";
        cout << "\nmax(img1.cols, img2.cols): " << max(img1.cols, img2.cols);
        cout << "\nimg1.rows: " << img1.rows;
        cout << "\nabs(height): " << abs(height) << "\n";

        cout << "\nH: " << H << "\n";

        double cols = max(img1.cols, img2.cols);
        double rows_2 = img1.rows + abs(height);
        double rows = rows_2 < img2.rows ? img2.rows : rows_2;
        cout << "SIZE: " << cols << ", " << rows << "\n";
        cout << "roi1: " << img2.cols << ", " << img2.rows << "\n";

        warpPerspective(img1, out, H, Size(cols, rows));
        Mat roi1(out, Rect(0, 0, img2.cols, img2.rows));
        img2.copyTo(roi1);
        ////////////////// resize(out, out, Size(), 0.5, 0.5);
        verticalImages.push_back(out);
    }

    // imshow("out", out);
    // waitKey(10000);
}

Mat doSIFT(vector<Mat> &imgList, string datasetName)
{
    int step = 0;
    while (true)
    {
        step++;
        int numImages = imgList.size();
        cout << "\nNumImages: " << numImages << "\n";
        if (numImages == 1)
        {
            break;
        }
        vector<KeyPoint> keypoints[numImages];
        Mat descriptors[numImages];

        computeKeypointsAndDescriptors(imgList, keypoints, descriptors);

        vector<tuple<vector<DMatch>, int, int>> allGoodMatches;
        computeAllGoodMatches(numImages, keypoints, descriptors, allGoodMatches);

        // vector<Mat> stitchedPairs;
        vector<Mat> horizontalImages;
        vector<Mat> verticalImages;
        int z = allGoodMatches.size(); // numImages;
        cout << z << "\n";
        for (int i = 0; i < z; i++)
        {
            cout << "\n"
                 << i << " / " << z << "\n";
            vector<DMatch> &goodMatches = get<0>(allGoodMatches[i]);
            int indexImg_1 = get<1>(allGoodMatches[i]);
            int indexImg_2 = get<2>(allGoodMatches[i]);
            Mat &img1 = imgList[indexImg_1];
            Mat &img2 = imgList[indexImg_2];
            vector<KeyPoint> &keypoints_1 = keypoints[indexImg_1];
            vector<KeyPoint> &keypoints_2 = keypoints[indexImg_2];

            try
            {
                Mat stitchedImage;
                stitchImages(img1, img2, goodMatches, keypoints_1, keypoints_2, stitchedImage, false, false, horizontalImages, verticalImages);
            }
            catch (bool k)
            {
                cout << "\nSkipped: " << i << "\n";
            }
        }
        for (int i = 0; i < horizontalImages.size(); i++)
        {
            string fileName = "./test_outputs/" + datasetName + "/horizontal_" + to_string(step) + "_" + to_string(i) + ".png";
            imwrite(fileName, horizontalImages[i]);
        }
        for (int i = 0; i < verticalImages.size(); i++)
        {
            string fileName = "./test_outputs/" + datasetName + "/vertical_" + to_string(step) + "_" + to_string(i) + ".png";
            imwrite(fileName, verticalImages[i]);
        }

        if (horizontalImages.size() > 0)
        {
            imgList = horizontalImages;
        }
        else
        {
            if (verticalImages.size() > 0)
            {
                imgList = verticalImages;
            }
            else
            {
                imgList.clear();
                imgList.push_back(imgList[0]);
            }
        }
    }
    string imageType = type2str(imgList[0].type());
    cout << "Image Type: " << type2str(imgList[0].type()) << "\n";
    if (imageType.compare("8UC4") != 0)
    {
        doInpaint(imgList[0]);
    }
    return imgList[0];
}

void doInpaint(Mat &img)
{ // https://answers.opencv.org/question/73645/reconstruct-stitched-image/
    // find black region:
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat mask = gray == 0;

    // pre-fill  black areas with mean color,
    // for easier interpolation
    Scalar m, s;
    cv::meanStdDev(img, m, s, ~mask); // inverted mask !
    img.setTo(m, mask);

    inpaint(img, mask, img, 30, INPAINT_TELEA);
}

string type2str(int type)
{ // https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth)
    {
    case CV_8U:
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
        break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}