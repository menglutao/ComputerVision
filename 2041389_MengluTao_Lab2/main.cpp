#include <iostream>
// #include <chrono>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <sys/stat.h>

#define NEIGHBORHOOD_Y 9 // Window size for averaging color after click
#define NEIGHBORHOOD_X 9
#define SIMILARITY_THRESHOLD 50 // Threshold for color similarity (Euclidean distance in CIELAB space)

using namespace cv;
using namespace std;

const int bins = 256; // bins -- >Histogram size range

const char *winTitle = "input image"; //
// task2
void calcHistogram(Mat &image, const char *title);
void showHistogram(vector<cv::Mat> &hists, string path);
// task3 & task 4
void equalizeHist(Mat &image);
// task5
void luminEualizeHist(Mat &image);

// Part 2 denosied images
void denoiseImage(Mat &image);

int main(int argc, char const *argv[])
{
    Mat src; // declare src as Mat datatype
    cout << "Starting... Loading image\n";
    src = imread("barbecue.png", IMREAD_UNCHANGED); // import picture as a Mat obj, here we dont make change to picture so we use IMREAD_UNCHANGED
    if (src.empty())
    {
        cout << "can't load image... try again\n";
        return 0;
    }
    cout << "Creating screen...\n";
    namedWindow(winTitle, WINDOW_AUTOSIZE); // create a new OpenCV window without destoying it
    imshow(winTitle, src);                  // show picture

    int op;
    cout << "Choose the task to execute:\n";
    cout << "1 - Histogram Equalization\n";
    cout << "2 - Image Filtering\n";
    cout << "\n-> ";
    cin >> op;

    switch (op)
    {
    case 1:
        mkdir("./img/calcHistogram", 0755);
        mkdir("./img/equalizeHist", 0755);
        mkdir("./img/luminEqualize", 0755);

        calcHistogram(src, "Histogram - Original Image");
        waitKey(0); // press Q to exit

        equalizeHist(src);
        waitKey(0);

        luminEualizeHist(src);
        waitKey(0);
        break;
    case 2:
        mkdir("./img/median", 0755); // create folders to store images.
        mkdir("./img/gaussianBlur", 0755);
        mkdir("./img/bilateralFilter", 0755);

        denoiseImage(src);

        break;
    default:
        cout << "Invalid option!";
    }

    return 0;
}

void calcHistogram(Mat &image, const char *title)
{
    const int channels[1] = {0}; // declare a array named channels and has only one value which is 0
    const int bins[] = {256};
    float hranges[] = {0, 255};
    const float *ranges[] = {hranges}; // address of the first element

    int dims = image.channels(); // check the dimensions of image, here we now is 3 channels
    // printf("%d\n"dims);
    cout << dims << "\n";

    vector<Mat> bgr_plane, calcBgrPlane;
    split(image, bgr_plane); // divide image into R,G,B planes

    Mat b_hist;
    Mat g_hist;
    Mat r_hist;

    // calculate blue,green,red channels each one's histogram
    /*
     arguments explanation:
     &bgr_plane[0]: source arrays
     1:number of source array
     0:channels to be measured, since each array is single channel so here I put 0
     Mat(): A mask to be used on the source array
     b_hist: destination to store the histogram
     1:histogram dimensionality
     bins: bins number of per each used dimension,here is 256
     ranges:ranges of values to be masures in each dimension
    */

    calcHist(&bgr_plane[0], 1, 0, Mat(),
             b_hist, 1, bins, ranges);
    calcHist(&bgr_plane[1], 1, 0, Mat(),
             g_hist, 1, bins, ranges);
    calcHist(&bgr_plane[2], 1, 0, Mat(),
             r_hist, 1, bins, ranges);

    // store each plane's histogram into a new ones
    calcBgrPlane.push_back(b_hist);
    calcBgrPlane.push_back(g_hist);
    calcBgrPlane.push_back(r_hist);

    showHistogram(calcBgrPlane, "calcHistogram");
}

// Code from Computer Vision 2022 (P. Zanuttigh, code M. Carraro) - LAB 2
void showHistogram(vector<cv::Mat> &hists, string path)
{
    // Min/Max computation
    double hmax[3];
    double min;
    cv::minMaxLoc(hists[0], &min, &hmax[0]);
    cv::minMaxLoc(hists[1], &min, &hmax[1]);
    cv::minMaxLoc(hists[2], &min, &hmax[2]);

    cout << "min = " << min << "\nmax: " << hmax[0] << " , " << hmax[1] << " , " << hmax[2] << "\n";

    string wname[3] = {"blue", "green", "red"};
    cv::Scalar colors[3] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0),
                            cv::Scalar(0, 0, 255)};

    vector<cv::Mat> canvas(hists.size());

    // Display each histogram in a canvas
    for (int i = 0, end = hists.size(); i < end; i++)
    {
        canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < hists[0].rows - 1; j++)
        {
            cv::line(
                canvas[i],
                cv::Point(j, rows),
                cv::Point(j, rows - (hists[i].at<float>(j) * rows / hmax[i])),
                hists.size() == 1 ? cv::Scalar(200, 200, 200) : colors[i],
                1, 8, 0);
        }
        string imgPath = "img/" + path + "/" + wname[i] + ".png";
        imwrite(imgPath, canvas[i]);
        cv::imshow(hists.size() == 1 ? "value" : wname[i], canvas[i]);
    }
}

// Task 3 & Task 4
void equalizeHist(Mat &image)
{
    Mat equalizedImg;
    const int bins[] = {256};
    float hranges[] = {0, 255};
    const float *ranges[] = {hranges}; // address of the first element

    vector<Mat> bgr_plane, calcBgrPlane;
    split(image, bgr_plane); // divide into b,g,r three individual planes
    Mat b_hist;
    Mat g_hist;
    Mat r_hist;
    // use method equalizeHist to equalize the planes.
    equalizeHist(bgr_plane[0], bgr_plane[0]);
    equalizeHist(bgr_plane[1], bgr_plane[1]);
    equalizeHist(bgr_plane[2], bgr_plane[2]);

    // since we did a split operation before so here we merge the "after" bgr_plane into the equalizedImg.
    merge(bgr_plane, equalizedImg);
    imshow("equalizedImg:", equalizedImg);
    string imgPath = "img/equalizeHist/equalizeHist.png";
    imwrite(imgPath, equalizedImg);

    // calculate blue,green,red each one's histogram after doing equalization operation
    calcHist(&bgr_plane[0], 1, 0, Mat(),
             b_hist, 1, bins, ranges);
    calcHist(&bgr_plane[1], 1, 0, Mat(),
             g_hist, 1, bins, ranges);
    calcHist(&bgr_plane[2], 1, 0, Mat(),
             r_hist, 1, bins, ranges);

    calcBgrPlane.push_back(b_hist);
    calcBgrPlane.push_back(g_hist);
    calcBgrPlane.push_back(r_hist);

    // cout << "red: " <<r_hist.size().width << "blue: " <<b_hist.size().width << "green: " <<g_hist.size().width;
    // cout << "red: " <<r_hist.size().height << "blue: " <<b_hist.size().height << "green: " <<g_hist.size().height;
    showHistogram(calcBgrPlane, "equalizeHist");
}

// Task 5
void luminEualizeHist(Mat &image)
{
    const int bins[] = {256};
    float hranges[] = {0, 255};
    const float *ranges[] = {hranges};

    Mat luminImg, luminImg_color;
    // convert color image to Lab
    cvtColor(image, luminImg, COLOR_BGR2Lab);
    // Extract the L channel
    vector<Mat> lab_planes;
    // split original luminal image into l/a/b planes
    split(luminImg, lab_planes);
    // equalize the first plane--L plane
    equalizeHist(lab_planes[0], lab_planes[0]);

    // merge the color planes back into a lab image
    merge(lab_planes, luminImg);
    // convert lab image back to a color image
    cvtColor(luminImg, luminImg_color, COLOR_Lab2BGR);
    imshow("equalized luminance image :", luminImg_color);
    string imgPath = "img/luminEqualize/luminImg_color.png";
    imwrite(imgPath, luminImg_color);

    vector<Mat> bgr_plane, calcBgrPlane;
    split(luminImg_color, bgr_plane);
    Mat b_hist;
    Mat g_hist;
    Mat r_hist;
    // Calculate histogram after image being equalized in luminance channel
    calcHist(&bgr_plane[0], 1, 0, Mat(),
             b_hist, 1, bins, ranges);
    calcHist(&bgr_plane[1], 1, 0, Mat(),
             g_hist, 1, bins, ranges);
    calcHist(&bgr_plane[2], 1, 0, Mat(),
             r_hist, 1, bins, ranges);

    calcBgrPlane.push_back(b_hist);
    calcBgrPlane.push_back(g_hist);
    calcBgrPlane.push_back(r_hist);

    // show the final image
    showHistogram(calcBgrPlane, "luminEqualize");
}

// PART 2 image filtering

void denoiseImage(Mat &image)
{
    // Applying medianFilter method to smooth image.This function using a median filter with ksize*ksize aperture
    const int medianFilterKernel[] = {3, 5, 7, 99}; // size of aperture varying from 3- 99, should be positive and odd numbers.
    for (int i = 0; i < sizeof(medianFilterKernel) / sizeof(int); i++)
    {
        Mat median;
        medianBlur(image, median, medianFilterKernel[i]);
        string imgPath = "img/median/" + to_string(medianFilterKernel[i]) + ".png";
        cout << "Saving file to: " << imgPath << "\n";
        imwrite(imgPath, median);
    }
    // The results of medianFilter are stored into img/median and as we can see, the bigger the aperture size is, the more smooth image gets.

    // Applying GaussianBlur, this function convolves the source image with the specified Gaussian kernel
    const int GaussianBlurKernel[] = {3, 5, 7, 9, 15, 55, 99}; // will be used into square numbers in function.

    for (int i = 0; i < sizeof(GaussianBlurKernel) / sizeof(int); i++)
    {
        Mat gaussianBlur;
        double sigma = 3;
        GaussianBlur(image, gaussianBlur, Size(GaussianBlurKernel[i], GaussianBlurKernel[i]), sigma, sigma); // sigma :Gaussian kernel standard deviation in X and Y direction. Here we assume they are equal.
        string imgPath = "img/gaussianBlur/" + to_string(GaussianBlurKernel[i]) + ".png";
        cout << "Saving file to: " << imgPath << "\n";
        imwrite(imgPath, gaussianBlur);
    }

    /* The results of GaussianBlur are stored into img/gaussianBlur and as we can see, the bigger the kernal size is, the more smooth image gets.
    However gaussianblur is not good at preserving edges.
    */

    // Applying bilateralFilter

    // sigmaColor:Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color.
    // sigmaSpace:Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough.
    for (int i = 1; i < 14; i = i + 2)
    {
        Mat bilateralFilterImg;
        bilateralFilter(image, bilateralFilterImg, i, i * 2, i / 2);
        string imgPath = "img/bilateralFilter/" + to_string(i) + ".png";
        cout << "Saving file to: " << imgPath << "\n";
        imwrite(imgPath, bilateralFilterImg);
    }

    // The results of bilateralFilter are stored into img/bilateral and we can see that from image 1 to image 13, more smooth the image gets, and preserve the edges as well
}
