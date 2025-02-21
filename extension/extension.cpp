/*Adit Shah
Jheel Kamdar
PRCV Project3
Date:02/20/2025*/


#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>

using namespace cv;
using namespace std;

Vec3b randomColor(RNG& rng) {
    return Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
}

// double computeEuclideanDistance(Point2f p1, Point2f p2) {
//     return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
// }
double computeManhattanDistance(Point2f p1, Point2f p2) {
    return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}

float computeCosineSimilarity(const vector<float>& A, const vector<float>& B) {
    float dotProduct = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < A.size(); i++) {
        dotProduct += A[i] * B[i];
        normA += A[i] * A[i];
        normB += B[i] * B[i];
    }
    normA = sqrt(normA);
    normB = sqrt(normB);
    if (normA < 1e-9 || normB < 1e-9) return 0;
    return dotProduct / (normA * normB);
}

unsigned char computeOtsuThreshold(const Mat& gray) {
    vector<int> histogram(256, 0);
    int totalPixels = gray.rows * gray.cols;
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            histogram[gray.at<uchar>(i, j)]++;
        }
    }
    int sumB = 0, wB = 0, wF = 0;
    float varMax = 0, threshold = 0;
    int sum1 = 0;
    for (int i = 0; i < 256; i++) sum1 += i * histogram[i];
    for (int t = 0; t < 256; t++) {
        wB += histogram[t];
        if (wB == 0) continue;
        wF = totalPixels - wB;
        if (wF == 0) break;
        sumB += t * histogram[t];
        float mB = sumB / (float)wB;
        float mF = (sum1 - sumB) / (float)wF;
        float varBetween = wB * wF * (mB - mF) * (mB - mF);
        if (varBetween > varMax) {
            varMax = varBetween;
            threshold = t;
        }
    }
    return (unsigned char)threshold;
}

Mat applyThreshold(const Mat& gray, unsigned char threshold) {
    Mat binary = gray.clone();
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            binary.at<uchar>(i, j) = (gray.at<uchar>(i, j) >= threshold) ? 255 : 0;
        }
    }
    return binary;
}

Mat erodeImage(const Mat& input, int kernelSize) {
    Mat output = input.clone();
    int offset = kernelSize / 2;
    for (int i = offset; i < input.rows - offset; i++) {
        for (int j = offset; j < input.cols - offset; j++) {
            uchar minPixel = 255;
            for (int ki = -offset; ki <= offset; ki++) {
                for (int kj = -offset; kj <= offset; kj++) {
                    minPixel = min(minPixel, input.at<uchar>(i + ki, j + kj));
                }
            }
            output.at<uchar>(i, j) = minPixel;
        }
    }
    return output;
}

Mat dilateImage(const Mat& input, int kernelSize) {
    Mat output = input.clone();
    int offset = kernelSize / 2;
    for (int i = offset; i < input.rows - offset; i++) {
        for (int j = offset; j < input.cols - offset; j++) {
            uchar maxPixel = 0;
            for (int ki = -offset; ki <= offset; ki++) {
                for (int kj = -offset; kj <= offset; kj++) {
                    maxPixel = max(maxPixel, input.at<uchar>(i + ki, j + kj));
                }
            }
            output.at<uchar>(i, j) = maxPixel;
        }
    }
    return output;
}

Mat openOperation(const Mat& input, int kernelSize) {
    return dilateImage(erodeImage(input, kernelSize), kernelSize);
}

Mat closeOperation(const Mat& input, int kernelSize) {
    return erodeImage(dilateImage(input, kernelSize), kernelSize);
}
// Function to refine segmentation using Depth Anything
Mat refineSegmentationUsingDepth(const Mat& frame, const Mat& depthMap) {
    Mat depthGray, mask, refined;
    normalize(depthMap, depthGray, 0, 255, NORM_MINMAX, CV_8U);
    
    unsigned char depthThreshold = computeOtsuThreshold(depthGray);
    mask = applyThreshold(depthGray, depthThreshold);

    Mat processed;
    bitwise_and(frame, frame, processed, mask);
    return processed;
}

void processFrame(const Mat& frame ,const Mat& depthMap) {
    Mat gray, thresholded, cleaned, depthSegmented;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    unsigned char threshold = computeOtsuThreshold(gray);

    thresholded = applyThreshold(gray, threshold);
    
    depthSegmented = refineSegmentationUsingDepth(frame, depthMap);
    
    int kernelSize = 5;
    cleaned = closeOperation(openOperation(thresholded, kernelSize), kernelSize);
    imshow("Original", frame);
    imshow("Thresholded", thresholded);
    imshow("Cleaned", cleaned);
    imshow("Depth Segmented", depthSegmented);
    waitKey(0);
}

int main() {
    string imagePath, depthPath;
    cout << "Enter image file path: ";
    cin >> imagePath;
    cout << "Enter depth file path: ";
    cin >> depthPath;

    Mat image = imread(imagePath);
    Mat depthMap = imread(depthPath, IMREAD_ANYDEPTH);

    if (image.empty() || depthMap.empty()) {
        cerr << "Error: Cannot open image or depth file!" << endl;
        return -1;
    }

    processFrame(image, depthMap);
    return 0;
}