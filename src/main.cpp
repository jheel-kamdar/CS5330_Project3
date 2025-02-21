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

double computeEuclideanDistance(Point2f p1, Point2f p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
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
    if (normA < 1e-9 || normB < 1e-9) return 0; // Avoid division by zero
    return dotProduct / (normA * normB);
}

void computeRegionFeatures(const Mat& regionMap, int regionID, const Mat& stats, const Mat& centroids, Mat& output, vector<vector<double>>& featureVectors) {
    Mat regionMask = (regionMap == regionID);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(regionMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        RotatedRect orientedBoundingBox = minAreaRect(contours[0]);
        double regionArea = contourArea(contours[0]);
        double boundingBoxArea = orientedBoundingBox.size.area();
        double percentFilled = (boundingBoxArea > 0) ? (regionArea / boundingBoxArea) * 100 : 0;
        double heightWidthRatio = orientedBoundingBox.size.height / orientedBoundingBox.size.width;
        double theta = orientedBoundingBox.angle;
        featureVectors.push_back({regionArea, percentFilled, heightWidthRatio, theta});

        Point2f vertices[4];
        orientedBoundingBox.points(vertices);
        for (int i = 0; i < 4; i++) {
            line(output, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 2);
        }
    }
}

void saveFeatureVectors(const vector<vector<double>>& featureVectors, const string& label) {
    ofstream file("/Users/aditshah/Desktop/object_db.csv", ios::app);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file for writing" << endl;
        return;
    }
    for (const auto& vec : featureVectors) {
        file << label;
        for (double feature : vec) {
            file << "," << feature;
        }
        file << endl;
    }
    file.close();
}

vector<pair<string, vector<double>>> loadFeatureVectors() {
    vector<pair<string, vector<double>>> featureVectors;
    ifstream file("/Users/aditshah/Desktop/object_db.csv");
    if (!file.is_open()) {
        cerr << "Error: Unable to open file for reading" << endl;
        return featureVectors;
    }
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string label;
        getline(ss, label, ',');
        vector<double> features;
        string feature;
        while (getline(ss, feature, ',')) {
            features.push_back(stod(feature));
        }
        featureVectors.push_back({label, features});
    }
    file.close();
    return featureVectors;
}

bool isKnownObject(const vector<double>& features, const vector<pair<string, vector<double>>>& knownObjects, double threshold = 0.9) {
    for (const auto& knownObject : knownObjects) {
        float similarity = computeCosineSimilarity(
            vector<float>(features.begin(), features.end()),
            vector<float>(knownObject.second.begin(), knownObject.second.end())
        );
        if (similarity > threshold) {
            return true;
        }
    }
    return false;
}

void processFrame(const Mat& frame, const vector<pair<string, vector<double>>>& knownObjects) {
    //Task`1: Convert the input frame to grayscale and apply Otsu's thresholding
    Mat gray, thresholded, cleaned;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    threshold(gray, thresholded, 127, 255, THRESH_BINARY | THRESH_OTSU);
    
    //Task`2: Apply morphological opening and closing operations to clean the binary image
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(thresholded, cleaned, MORPH_OPEN, kernel);
    morphologyEx(cleaned, cleaned, MORPH_CLOSE, kernel);
    
    // Invert the binary image to detect black objects
    Mat inverted;
    bitwise_not(cleaned, inverted);

    //Task`3: Find connected components and compute their features
    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(inverted, labels, stats, centroids);
    
    Mat regionMap = Mat::zeros(labels.size(), CV_8UC3);
    RNG rng(12345);
    vector<Vec3b> colors(numLabels);
    vector<vector<double>> featureVectors;
    Point2f imageCenter(frame.cols / 2.0f, frame.rows / 2.0f);

    for (int i = 1; i < numLabels; i++) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int width = stats.at<int>(i, CC_STAT_WIDTH);
        int height = stats.at<int>(i, CC_STAT_HEIGHT);

        bool touchesBoundary = (x <= 0 || y <= 0 || (x + width) >= frame.cols || (y + height) >= frame.rows);
        Point2f regionCentroid(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
        float distanceToCenter = computeEuclideanDistance(regionCentroid, imageCenter);

        if (area > 1000 && !touchesBoundary && distanceToCenter < 200) {
            colors[i] = randomColor(rng);
            regionMap.setTo(colors[i], labels == i);
            computeRegionFeatures(labels, i, stats, centroids, regionMap, featureVectors);

            // Check if the object is known
            if (!isKnownObject(featureVectors.back(), knownObjects)) {
                string label;
                cout << "Unknown object detected. Enter object label: ";
                cin >> label;
                saveFeatureVectors({featureVectors.back()}, label);
                cout << "Feature vectors saved for label: " << label << endl;
            }
        }
    }

    imshow("Original", frame);
    imshow("Thresholded", thresholded);
    imshow("Cleaned", cleaned);
    imshow("Region Map", regionMap);

    char key = waitKey(0);
    if (key == 'N' || key == 'n') {
        string label;
        cout << "Enter object label: ";
        cin >> label;
        saveFeatureVectors(featureVectors, label);
        cout << "Feature vectors saved for label: " << label << endl;
    }
}

int main() {
    // Load known objects from the database
    vector<pair<string, vector<double>>> knownObjects = loadFeatureVectors();

    char choice;
    cout << "Press 'V' for video mode or 'I' for image mode: ";
    cin >> choice;

    if (choice == 'V' || choice == 'v') {
        VideoCapture cap(0);
        if (!cap.isOpened()) {
            cerr << "Error: Unable to open the camera" << endl;
            return -1;
        }

        while (true) {
            Mat frame;
            cap >> frame;
            if (frame.empty()) {
                cerr << "Error: Unable to capture frame" << endl;
                break;
            }

            processFrame(frame, knownObjects);
            char key = waitKey(30);
            if (key == 27) break;
        }

        cap.release();
        destroyAllWindows();
    } 
    else if (choice == 'I' || choice == 'i') {
        string imagePath;
        cout << "Enter image file path: ";
        cin >> imagePath;

        Mat image = imread(imagePath);
        if (image.empty()) {
            cerr << "Error: Cannot open image file!" << endl;
            return -1;
        }

        processFrame(image, knownObjects);
    } 
    else {
        cerr << "Invalid input! Restart and press 'V' or 'I'." << endl;
        return -1;
    }
    return 0;
}