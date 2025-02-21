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
#include <filesystem> // For directory handling

using namespace cv;
using namespace std;
namespace fs = std::filesystem; // Alias for filesystem

Vec3b randomColor(RNG& rng) {
    return Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
}

double computeEuclideanDistance(Point2f p1, Point2f p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}
//Extension add one or two more classifier and distance metrics
// double computeManhattanDistance(Point2f p1, Point2f p2) {
//     return abs(p1.x - p2.x) + abs(p1.y - p2.y);
// }

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
    ofstream file("object_db.csv", ios::app);
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

vector<pair<string, vector<double>>> loadObjectDatabase(const string& filename) {
    vector<pair<string, vector<double>>> database;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file for reading" << endl;
        return database;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string label;
        getline(ss, label, ',');
        vector<double> features;
        double feature;
        while (ss >> feature) {
            features.push_back(feature);
            if (ss.peek() == ',') ss.ignore();
        }
        database.push_back({label, features});
    }

    file.close();
    return database;
}

string classifyDecisionTree(const vector<double>& featureVector) {
    double regionArea = featureVector[0];
    double percentFilled = featureVector[1];
    double heightWidthRatio = featureVector[2];
    double theta = featureVector[3];

    // Decision Tree Rules
    if (regionArea < 5000) {
        if (percentFilled < 50) {
            return "ObjectA"; // Small and sparse
        } else {
            return "ObjectB"; // Small and dense
        }
    } else {
        if (heightWidthRatio > 1.5) {
            return "ObjectC"; // Large and tall
        } else {
            return "ObjectD"; // Large and wide
        }
    }
}

void processImage(const Mat& image, const vector<pair<string, vector<double>>>& database) {
    Mat gray, thresholded, cleaned;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    threshold(gray, thresholded, 127, 255, THRESH_BINARY | THRESH_OTSU);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(thresholded, cleaned, MORPH_OPEN, kernel);
    morphologyEx(cleaned, cleaned, MORPH_CLOSE, kernel);

    // Invert the binary image to detect black objects
    Mat inverted;
    bitwise_not(cleaned, inverted);

    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(inverted, labels, stats, centroids);

    Mat regionMap = Mat::zeros(labels.size(), CV_8UC3);
    RNG rng(12345);
    vector<Vec3b> colors(numLabels);
    vector<vector<double>> featureVectors;
    Point2f imageCenter(image.cols / 2.0f, image.rows / 2.0f);

    for (int i = 1; i < numLabels; i++) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int width = stats.at<int>(i, CC_STAT_WIDTH);
        int height = stats.at<int>(i, CC_STAT_HEIGHT);

        bool touchesBoundary = (x <= 0 || y <= 0 || (x + width) >= image.cols || (y + height) >= image.rows);
        Point2f regionCentroid(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
        float distanceToCenter = computeEuclideanDistance(regionCentroid, imageCenter);

        if (area > 1000 && !touchesBoundary && distanceToCenter < 200) {
            colors[i] = randomColor(rng);
            regionMap.setTo(colors[i], labels == i);

            // Compute features for the region
            vector<vector<double>> currentFeatureVectors;
            computeRegionFeatures(labels, i, stats, centroids, regionMap, currentFeatureVectors);

            // Classify the region using the decision tree
            if (!currentFeatureVectors.empty()) {
                string predictedLabel = classifyDecisionTree(currentFeatureVectors[0]);
                putText(regionMap, predictedLabel, Point(x, y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
            }
        }
    }

    // Display results
    imshow("Original", image);
    imshow("Thresholded", thresholded);
    imshow("Cleaned", cleaned);
    imshow("Region Map", regionMap);
    waitKey(0); // Wait for a key press before moving to the next image
}

void processDirectory(const string& dirPath, const vector<pair<string, vector<double>>>& database) {
    if (!fs::exists(dirPath)) {
        cerr << "Error: Directory does not exist!" << endl;
        return;
    }

    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            Mat image = imread(entry.path().string());
            if (image.empty()) {
                cerr << "Error: Cannot open image file: " << entry.path() << endl;
                continue;
            }

            cout << "Processing image: " << entry.path().filename() << endl;
            processImage(image, database);
        }
    }
}

int main() {
    // Load the object database
    vector<pair<string, vector<double>>> database = loadObjectDatabase("/Users/aditshah/Desktop/object_db.csv");

    char choice;
    cout << "Press 'V' for video mode, 'I' for single image mode, or 'D' for directory mode: ";
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

            processImage(frame, database);
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

        processImage(image, database);
    } 
    else if (choice == 'D' || choice == 'd') {
        string dirPath;
        cout << "Enter directory path: ";
        cin >> dirPath;

        processDirectory(dirPath, database);
    } 
    else {
        cerr << "Invalid input! Restart and press 'V', 'I', or 'D'." << endl;
        return -1;
    }

    return 0;
}