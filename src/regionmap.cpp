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
#include <map>
#include <filesystem> // For directory handling

using namespace cv;
using namespace std;
namespace fs = std::filesystem; // Alias for filesystem

Vec3b randomColor(RNG& rng) {
    return Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
}

// double computeEuclideanDistance(Point2f p1, Point2f p2) {
//     return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
// }
double computeManhattanDistance(Point2f p1, Point2f p2) {
    return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}

//Task 4: Compute features for each region
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
//Task 5: Save feature vectors to a CSV file
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
//Task 6: Load object database from a CSV file
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

string classifyObject(const vector<double>& featureVector, const vector<pair<string, vector<double>>>& database) {
    string predictedLabel;
    double minDistance = numeric_limits<double>::max();

    for (const auto& entry : database) {
        const string& label = entry.first;
        const vector<double>& dbFeatureVector = entry.second;

        double distance = 0.0;
        for (size_t i = 0; i < featureVector.size(); i++) {
            distance += pow(featureVector[i] - dbFeatureVector[i], 2);
        }
        distance = sqrt(distance);

        if (distance < minDistance) {
            minDistance = distance;
            predictedLabel = label;
        }
    }

    return predictedLabel;
}
//Task 7: Evaluate performance using confusion matrix
void evaluatePerformance(const vector<string>& trueLabels, const vector<string>& predictedLabels, const vector<pair<string, vector<double>>>& database) {
    map<string, int> labelToIndex;
    vector<string> labels;
    for (const auto& entry : database) {
        if (labelToIndex.find(entry.first) == labelToIndex.end()) {
            labelToIndex[entry.first] = labels.size();
            labels.push_back(entry.first);
        }
    }

    int numLabels = labels.size();
    Mat confusionMatrix = Mat::zeros(numLabels, numLabels, CV_32S);

    for (size_t i = 0; i < trueLabels.size(); i++) {
        int trueIndex = labelToIndex[trueLabels[i]];
        int predictedIndex = labelToIndex[predictedLabels[i]];
        confusionMatrix.at<int>(trueIndex, predictedIndex)++;
    }

    // Print the confusion matrix with labels
    cout << "Confusion Matrix:" << endl;

    // Determine the maximum label length for proper alignment
    size_t maxLabelLength = 0;
    for (const string& label : labels) {
        if (label.length() > maxLabelLength) {
            maxLabelLength = label.length();
        }
    }

    // Print column headers
    cout << setw(maxLabelLength + 4) << " "; // Align the first column header
    for (const string& label : labels) {
        cout << setw(10) << label; // Adjust width for column headers
    }
    cout << endl;

    // Print rows with row labels and values
    for (int i = 0; i < numLabels; i++) {
        cout << setw(maxLabelLength + 4) << labels[i]; // Align row labels
        for (int j = 0; j < numLabels; j++) {
            cout << setw(10) << confusionMatrix.at<int>(i, j); // Align values
        }
        cout << endl;
    }
}

void processImage(const Mat& image, const string& trueLabel, vector<vector<double>>& featureVectors, vector<string>& trueLabels, vector<string>& predictedLabels, const vector<pair<string, vector<double>>>& database) {
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
    Point2f imageCenter(image.cols / 2.0f, image.rows / 2.0f);

    for (int i = 1; i < numLabels; i++) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int width = stats.at<int>(i, CC_STAT_WIDTH);
        int height = stats.at<int>(i, CC_STAT_HEIGHT);

        //bool touchesBoundary = (x <= 0 || y <= 0 || (x + width) >= image.cols || (y + height) >= image.rows);
        Point2f regionCentroid(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
        float distanceToCenter = computeManhattanDistance(regionCentroid, imageCenter);

        if (area > 1000  && distanceToCenter < 200) {
            colors[i] = randomColor(rng);
            regionMap.setTo(colors[i], labels == i);

            // Compute features for the region
            vector<vector<double>> currentFeatureVectors;
            computeRegionFeatures(labels, i, stats, centroids, regionMap, currentFeatureVectors);

            // Classify the region
            if (!currentFeatureVectors.empty()) {
                string predictedLabel = classifyObject(currentFeatureVectors[0], database);
                featureVectors.push_back(currentFeatureVectors[0]);
                trueLabels.push_back(trueLabel); // Use the true label
                predictedLabels.push_back(predictedLabel);

                // Display the predicted label on the regionMap
                putText(regionMap, predictedLabel, Point(x, y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
            }
        }
    }

    imshow("Region Map", regionMap);
    waitKey(0);
}

int main() {
    // Load the object database
    vector<pair<string, vector<double>>> database = loadObjectDatabase("/Users/aditshah/Desktop/object_db.csv");

    // Directory containing test images
    string testDir = "/Users/aditshah/Desktop/test"; // Change this to your test directory path
    if (!fs::exists(testDir)) {
        cerr << "Error: Test directory does not exist!" << endl;
        return -1;
    }

    vector<vector<double>> featureVectors;
    vector<string> trueLabels;
    vector<string> predictedLabels;

    // Process each image in the directory
    for (const auto& entry : fs::directory_iterator(testDir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            Mat image = imread(entry.path().string());
            if (image.empty()) {
                cerr << "Error: Cannot open image file: " << entry.path() << endl;
                continue;
            }

            // Extract the true label from the filename or a separate mapping
            string trueLabel = entry.path().stem().string(); // Assuming the filename is the true label

            cout << "Processing image: " << entry.path().filename() << " with true label: " << trueLabel << endl;
            processImage(image, trueLabel, featureVectors, trueLabels, predictedLabels, database);
        }
    }

    // Evaluate performance
    evaluatePerformance(trueLabels, predictedLabels, database);

    return 0;
}