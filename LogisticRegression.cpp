/* Author:          Huy Nguyen		HNN190000
   Course:          CS375.004
   Date:            03/03/2023  -   03/03/2023
   Assignment:      Portfolio: ML Algorithms from Scratch
   Compiler:        GNU GCC Compiler

   Description:     Recreate logistic regression model in C++ code for titanic_project.csv
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

using namespace std;

//structure to hold data for survived and sex
struct observation {
    double survived;
    double sex;
};

//leftover function headers from data exploration
void print_stats(vector<double>);
double findSum(vector<double>);
double findMean(vector<double>);
double findMedian(vector<double>);
double findMin(vector<double>);
double findMax(vector<double>);
double findStanDev(vector<double>);
double findCovar(vector<double>, vector<double>);
double findCor(vector<double>, vector<double>);

//function headers for logisitic regression
vector<double> sigmoid(vector<double>);
vector<vector<double>> transposeMatrix(vector<vector<double>>);
vector<double> matrixTimesVector(vector<vector<double>>, vector<double>);


int main() {

    ifstream inFS;                                 //Input file stream
    string line;                                   //string to process lines of csv
    string pclass_in, survived_in, sex_in, age_in; //strings to process values of csv
    const int MAX_LEN = 1500;                      //const to set maximum length of vectors
    const int TRAIN_SIZE = 800;                    //const to set train size

    vector<double> survived(MAX_LEN);           //vector to hold survive stat
    vector<double> sex(MAX_LEN);                //vector to hold sex stat

    vector<observation> train;                     //vector to hold all train data
    vector<observation> test;                      //vector to hold all test data
    vector<vector<double>> observationMatrix;      //2d vector to do matrix calculations

    vector<double> weights{ 1, 1 };                //vector to do matrix calculations
    double learningRate = .001;                    //double for learning rate

    //open titanic_project.csv file
    cout << "Opening file titanic_project.csv." << endl;
    inFS.open("titanic_project.csv");

    //if file failed to open print error message and end program
    if (!inFS.is_open()) {
        cout << "Could not open file titanic_project.csv." << endl;
        return 1;
    }

    //read & print the heading
    getline(inFS, line);
    cout << "heading: " << line << endl;

    //process data into survived and sex vectors
    int numObservations = 0;
    while (inFS.good()) {
        getline(inFS, pclass_in, ',');
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');

        survived.at(numObservations) = stof(survived_in);
        sex.at(numObservations) = stof(sex_in);

        numObservations++;
    }

    //resize vectors to match appropriate size
    survived.resize(numObservations);
    sex.resize(numObservations);


    //close file
    cout << "Closing file titanic_project.csv." << endl << endl;
    inFS.close();

    //insert training data into train vector
    for (int i = 0; i < TRAIN_SIZE; i++) {
        observation o = { survived.at(i),sex.at(i) };
        train.push_back(o);
    }

    //insert test data into test vector
    for (int i = TRAIN_SIZE; i < numObservations; i++) {
        observation o = { survived.at(i),sex.at(i) };
        test.push_back(o);
    }

    //create matrix for calculations
    for (int i = 0; i < train.size(); i++) {
        vector<double> vect{ 1, (double)train.at(i).survived };
        observationMatrix.push_back(vect);
    }

    //transpose matrix for calculations
    vector<vector<double>> matrixTransposed = transposeMatrix(observationMatrix);

    //start timer to track algorithm runtime
    auto start = std::chrono::high_resolution_clock::now();

    //train the model
    for (int i = 0; i < 30000; i++) {
        vector<double> probVector = sigmoid(matrixTimesVector(observationMatrix, weights));
        vector<double> error(probVector.size());

        for (int i = 0; i < train.size(); i++)
            error.at(i) = train.at(i).survived - probVector.at(i);

        vector<double> transposedMatrixProduct = matrixTimesVector(matrixTransposed, error);

        for (int i = 0; i < transposedMatrixProduct.size(); i++)
            transposedMatrixProduct.at(i) *= learningRate;

        weights.at(0) += transposedMatrixProduct.at(0);
        weights.at(1) += transposedMatrixProduct.at(1);
    }

    //stop timer and print algorithm runtime
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "duration: " << duration.count() << " milliseconds" << endl;


    //create matrix for calculations
    vector<vector<double>> testMatrix;
    for (int i = 0; i < test.size(); i++) {
        vector<double> vect{ 1, (double)test.at(i).sex };
        testMatrix.push_back(vect);
    }

    //start calculating probabilities
    vector<double> predicted = matrixTimesVector(testMatrix, weights);
    vector<double> probability;
    for (int i = 0; i < testMatrix.size(); i++) {
        probability.push_back( exp(predicted.at(i) /  ( 1 + exp(predicted.at(i)))));
    }
    vector<double> predictions;
    for (int i = 0; i < probability.size(); i++) {
        if (probability.at(i) > .5)
            predictions.push_back(1);
        else
            predictions.push_back(0);
    }

    //calculating accuracy and variables for sensitivity and specificity
    double accuracy = 0;
    double truePositive = 0, falseNegative = 0, trueNegative = 0, falsePositive = 0;
    for (int i = 0; i < predictions.size(); i++) {
        int var = predictions.at(i);
        if (var == test.at(i).survived) {
            accuracy++;
            if (var == 0)
                trueNegative++;
            else
                truePositive++;
        }
        else {
            if (var == 0)
                falseNegative++;
            else
                falsePositive++;
        }
    }

    //output model data
    cout << endl << "Coefficients and Accuracy" << endl;
    cout << "Weight 1: " << weights.at(0) << endl;
    cout << "Weight 2: " << weights.at(1) << endl << endl;;
    accuracy /= predictions.size();
	cout << "Accuracy: " << accuracy << endl << endl;
	
    cout << "Sensitivity: " << truePositive / (truePositive + falseNegative) << endl;
    cout << "Specificity: " << trueNegative / (trueNegative+falsePositive) << endl << endl;

    cout << setw(4) << setfill(' ') << right <<"pred" << setw(4) <<"0" << setw(4) << "1" << endl;
    cout << setfill(' ') << setw(4) << right << '0' << setw(4) << trueNegative << setw(4) << falseNegative << endl;
    cout << setfill(' ') << setw(4) << right << '1' << setw(4) << falsePositive << setw(4) << truePositive << endl << endl;

    //end program
    cout << "\nProgram terminated.";
    return 0;

}

vector<double> sigmoid(vector<double> matrix) {
    vector<double> result(matrix.size());
    for (int i = 0; i < matrix.size(); i++)
        result.at(i) = 1.0 / (1 + exp(-(matrix.at(i))));
    return result;
}

vector<vector<double>> transposeMatrix(vector<vector<double>> matrix) {
    vector<vector<double>> transpose;
    vector<double> firstRow;
    vector<double> secondRow;
    for (int i = 0; i < matrix.size(); i++) {
        firstRow.push_back(matrix.at(i).at(0));
        secondRow.push_back(matrix.at(i).at(1));
    }
    transpose.push_back(firstRow);
    transpose.push_back(secondRow);
    return transpose;
}

vector<double> matrixTimesVector(vector<vector<double>> matrix, vector<double> vect) {
    vector<double> result(matrix.size());

    for (int i = 0; i < matrix.size(); i++) {
        vector<double> m = matrix.at(i);
        double prod = 0;
        for (int j = 0; j < m.size(); j++)
            prod += m.at(j) * vect.at(j);
        result.at(i) = prod;
    }
    return result;
}


// Code from data exploration
void print_stats(vector<double> vect){
    cout << fixed << setprecision(3);
    cout << "Mean:\t" << findMean(vect) << endl;
    cout << "Median:\t" << findMedian(vect) << endl;
    cout << "Min:\t" << findMin(vect) << endl;
    cout << "Max:\t" << findMax(vect) << endl;
    cout << "Sum:\t" << findSum(vect) << endl;
    cout << endl;
}

double findSum(vector<double> vect){
    return accumulate(vect.begin(), vect.end(), 0.0);
}

double findMean(vector<double> vect){
    return findSum(vect) / vect.size();
}

double findMedian(vector<double> vect){
    sort(vect.begin(), vect.end());
    return vect[vect.size()/2];
}

double findMin(vector<double> vect){
    sort(vect.begin(), vect.end());
    return vect[0];
}

double findMax(vector<double> vect){
    sort(vect.begin(), vect.end());
    return vect[vect.size()-1];
}

double findStanDev(vector<double> vect1) {
    double stanDev = 0.0;
    for (int i = 0; i < vect1.size(); i++){
        stanDev += pow(vect1[i] - findMean(vect1), 2);
    }
    return sqrt(stanDev/vect1.size());
}

double findCovar(vector<double> vect1, vector<double> vect2){
    if(vect1.size() == vect2.size()){
        double sum = 0.0;
        for (int i = 0; i < vect1.size(); i++){
            sum = sum + (vect1[i] - findMean(vect1)) * (vect2[i] - findMean(vect2));
        }
        return sum / (vect1.size()-1);
    }
    return -1.0;
}

double findCor(vector<double> vect1, vector<double> vect2){
    return findCovar(vect1,vect2) / (findStanDev(vect1) * findStanDev(vect2));
}




