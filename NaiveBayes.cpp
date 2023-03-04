/* Author:          Huy Nguyen		HNN190000
   Course:          CS375.004
   Date:            03/03/2023  -   03/03/2023
   Assignment:      Portfolio: ML Algorithms from Scratch
   Compiler:        GNU GCC Compiler

   Description:     Recreate naive bayes model in C++ code for titanic_project.csv
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

//function headers for naive bayes
double calcAgeChance(double, double, double);


int main() {

    ifstream inFS;                                   //Input file stream
    string line;                                     //string to process lines of csv
    string pclass_in, survived_in, sex_in, age_in;   //strings to process values of csv
    const int TRAIN_SIZE = 800;                      //const to set train size
    const int TEST_SIZE = 246;                       //const to set test size

    vector<double> pclass(TRAIN_SIZE);            //vector to hold training pclass data
    vector<double> survived(TRAIN_SIZE);          //vector to hold training survived data
    vector<double> sex(TRAIN_SIZE);               //vector to hold training sex data
    vector<double> age(TRAIN_SIZE);               //vector to hold training age data

    vector<double> pclassTest(TEST_SIZE);         //vector to hold test pclass data
    vector<double> survivedTest(TEST_SIZE);       //vector to hold test survived data
    vector<double> sexTest(TEST_SIZE);            //vector to hold test sex data
    vector<double> ageTest(TEST_SIZE);            //vector to hold test age data

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

    //process data into all training and test vectors
    int numObservations = 0;
    while (inFS.good()) {
        getline(inFS, pclass_in, ',');
        getline(inFS, pclass_in, ',');
        getline(inFS, survived_in, ',');
        getline(inFS, sex_in, ',');
        getline(inFS, age_in, '\n');

        if (numObservations < TRAIN_SIZE) {
            pclass.at(numObservations) = stof(pclass_in);
            survived.at(numObservations) = stof(survived_in);
            sex.at(numObservations) = stof(sex_in);
            age.at(numObservations) = stof(age_in);
        } else {
            pclassTest.at(numObservations - TRAIN_SIZE) = stof(pclass_in);
            survivedTest.at(numObservations - TRAIN_SIZE) = stof(survived_in);
            sexTest.at(numObservations - TRAIN_SIZE) = stof(sex_in);
            ageTest.at(numObservations - TRAIN_SIZE) = stof(age_in);
        }
        numObservations++;
    }

    //resize vectors to match appropriate size
    pclass.resize(TRAIN_SIZE);
    survived.resize(TRAIN_SIZE);
    sex.resize(TRAIN_SIZE);
    age.resize(TRAIN_SIZE);
    pclassTest.resize(TEST_SIZE);
    survivedTest.resize(TEST_SIZE);
    sexTest.resize(TEST_SIZE);
    ageTest.resize(TEST_SIZE);


    //close file
    cout << "Closing file titanic_project.csv." << endl << endl << endl;
    inFS.close();

    //calculate original probabilities
    double originalProb [2] = {};
    for (int i = 0; i < TRAIN_SIZE; i++) {
        if (survived.at(i) == 0)
            originalProb[0]++;
        if (survived.at(i) == 1)
            originalProb[1]++;
    }
    originalProb[0] /= TRAIN_SIZE; originalProb[1] /= TRAIN_SIZE;

    //calculate survival count
    int surviveCount[2] = {};
    for (int i = 0; i < TRAIN_SIZE; i++) {
        if (survived.at(i) == 0)
            surviveCount[0]++;
        if (survived.at(i) == 1)
            surviveCount[1]++;
    }

    //arrays to hold data for calculations
    double pclassChance[2][3] = {};
    int surviveValues [] = { 0, 1 };
    int pclassFactors [] = {1, 2, 3};
    double sexChance[2][2] = {};
    int sexValues[] = { 0, 1 };
    double ageMean[2] = {};
    double ageCovar[2] = {};

    //start timer to track algorithm runtime
    auto start = std::chrono::high_resolution_clock::now();

    //train the model
    for (int s: surviveValues) {
        for (int p: pclassFactors) {
            int count = 0;
            for (int i = 0; i < TRAIN_SIZE; i++) {
                if (survived.at(i)==s && pclass.at(i)==p)
                    count++;
            }
            pclassChance[s][p-1] = ((double) count)/ ((double) surviveCount[s]);
        }

        for (int sv : sexValues) {
            int count = 0;
            for (int i = 0; i < TRAIN_SIZE; i++) {
                if (survived.at(i) == s && sex.at(i) == sv) {
                    count++;
                }
            }
            sexChance[s][sv] = ((double)count) / ((double)surviveCount[s]);
        }

        vector<double> sAge;
        for (int i = 0; i < TRAIN_SIZE; i++) {
            if (survived.at(i)==s) {
                sAge.push_back(age.at(i));
            }
        }
        ageMean[s] = findMean(sAge);
        ageCovar[s] = findCovar(sAge, sAge);
    }

    //stop timer and print algorithm runtime
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Duration: " << duration.count() << " milliseconds" << endl;

    //perform test
    double results[TEST_SIZE] = {};
    for (int i = 0; i < TEST_SIZE; i++) {
        double probs[2] = {};

        double numS = pclassChance[1][(int) (pclassTest.at(i) - 1)] *
                sexChance[1][(int) sexTest.at(i)] *
                originalProb[1] * calcAgeChance(ageTest.at(i), ageMean[1], ageCovar[1]);

        double numP = pclassChance[0][(int) (pclassTest.at(i) - 1)] *
                sexChance[0][(int) sexTest.at(i)] *
                originalProb[0] * calcAgeChance(ageTest.at(i), ageMean[0], ageCovar[0]);

        probs[0] = numS / (numS + numP);
        probs[1] = numP / (numS + numP);
        results[i] = (probs[0] >= probs[1]) ? 1 : 0;
    }

    //output model data
    cout << "Original Probabilities: "<< endl;
    cout << "0: " << originalProb[0] << " 1: " << originalProb[1] << endl << endl;

    //print out Probabilities
    cout << "Pclass Probabilities:" << endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            cout << pclassChance[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    cout << "Sex Probabilities:" << endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            cout << sexChance[i][j] << "\t";
        }
        cout << endl;
    }
    cout << endl;

    //printing age stats
    cout << "Age: " << endl;
    cout << "Mean: " << ageMean[0] << ", Variance: " << ageCovar[0] << endl;
    cout << "Mean: " << ageMean[1] << ", Variance: " << ageCovar[1] << endl;
    cout << endl;

    //calculating accuracy and variables for sensitivity and specificity
    double accuracy = 0;
    for (int i = 0; i < TEST_SIZE; i++) {
        if (survivedTest.at(i) == results[i]) {
            accuracy++;
        }
    }

    //calculating accuracy and variables for sensitivity and specificity
    double truePostive = 0, falseNegative = 0, trueNegative = 0, falsePositive = 0;
    for (int i = 0; i < TEST_SIZE; i++) {
        if (survivedTest.at(i) == results[i] && results[i]==1) {
            truePostive++;
        } else if (survivedTest.at(i) != results[i] && results[i] == 0) {
            falseNegative++;
        }
        else if (survivedTest.at(i) == results[i] && results[i] == 0) {
            trueNegative++;
        }
        else if (survivedTest.at(i) != results[i] && results[i] == 1) {
            falsePositive++;
        }
    }

    cout << setw(4) << setfill(' ') << right <<"pred" << setw(4) <<"0" << setw(4) << "1" << endl;
    cout << setfill(' ') << setw(4) << right << '0' << setw(4) << trueNegative << setw(4) << falseNegative << endl;
    cout << setfill(' ') << setw(4) << right << '1' << setw(4) << falsePositive << setw(4) << falsePositive << endl << endl;

    accuracy /= TEST_SIZE;
    cout << "Accuracy: " << accuracy << endl;
    cout << "Sensitivity: " << truePostive / (truePostive + falseNegative) << endl;
    cout << "Specificity: " << trueNegative / (trueNegative+falsePositive) << endl << endl;

    //end program
    cout << "\nProgram terminated.";
    return 0;
}

double calcAgeChance(double v, double mean, double var) {
    double chance = 1 / sqrt(2* M_PI * var);
    chance *= exp(((-1 * pow((v - mean), 2))/(2*var)));
    return chance;
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

