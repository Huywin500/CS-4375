/* Author:          Huy Nguyen		HNN190000
   Course:          CS375.004
   Date:            02/03/2023  -   02/03/2023
   Assignment:      Portfolio: C++ Data Exploration
   Compiler:        GNU GCC Compiler

   Description:     Recreate data exploration functions in C++ code
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

void print_stats(vector<double>);
double findSum(vector<double>);
double findMean(vector<double>);
double findMedian(vector<double>);
double findMin(vector<double>);
double findMax(vector<double>);
double findStanDev(vector<double>);
double findCovar(vector<double>, vector<double>);
double findCor(vector<double>, vector<double>);


int main() {

    ifstream inFS;                      // Input file stream
    string line;                        //string to process lines of csv
    string rm_in, medv_in;              //strings to process values of csv
    const int MAX_LEN = 1000;           //const to set maximum length of vectors
    vector<double> rm(MAX_LEN);     //vector to hold data of rm
    vector<double> medv(MAX_LEN);   //vector to hold data of medv

    //open Boston.csv file
    cout << "Opening file Boston.csv." << endl;
    inFS.open("Boston.csv");

    //if file failed to open print error message and end program
    if (!inFS.is_open()) {
        cout << "Could not open file Boston.csv." << endl;
        return 1;
    }

    //read & print the heading
    getline(inFS, line);
    cout << "heading: " << line << endl;

    //process data into rm and medv vectors
    int numObservations = 0;
    while (inFS.good()) {
        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');

        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);
        numObservations++;
    }

    //resize vectors to match appropriate size
    rm.resize(numObservations);
    medv.resize(numObservations);

    //close file
    cout << "Closing file Boston.csv." << endl;
    inFS.close();

    //print number of records
    cout << "Number of records: " << numObservations << endl;

    //print summary of rm and medv
    cout << "\nStats for rm" << endl;
    print_stats(rm);
    cout << "\nStats for medv" << endl;
    print_stats(medv);

    //print covariance and correlation of rm and medv
    cout << "\nCovariance = " << findCovar(rm, medv) << endl;
    cout << "\nCorrelation = " << findCor(rm, medv) << endl;

    //end program
    cout << "\nProgram terminated.";
    return 0;

}

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




