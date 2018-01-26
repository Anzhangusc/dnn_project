#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define innode 2        //input layer nodes number
#define hidenode 4      //hiddenLayer nodes number
#define hidelayer 1     //hiddenLayer number
#define outnode 1       //outputLayer nodes number
#define learningRate 0.9//alpha

// --- -1~1 random generator --- 
inline double get_11Random()    // -1 ~ 1
{
    return ((2.0*(double)rand()/RAND_MAX) - 1);
}

// --- sigmoid  --- 
inline double sigmoid(double x)
{
    double ans = 1 / (1+exp(-x));
    return ans;
}

// --- inputLayer--- 
// 1.value:     X 
// 2.weight:    W 
// 3.wDeltaSum: Sum of W delta
typedef struct inputNode
{
    double value;
    vector<double> weight, wDeltaSum;
}inputNode;

// --- outputLayer--- 
// 1.value:     A 
// 2.delta:     Cost 
// 3.rightout:  Y
// 4.bias:      Bias
// 5.bDeltaSum: Sum of bias delta
typedef struct outputNode   // outputLayer
{
    double value, delta, rightout, bias, bDeltaSum;
}outputNode;

// --- hiddenLayer--- 
// 1.value:     A 
// 2.delta:     BP delta
// 3.bias:      Bias
// 4.bDeltaSum: Sum of Bias delta
// 5.weight:    W； 
// 6.wDeltaSum： Sum of Weight delta
typedef struct hiddenNode   // hiddenLayer
{
    double value, delta, bias, bDeltaSum;
    vector<double> weight, wDeltaSum;
}hiddenNode;

// --- one sample --- 
typedef struct sample
{
    vector<double> in, out;
}sample;

// --- BP neural network --- 
class BpNet
{
public:
    BpNet();    //contruct function
    void forwardPropagationEpoc();  // foward propagatiuon for one sample
    void backPropagationEpoc();     // backward propagation for one sample

    void training (vector<sample> sampleGroup, double threshold);// update weight, bias
    void predict  (vector<sample>& testGroup);                          // Neural network prediction

    void setInput (vector<double> sampleIn);     // set trainning input
    void setOutput(vector<double> sampleOut);    // set trainning output

public:
    double error;
    inputNode* inputLayer[innode];                      // inputLayer
    outputNode* outputLayer[outnode];                   // outputLayer
    hiddenNode* hiddenLayer[hidelayer][hidenode];       // hiddenLayer
};
