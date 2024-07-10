#include <stdio.h>
#include <stdlib.h>

const float LEARNING_RATE = 0.001;
const int TRAINING_SET_SIZE = 120;
const int TEST_SET_SIZE = 150 - TRAINING_SET_SIZE;

/*
I use a simple linear activiation function. This would break backpropagation 
in a neural network, but will work fine for this single perceptron.

I experimented with using a step function, but found the linear function gave
greater feedback and resulted in faster learning.
*/
float ACTIVATION_FUNC(float x) { return x; }

/*
I initially struggled with this due to my unfamiliarity with C. What got me
unstuck was consciously deciding to model the problem. To me, a model is a 
bridge connecting what you're trying to do with the capabilities of the tech
you're using. It's a statment of intent encoded in the language of your tools.

Since this is C, I thought I'd try modeling the Perceptron as a state machine.
C's implementation of structs and funcs makes it amenable to this kind of
thinking.

For instance, classification is represented as a state transition. The
perceptron's output is a field on the struct. Updating the sample also causes
the output to be recomputed - here an output not matching the associated
weights and sample is considered an invalid state.

This is odd coming from a functional perspective; my initial gut instinct is
that calculating the perceptron's output must be a standalone function. Yet
doing it this way makes the output easily accessible when the weights are 
updated.

The nomenclature for the functions following is that functions named normally
can be considered a state transition, while those starting with an underscore
are helper functions for those transitions.
*/

struct DataPoint {
    float sepalLen;
    float sepalWid;
    float petalLen;
    float petalWid;
    float setosaProb;
};

struct Perceptron {
    float sepalLenWeight;
    float sepalWidWeight;
    float petalLenWeight;
    float petalWidWeight;
    float bias;

    struct DataPoint sample;
    float setosaProb;
};

struct PerceptronStats {
    float accuracy;
    int truePositives;
    int trueNegatives;
    int falsePositives;
    int falseNegatives;
};

struct Perceptron thePerceptron;
struct PerceptronStats theStats;

struct DataPoint trainingSet[TRAINING_SET_SIZE];
struct DataPoint testSet[TEST_SET_SIZE];

void _resetStats() {
    theStats.accuracy = 0;
    theStats.truePositives = 0;
    theStats.trueNegatives = 0;
    theStats.falsePositives = 0;
    theStats.falseNegatives = 0;
}

void _initializePerceptron() {
    /*
    It's important to remember to initialize the bias to a negative value. It's
    what keeps the perceptron from firing.

    In a previous iteration I had accidentally set this as positive, and the
    perceptron was never able to learn to stop giving false positives.
    */
    thePerceptron.bias = -0.5;

    thePerceptron.sepalLenWeight = 0.1;
    thePerceptron.sepalWidWeight = -0.2;
    thePerceptron.petalLenWeight = 0.1;
    thePerceptron.petalWidWeight = -0.1;
}

void _populateSet(struct DataPoint * set, int setSize, FILE * datasetFile) {
    char line[100];
    char * parsePtr;
    float class;

    for (int i = 0; i < setSize; i++) {
        struct DataPoint * entry = &set[i];
        
        fgets(line, 100, datasetFile);
        entry->sepalLen = strtof(line, &parsePtr);
        entry->sepalWid = strtof(parsePtr, &parsePtr);
        entry->petalLen = strtof(parsePtr, &parsePtr);
        entry->petalWid = strtof(parsePtr, &parsePtr);
        entry->setosaProb = strtof(parsePtr, NULL);
    }
}
void _populateTrainingSet(FILE * datasetFile) {
    _populateSet(trainingSet, TRAINING_SET_SIZE, datasetFile);
}
void _populateValidationSet(FILE * datasetFile) {
    _populateSet(testSet, TEST_SET_SIZE, datasetFile);
}
void _initializeDatasets() {
    FILE * irisDataset = fopen("./irisDataset/preprocessedStdSetosa.data", "r");
    _populateTrainingSet(irisDataset);
    _populateValidationSet(irisDataset);
    fclose(irisDataset);
}

void startUp() {
    _initializePerceptron();
    _initializeDatasets();
    _resetStats();
}

void _updateOutput() {
    struct DataPoint sample = thePerceptron.sample;
    float x; 
    x  = thePerceptron.bias;
    x += thePerceptron.sepalLenWeight * sample.sepalLen;
    x += thePerceptron.sepalWidWeight * sample.sepalWid;
    x += thePerceptron.petalLenWeight * sample.petalLen;
    x += thePerceptron.petalWidWeight * sample.petalWid;

    thePerceptron.setosaProb = ACTIVATION_FUNC(x);
}

void updateSample(struct DataPoint newSample) {
    thePerceptron.sample = newSample;
    _updateOutput();
}

/*
The learning rule. Note the elegant simplicity - if the perceptron "fires" (is 
around 1) when it shouldn't have, the error will be negative. When multiplied 
by the learning rate and added to the weights, this will make all the weights 
smaller. Positive weights will be less positive, negative weights (including 
the bias) will be more negative, and the net effect is that the perceptron will
be less likely to fire when encountering the same sample.

The other aspect which allows this to work is that the adjustment factor is 
multiplied by the input associated with each weight. The larger the abs value 
of the input, the greater the change in weight. This is brilliant in its own 
way: particularly large inputs communicate more "signal" than smaller ones, and 
therfor the perceptron should "pay greater attention" to that input by 
adjusting its corresponding weight by a larger amount.

This aspect of the learning rule suggests why it is advisable to normalize or
standardize data before training. We usually don't care about the total, real
value of the inputs, but of how large or small they are relative to all other
inputs.

This is also suggests why I've had better luck here standardizing my data
instead of normalizing it. With normalization, all inputs are squshed to fit
between 0 and 1. Particularly small values will be at or close to zero, while
particularly large values will be close to one, and average values will fall
in the middle. This means that particularly large values will have the most
effect on the perceptrons weights, followed by average values, and then
particularly small values.

This isn't what I want. Both the particularly large and the particularly small
values carry a greater "signal" than the average values, I think. So I want the
very large and small values to have a big effect on weights, and the average 
values to have less of an effect. Standardization does exactly that.
*/
void updateWeights() {
    struct DataPoint sample = thePerceptron.sample;
    
    float error = sample.setosaProb - thePerceptron.setosaProb;
    float adjFactor = error * LEARNING_RATE;

    thePerceptron.bias += adjFactor;
    thePerceptron.sepalLenWeight += sample.sepalLen * adjFactor;
    thePerceptron.sepalWidWeight += sample.sepalWid * adjFactor;
    thePerceptron.petalLenWeight += sample.petalLen * adjFactor;
    thePerceptron.petalWidWeight += sample.petalWid * adjFactor;
}

void trainForEpoch() {
    for (int i = 0; i < TRAINING_SET_SIZE; i++) {
        updateSample(trainingSet[i]);
        updateWeights();
    }
}

int _isTruePositive() {
    struct DataPoint sample = thePerceptron.sample;
    return thePerceptron.setosaProb >= 0.5 && sample.setosaProb == 1.0;
}
int _isTrueNegative() {
    struct DataPoint sample = thePerceptron.sample;
    return thePerceptron.setosaProb < 0.5 && sample.setosaProb == 0.0;
}
int _isFalsePositive() {
    struct DataPoint sample = thePerceptron.sample;
    return thePerceptron.setosaProb >= 0.5 && sample.setosaProb == 0.0;
}
int _isFalseNegative() {
    struct DataPoint sample = thePerceptron.sample;
    return thePerceptron.setosaProb < 0.5 && sample.setosaProb == 1.0;
}

void testPerceptron() {
    _resetStats();
    for (int i = 0; i < TEST_SET_SIZE; i++) {
        updateSample(testSet[i]);

        theStats.truePositives += _isTruePositive();
        theStats.trueNegatives += _isTrueNegative();
        theStats.falsePositives += _isFalsePositive();
        theStats.falseNegatives += _isFalseNegative();
    }
    int totalCorrect = theStats.truePositives + theStats.trueNegatives;
    theStats.accuracy = (float)totalCorrect / (float)TEST_SET_SIZE;
}

void _printStats() {
    printf("  P\tN\n");
    printf("T %d\t%d\n", theStats.truePositives, theStats.trueNegatives);
    printf("F %d\t%d\n", theStats.falsePositives, theStats.falseNegatives);
    printf("Accuracy: %f\n", theStats.accuracy);
}

int main() {
    startUp();
    for (int i = 0; i < 20; i++) { 
        testPerceptron();
        _printStats();
        trainForEpoch(); 
    }
}
