DATASET=./irisDataset/iris.data
OUT_DATASET=./irisDataset/preprocessedStdSetosa.data

SEPAL_LEN_MEAN=5.84
SEPAL_LEN_STD=0.83
SEPAL_WID_MEAN=3.05
SEPAL_WID_STD=0.43
PETAL_LEN_MEAN=3.76
PETAL_LEN_STD=1.76
PETAL_WID_MEAN=1.20
PETAL_WID_STD=0.76

function main() {
    preprocessDataset && ./main
}

function build() {
    gcc -O2 **/*.c -o main
}

function preprocessDataset() {
    cat $DATASET | shuffle | removeCommas | standardize | oneHotSetosa > $OUT_DATASET
}

function removeCommas() {
    sed 's/,/ /g'
}

function shuffle() {
    sort -R
}

function standardize() {
    awk "{
        stdSepalLen = (\$1 - $SEPAL_LEN_MEAN) / $SEPAL_LEN_STD
        stdSepalWid = (\$2 - $SEPAL_WID_MEAN) / $SEPAL_WID_STD
        stdPetalLen = (\$3 - $PETAL_LEN_MEAN) / $PETAL_LEN_STD
        stdPetalWid = (\$4 - $PETAL_WID_MEAN) / $PETAL_WID_STD
        class = \$5
        print stdSepalLen, stdSepalWid, stdPetalLen, stdPetalWid, class
    }"
}

function oneHotSetosa() {
    sed 's/Iris-setosa/1/' | sed 's/Iris.*/0/'
}

main
