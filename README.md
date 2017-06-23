# Titanic-Dataset-Kaggle
This is a simple approach to solve the <a href="https://www.kaggle.com/c/titanic"> Titanic: Machine Learning From disaster</a> problem.

Pre-requisites:
1. Python 2/3
2. Numpy
3. Scikit-Learn
4. Pandas

I used patsy's dmatrices function to shape and train over the given dataset. It works fine with sklearn, except for cross_validation() function. I splitted the dataset randomly into train and test sets using scikit-learn's another built in function "train_test_split()"(80% for train and 20% for test data).

The algorithm achieved the score 0.7655

To improve this score,you can follow this <a href="http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html"> link </a>.
