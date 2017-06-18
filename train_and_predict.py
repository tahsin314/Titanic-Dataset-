import pandas as pd
import numpy as np
from patsy import dmatrices
from sklearn.model_selection import	train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score


data = pd.read_csv("train.csv")

#Train_test split
train_data, test_data = train_test_split(data, test_size=0.2)

# Remove 'ticket' and 'cabin' because of many missing values in them
train_data = train_data.drop(['Ticket', 'PassengerId', 'Name'], axis=1)
test_data = test_data.drop(['Ticket', 'PassengerId', 'Name'], axis=1)

#Declaring some trainer functions here:
sgd_clf = SGDClassifier(alpha=.00002, random_state=42)
nb = GaussianNB()

# Replacing missing values with median value
train_data = train_data.fillna((train_data.mean()), inplace=True)
test_data = test_data.fillna((test_data.mean()), inplace=True)
train_data["Cabin"].fillna("C")

# Create a regression friendly dataframe using patsy's dmatrices function
formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp  + C(Embarked) + Fare + Parch'

y_train, X_train = dmatrices(formula, data=train_data, return_type='dataframe')
y_test, X_test = dmatrices(formula, data=test_data, return_type='dataframe')

# Fitting and predicting using Naive Bayes. You can replace nb with sgd_clf
model = nb.fit(X_train, y_train)
result = nb.predict(X_test)


# print precision_score(y_test,result)
def accuracy(y, result):
    t = 0
    y1 = y.as_matrix()
    for i in range(len(y1)):
        if y1[i] == result[i]:
            t += 1
    return t*1.0/len(y1)


def plot_accuracy():
    return accuracy(y_test, result)

output = []
for i in range(100):
    output.append(100*plot_accuracy())

print "Accuracy:",
print "%.2f" % np.mean(output)
p = 100.0*precision_score(y_test, result)
r = 100.0*recall_score(y_test, result)
f1 = 100.0*f1_score(y_test, result)

print "Precision \t Recall \t F1 Score"
print " %.2f \t" % p,
print "      %.2f \t" % r,
print "    %.2f \t" % f1

# writing predictions to a csv file called "result.csv"
files = open("result.csv", "w")
for i in range(len(result)):
    if i == 0:
        files.write("Survived\n")
    else:
     files.write("%s" %result[i])
    files.write("\n")

files.close()