import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier


def main(targets):

    if 'test' in targets:
        trips = pd.read_csv('data/trips.csv')
        utility = pd.read_csv('data/utilityvars.csv')

        sample1 = utility.sample(n=1000000, random_state=1)
        size = 500000
        mode1 = utilityvars[utilityvars['targettripmode']==1].sample(n=int(round(size*props[1])), random_state=1)
        mode2 = utilityvars[utilityvars['targettripmode']==2].sample(n=int(round(size*props[2])), random_state=1)
        mode3 = utilityvars[utilityvars['targettripmode']==3].sample(n=int(round(size*props[3])), random_state=1)
        mode4 = utilityvars[utilityvars['targettripmode']==4].sample(n=int(round(size*props[4])), random_state=1)
        mode5 = utilityvars[utilityvars['targettripmode']==5].sample(n=int(round(size*props[5])), random_state=1)
        mode6 = utilityvars[utilityvars['targettripmode']==6].sample(n=int(round(size*props[6])), random_state=1)
        mode7 = utilityvars[utilityvars['targettripmode']==7].sample(n=int(round(size*props[7])), random_state=1)
        mode8 = utilityvars[utilityvars['targettripmode']==8].sample(n=int(round(size*props[8])), random_state=1)
        mode9 = utilityvars[utilityvars['targettripmode']==9].sample(n=int(round(size*props[9])), random_state=1)
        mode10 = utilityvars[utilityvars['targettripmode']==10].sample(n=int(round(size*props[10])), random_state=1)
        mode11 = utilityvars[utilityvars['targettripmode']==11].sample(n=int(round(size*props[11])), random_state=1)
        mode12 = utilityvars[utilityvars['targettripmode']==12].sample(n=int(round(size*props[12])), random_state=1)
        sample2 = pd.concat([mode1,mode2,mode3,mode4,mode5,mode6,mode7,mode8,mode9,mode10,mode11,mode12], axis=0)

        new_utilityvars1 = pd.get_dummies(sample1, columns=["tourpurpose"], prefix=["tourpurpose"])
        new_utilityvars2 = pd.get_dummies(sample2, columns=["tourpurpose"], prefix=["tourpurpose"])

        seed = 7
        test_size = 0.33

        new_utilityvars1 = pd.merge(new_utilityvars1.drop(['targettripmode', 'tourmode', 'activityid'], axis=1), new_utilityvars1[['targettripmode']], left_index=True, right_index=True, how="outer")
        df_array1 = new_utilityvars1.values
        X1 = df_array1[:,0:34]
        Y1 = df_array1[:,34]
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=test_size, random_state=seed)
        model1 = xgb.XGBClassifier()
        model1.fit(X_train1, y_train1, eval_set = [(X_train1, y_train1), (X_test1, y_test1)], verbose = 0)
        y_pred1 = [round(value) for value in model1.predict(X_test1)]
        y_test1 = [round(value) for value in y_test1]

        new_utilityvars2 = pd.merge(new_utilityvars2.drop(['targettripmode', 'tourmode', 'activityid'], axis=1), new_utilityvars2[['targettripmode']], left_index=True, right_index=True, how="outer")
        df_array2 = new_utilityvars2.values
        X2 = df_array2[:,0:34]
        Y2 = df_array2[:,34]
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size=test_size, random_state=seed)
        model2 = xgb.XGBClassifier()
        model2.fit(X_train2, y_train2, eval_set = [(X_train2, y_train2), (X_test2, y_test2)], verbose = 0)
        y_pred2 = [round(value) for value in model2.predict(X_test2)]
        y_test2 = [round(value) for value in y_test2]

        accuracy1 = metrics.accuracy_score(y_test1, y_pred1)
        sensitivity = metrics.recall_score(y_test1, y_pred1, average = 'macro')
        precision = metrics.precision_score(y_test1, y_pred1, average = 'macro')
        f1 = (2 * precision * sensitivity) / (precision + sensitivity)
        cf_matrix = metrics.confusion_matrix(y_test1, y_pred1)
        accuracy2 = metrics.accuracy_score(y_test2, y_pred2)

        print("The evaluation metric stats")
        print("Accuracy: %.2f%%" % (accuracy1 * 100.0))
        print("Sensitivity: %.2f%%" % (sensitivity * 100.0))
        print("Precision: %.2f%%" % (precision * 100.0))
        print("F1: %.2f%%" % (f1 * 100.0))

        model1.save_model("model1.json")

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
