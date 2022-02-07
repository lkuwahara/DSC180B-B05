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

        # drop columns with no significant values (e.g. only 1 value, only unique values)
        utility = utility.drop(columns=['activityid', 'tourmode'])

        utility['gender'] = np.where(utility['gender'] == True, 'female', 'male')
        utility['autosuf'] = np.where(utility['autosuf'] == 0, 'no_vehicles', np.where(utility['autosuf'] == 1, 'insufficient', 'sufficient'))
        utility['tourpurpose'] = np.where(utility['tourpurpose'] == 0, 'work', np.where(utility['tourpurpose'] == 1, 'university', np.where(utility['tourpurpose'] == 2, 'school', np.where(utility['tourpurpose'] == 3, 'maintenance', np.where(utility['tourpurpose'] == 4, 'escort', np.where(utility['tourpurpose'] == 5, 'discretionary', np.where(utility['tourpurpose'] == 6, 'work-based', 'cross-border')))))))

        Y = utility['targettripmode'] - 1
        X1 = utility.select_dtypes(include=['object','bool'])
        X2 = utility.select_dtypes(exclude=['object','bool']).drop(columns='targettripmode')

        expanded_data = []
        for col in X1.columns:
            dummies = pd.get_dummies(X1[col], prefix=col)
            expanded_data.append(dummies)

        expanded_data = pd.concat(expanded_data, axis=1)

        data_encoded = pd.concat([expanded_data, X2, Y], axis=1)

        mode1 = data_encoded[data_encoded['targettripmode']==0].sample(n=1000, random_state=1)
        mode2 = data_encoded[data_encoded['targettripmode']==1].sample(n=1000, random_state=1)
        mode3 = data_encoded[data_encoded['targettripmode']==2].sample(n=1000, random_state=1)
        mode4 = data_encoded[data_encoded['targettripmode']==3].sample(n=1000, random_state=1)
        mode5 = data_encoded[data_encoded['targettripmode']==4].sample(n=1000, random_state=1)
        mode6 = data_encoded[data_encoded['targettripmode']==5].sample(n=1000, random_state=1)
        mode7 = data_encoded[data_encoded['targettripmode']==6].sample(n=1000, random_state=1)
        mode8 = data_encoded[data_encoded['targettripmode']==7].sample(n=1000, random_state=1)
        mode9 = data_encoded[data_encoded['targettripmode']==8].sample(n=1000, random_state=1)
        mode10 = data_encoded[data_encoded['targettripmode']==9].sample(n=1000, random_state=1)
        mode11 = data_encoded[data_encoded['targettripmode']==10].sample(n=1000, random_state=1)
        mode12 = data_encoded[data_encoded['targettripmode']==11].sample(n=1000, random_state=1)

        combined_subsamples = pd.concat([mode1,mode2,mode3,mode4,mode5,mode6,mode7,mode8,mode9,mode10,mode11,mode12], axis=0)

        data_array = combined_subsamples.values
        X = data_array[:, 0:(data_array.shape[1]-1)]
        Y = data_array[:,(data_array.shape[1]-1)]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        xgb_model = XGBClassifier(objective='multi:softprob',eval_metric='auc',num_class=12,use_label_encoder=False)
        xgb_model.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_test,y_test)],verbose=0)
        y_pred = xgb_model.predict(X_test)
        y_pred = [round(value) for value in y_pred]

        accuracy = metrics.accuracy_score(y_test, y_pred)
        sensitivity = metrics.recall_score(y_test, y_pred, average = 'macro')
        precision = metrics.precision_score(y_test, y_pred, average = 'macro')
        f1 = (2 * precision * sensitivity) / (precision + sensitivity)
        cf_matrix = metrics.confusion_matrix(y_test, y_pred)

        print("The evaluation metric stats")
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print("Sensitivity: %.2f%%" % (sensitivity * 100.0))
        print("Precision: %.2f%%" % (precision * 100.0))
        print("F1: %.2f%%" % (f1 * 100.0))

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
