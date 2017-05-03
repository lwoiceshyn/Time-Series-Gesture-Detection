import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

from sklearn.externals import joblib

import sys
import os

import warnings
warnings.filterwarnings('ignore')

def main(argv):
    try:
        path = argv[0]
    except:
        pass
    print path

    if not os.path.exists(path):
        print "Invalid Path"
        return

    #Load our ML Model
    clf = joblib.load('bestrandomforest.pkl')

    #Create a list with which we will append values to
    class_paths = []
    classes = []
    truths = []

    #Bad TSFresh features to filter out
    bad_features = []
    for i in range(8):
        langevin = str(i) + "__max_langevin_fixed_point__m_3__r_30"
        bad_features.append(langevin)
        for j in range(9):
            quantile = (j + 1) * 0.1
            if quantile != 0.5:
                feature_name = str(i) + "__index_mass_quantile__q_" + str(quantile)
                bad_features.append(feature_name)

    total_predictions = 0
    true_predictions = 0
    for file in os.listdir(path):

        if sys.platform == "win32" or sys.platform == "win64":
            filepath = path + '\\' + file
        else:
            filepath = path + '/' + file

        true_label = int(file[7])

        if file.endswith(".txt") or file.endswith(".csv"):
            try:
                sample = pd.read_csv(filepath, header=None)
            except:
                pass

            #Preprocess the data
            sample[8] = sample.index.astype(float)
            sample[9] = 1.0
            sample = extract_features(sample, column_id=9, column_sort=8)
            impute(sample)
            sample = sample.fillna(0)
            sample.columns = sample.columns.map(lambda t: str(t))
            sample = sample.sort_index(axis=1)
            sample = sample.drop(bad_features, axis=1)

            #Predict the class
            gesture = clf.predict(sample.loc[0:1])[0]
            if gesture == "One":
                predicted_label = 1
            if gesture == "Two":
                predicted_label = 2
            if gesture == "Three":
                predicted_label = 3
            if gesture == "Four":
                predicted_label = 4
            if gesture == "Five":
                predicted_label = 5
            if gesture == "Six":
                predicted_label = 6

            if true_label == predicted_label:
                true_predictions += 1
            total_predictions += 1



            class_paths.append(file)
            classes.append(predicted_label)
            truths.append(true_label)

    accuracy = (float(true_predictions) / total_predictions) * 100
    print "Accuracy:", accuracy, "%"

    #Save Output as CSV File
    output = pd.DataFrame()
    output['Filename'] = class_paths
    output['Predicted'] = classes
    output['True'] = truths
    output = output.assign(Accuracy="")
    accuracy_string = str(accuracy) + "%"
    output['Accuracy'][0] = accuracy_string

    output.to_csv('ClassificationResults.csv', index=False)

if __name__ == "__main__":
    main(sys.argv[1:])