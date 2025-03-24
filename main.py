import argparse

import pandas as pd

import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import approaches
from approaches.flesch_reading_ease import get_flesch_reading_ease
from approaches.xlmr import get_formality_xlmr


parser = argparse.ArgumentParser()
parser.add_argument("-a", "--approach", type=str, help="Approach to formality detection: \"flesch\" or \"xlmr\"")

args = parser.parse_args()
approach = args.approach



def main():
    print("Hello!")
    
    if approach == "flesch":
        df = pd.read_csv("data/test.csv")
        df["pred"] = df["sentence"].apply(lambda x: get_flesch_reading_ease(x))
        mae = mean_absolute_error(df["avg_score"], df["pred"])
        mse = mean_squared_error(df["avg_score"], df["pred"])
        print("For Flesch Reading Ease: Mean Absolute Error - {0}, Mean Squared Error - {1}".format(mae, mse))
    elif approach == "xlmr":
        df["pred"] = df["sentence"].apply(lambda x: get_formality_xlmr(x))
        accuracy = accuracy_score(df["formal"], df["pred"])
        precision = precision_score(df["formal"], df["pred"])
        recall = recall_score(df["formal"], df["pred"])
        f1 = f1_score(df["formal"], df["pred"])
        print("For XLM-Roberta-based classifier: Accuracy - {0}, Precision - {1}, Recall - {2}, F1 - {3}".format(accuracy, precision, recall, f1))



if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()