from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import sys

EXECUTION_DATETIME = datetime.now()
FORMATTED_EXECUTION_DATETIME = "{}T{}".format(
    EXECUTION_DATETIME.strftime("%x").replace("/", "-"),
    EXECUTION_DATETIME.strftime("%X"))
CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

TRAINING_DATASET_CSV_FILE_PATH = CURRENT_DIR_PATH + "/data/training_dataset.csv"
LOGS_PATH = CURRENT_DIR_PATH + "/logs"


def write_to_log(value):
    if not os.path.exists(LOGS_PATH):
        os.makedirs(LOGS_PATH)

    log_file_path = "{}/{}.log".format(
        LOGS_PATH, FORMATTED_EXECUTION_DATETIME)

    file = open(log_file_path, "a")
    file.write(value)
    file.close()


def import_dataframe(file_path):
    try:
        training_dataframe = pd.read_csv(file_path, delimiter=',',
                                         header=0)
        return training_dataframe
    except FileNotFoundError:
        errorMessage = 'Training File not found at: {}'.format(
            file_path)
        print(errorMessage)
        sys.exit()


def get_initial_weights(dataframe):
    weights = []
    num_columns = len(dataframe.columns[:-1])

    for i in range(num_columns):
        weights.append(random.uniform(0.0, 0.5))
    return weights


def train(dataset, initialized_weights):
    # TODO - Implement Adaline Training algorithm
    raise NotImplementedError


if __name__ == "__main__":
    write_to_log("-----------Started Script-----------")

    training_dataframe = import_dataframe(TRAINING_DATASET_CSV_FILE_PATH)
    write_to_log("\nImported Training DataFrame:\n {}".format(training_dataframe))

    weights = get_initial_weights(training_dataframe)
    bias = random.uniform(0.0, 0.5)
    initial_weights = weights.copy()
    initial_bias = bias

    write_to_log("\nInitialized Weights:\n {}".format(initial_weights))
    write_to_log("\nInitialized Bias:\n {}".format(initial_bias))

    # write_to_file()

    # train(training_dataset)
