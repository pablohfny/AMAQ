# Pablo Nunes 11411ECP001 17/05/2021 MLP
from datetime import datetime
import os
import pandas as pd
import random
import sys

EXECUTION_DATETIME = datetime.now()
FORMATTED_EXECUTION_DATETIME = "{}T{}".format(
    EXECUTION_DATETIME.strftime("%x").replace("/", "-"),
    EXECUTION_DATETIME.strftime("%X"))
CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
TRAINING_DATASET_CSV_FILE_PATH = CURRENT_DIR_PATH + "/data/training_dataset.csv"
LOGS_PATH = CURRENT_DIR_PATH + "/logs"

ALPHA = 1e-3
EPSILON = 1e-4
MAX_CYCLES = 10000
NUM_INPUT_NEURONS = 1
NUM_HIDDEN_NEURONS = 4
NUM_OUTPUT_NEURONS = 1

HIDDEN_WEIGHTS = [[round(random.uniform(-0.5, 0.5), 4)
                   for i in range(NUM_HIDDEN_NEURONS)]
                  for j in range(NUM_INPUT_NEURONS)]
HIDDEN_BIAS = [round(random.uniform(-0.5, 0.5), 4)
               for i in range(NUM_HIDDEN_NEURONS)]
HIDDEN_WEIGHTS_DELTA = [0 for i in range(NUM_HIDDEN_NEURONS)]
HIDDEN_BIAS_DELTA = [0 for i in range(NUM_HIDDEN_NEURONS)]


OUTPUT_WEIGHTS = [[round(random.uniform(-0.5, 0.5), 4)
                   for j in range(NUM_OUTPUT_NEURONS)]
                  for k in range(NUM_HIDDEN_NEURONS)]
OUTPUT_BIAS = round(random.uniform(-0.5, 0.5), 4)
OUTPUT_WEIGHTS_DELTA = [0
                        for i in range(NUM_OUTPUT_NEURONS)]


def write_to_log(value):
    if not os.path.exists(LOGS_PATH):
        os.makedirs(LOGS_PATH)

    log_file_path = "{}/{}.log".format(
        LOGS_PATH, FORMATTED_EXECUTION_DATETIME)

    file = open(log_file_path, "a")
    file.write(value)
    file.close()

    print(value)


def import_dataframe(file_path):
    try:
        return pd.read_csv(file_path, delimiter=',',
                           header=None, dtype="float64")
    except FileNotFoundError:
        errorMessage = 'Training File not found at: {}'.format(
            file_path)
        print(errorMessage)
        sys.exit()

def train(training_dataset):
    raise NotImplementedError
    
if __name__ == "__main__":
    try:
        write_to_log("-----------Started Script-----------")
        training_dataframe = import_dataframe(TRAINING_DATASET_CSV_FILE_PATH)

        write_to_log("\nImported Training DataFrame:\n {}".format(
            training_dataframe))
        write_to_log("\nAlpha:\n {}".format(ALPHA))
        write_to_log("\nEpsilon:\n {}".format(EPSILON))

        write_to_log("\nInitial Hidden Weights:\n {}".format(HIDDEN_WEIGHTS))
        write_to_log("\nInitial Hidden Bias:\n {}".format(HIDDEN_BIAS))

        write_to_log("\nInitial Output Weights:\n {}".format(OUTPUT_WEIGHTS))
        write_to_log("\nInitial Output Bias:\n {}".format(OUTPUT_BIAS))
    except:
        print("Error occured")
