from datetime import datetime
import os
import pandas as pd
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
        return pd.read_csv(file_path, delimiter=',',
                           header=0, dtype="float64")
    except FileNotFoundError:
        errorMessage = 'Training File not found at: {}'.format(
            file_path)
        print(errorMessage)
        sys.exit()

if __name__ == "__main__":
    try:
        raise NotImplementedError
    except:
        print("Error occured") 