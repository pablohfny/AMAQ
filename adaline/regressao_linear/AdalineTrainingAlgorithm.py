from datetime import datetime
import math
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
        return pd.read_csv(file_path, delimiter=',',
                           header=0, dtype="float64")
    except FileNotFoundError:
        errorMessage = 'Training File not found at: {}'.format(
            file_path)
        print(errorMessage)
        sys.exit()


def get_initial_weights(dataframe):
    weights = []
    num_columns = len(dataframe.columns[:-1])

    for i in range(num_columns):
        weights.append(round(random.uniform(0.0, 0.5), 4))
    return weights


def calculate_output(weights, inputs, bias):
    y = 0
    for i in range(len(inputs)):
        y += weights[i] * inputs[i]
    y += bias
    return round(y, 7)


def update_weights(weights, expected_output, calculated_output, alpha, inputs):
    for i in range(len(weights)):
        weights[i] += alpha * (expected_output - calculated_output) * inputs[i]
    return weights


def train(training_dataset, weights, bias, alpha, epsilon):
    cycles = 0
    quadratic_errors = []
    quadratic_error = 0
    while(True):
        cycles += 1
        last_quadratic_error = quadratic_error
        quadratic_error = 0
        for row in training_dataset:
            inputs = row[:-1]
            expected_output = row[-1]
            calculated_output = calculate_output(weights, inputs, bias)
            quadratic_error += (expected_output - calculated_output) ** 2
            weights = update_weights(
                weights, expected_output, calculated_output, alpha, inputs)
            bias += alpha * (expected_output - calculated_output)
        quadratic_errors.append(quadratic_error)
        if(math.sqrt((quadratic_error - last_quadratic_error)**2) <= epsilon):
            break

    plt.figure()
    plt.plot(quadratic_errors, color='red')
    plt.title('Erro Quadratico X Ciclos')
    plt.xlabel('Ciclos')
    plt.ylabel('Erro Quadratico')
    plt.show()

    return [weights, bias, quadratic_error, cycles]


if __name__ == "__main__":
    write_to_log("-----------Started Script-----------")

    training_dataframe = import_dataframe(TRAINING_DATASET_CSV_FILE_PATH)
    write_to_log("\nImported Training DataFrame:\n {}".format(
        training_dataframe))

    training_dataset = training_dataframe.values
    weights = get_initial_weights(training_dataframe)
    bias = round(random.uniform(0.0, 0.5), 4)
    alpha = 0.0025
    epsilon = 1e-06

    write_to_log("\nInitial Weights:\n {}".format(weights))
    write_to_log("\nInitial Bias:\n {}".format(bias))
    write_to_log("\nAlpha:\n {}".format(alpha))
    write_to_log("\nepsilon:\n {}".format(epsilon))

    [weights, bias, quadratic_error, cycles] = train(
        training_dataset, weights, bias, alpha, epsilon)

    inputs = training_dataframe.iloc[:, :-1].values
    outputs = training_dataframe.iloc[:, -1:].values

    plt.figure()
    plt.plot(inputs, outputs, 'bo')
    plt.title('X vs Y')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()

    line_x_values = np.linspace(-1, 6, 100)
    line_equation = weights[0] * line_x_values + bias
    line_equation_representation = "y={}*x+{}".format(weights[0], bias)
    plt.plot(line_x_values, line_equation, '-r',
             label=line_equation_representation)
    plt.show()

    write_to_log("\nNew Weights:\n {}".format(weights))
    write_to_log("\nNew Bias:\n {}".format(bias))
    write_to_log("\nQuadratic Error:\n {}".format(quadratic_error))
    write_to_log("\nCycles:\n {}".format(cycles))
    write_to_log("\nLine equation:\n {}".format(line_equation_representation))
