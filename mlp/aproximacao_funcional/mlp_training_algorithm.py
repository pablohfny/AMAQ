# Pablo Nunes 11411ECP001 17/05/2021 MLP
from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys

EXECUTION_DATETIME = datetime.now()
FORMATTED_EXECUTION_DATETIME = "{}T{}".format(
    EXECUTION_DATETIME.strftime("%x").replace("/", "-"),
    EXECUTION_DATETIME.strftime("%X").replace(":", "_"))
CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
TRAINING_DATASET_CSV_FILE_PATH = CURRENT_DIR_PATH + "\\data\\training_dataset.csv"
LOGS_PATH = CURRENT_DIR_PATH + "\\logs"

ALPHA = 3e-2
EPSILON = 1e-3
MAX_CYCLES = 10000
NUM_INPUT_NEURONS = 2
NUM_HIDDEN_NEURONS = 4
NUM_OUTPUT_NEURONS = 1

HIDDEN_WEIGHTS = [[round(random.uniform(-0.5, 0.5), 4)
                   for i in range(NUM_HIDDEN_NEURONS)]
                  for j in range(NUM_INPUT_NEURONS)]
HIDDEN_BIAS = [round(random.uniform(-0.5, 0.5), 4)
               for i in range(NUM_HIDDEN_NEURONS)]
HIDDEN_WEIGHTS_DERIVATIVE_DELTA = [0 for j in range(NUM_HIDDEN_NEURONS)]

OUTPUT_WEIGHTS = [[round(random.uniform(-0.5, 0.5), 4)
                   for i in range(NUM_OUTPUT_NEURONS)]
                  for j in range(NUM_HIDDEN_NEURONS)]
OUTPUT_WEIGHTS_DELTA = [0 for i in range(NUM_HIDDEN_NEURONS)]


def write_to_log(value):
    if not os.path.exists(LOGS_PATH):
        os.makedirs(LOGS_PATH)

    log_file_path = "{}\\{}.log".format(
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


def activate_function(value):
    return (2/(1 + math.exp(-value)))-1


def derivative_activate_function(value):
    return 0.5 * (1 + value) * (1 - value)


def calculate_hidden_inputs(inputs):
    return [[inputs[i] * weight for weight in HIDDEN_WEIGHTS[i]] for i in range(NUM_INPUT_NEURONS)]


def calculate_hidden_outputs(hidden_inputs):
    return [activate_function(sum(hidden_inputs[i][j] for i in range(NUM_INPUT_NEURONS)) + HIDDEN_BIAS[j]) for j in range(NUM_HIDDEN_NEURONS)]


def calculate_output_layer_input(hidden_outputs, output_bias):
    return sum(sum(np.transpose(hidden_outputs) * OUTPUT_WEIGHTS)) + output_bias


def calculate_output_derivative_delta(expected_output, calculated_output):
    return (expected_output - calculated_output) \
        * derivative_activate_function(calculated_output)


def calculate_output_weight_deltas(output_derivative_delta, hidden_outputs):
    return [(ALPHA * output_derivative_delta * hidden_outputs[i]) for i in range(len(hidden_outputs))]


def calculate_output_bias_delta(output_derivative_delta):
    return ALPHA * output_derivative_delta


def calculate_hidden_derivative_weight_deltas(output_derivative_delta, hidden_outputs):
    return [sum(output_derivative_delta * weight for weight in OUTPUT_WEIGHTS[i]) * derivative_activate_function(hidden_outputs[i]) for i in range(len(hidden_outputs))]


def calculate_hidden_weight_deltas(hidden_derivative_weight_deltas, inputs):
    return [[(ALPHA * np.transpose(hidden_derivative_weight_deltas[j]) * inputs[i]) for j in range(NUM_HIDDEN_NEURONS)]
            for i in range(NUM_INPUT_NEURONS)]


def calculate_hidden_bias_delta(hidden_derivative_weight_deltas):
    return [ALPHA * delta for delta in hidden_derivative_weight_deltas]


def update_output_weights(output_weight_deltas):
    for i in range(len(OUTPUT_WEIGHTS)):
        OUTPUT_WEIGHTS[i] += output_weight_deltas[i]


def update_output_bias(output_bias, output_bias_delta):
    return output_bias + output_bias_delta


def update_hidden_weights(hidden_weight_deltas):
    for i in range(NUM_INPUT_NEURONS):
        for j in range(NUM_HIDDEN_NEURONS):
            HIDDEN_WEIGHTS[i][j] += hidden_weight_deltas[i][j]


def update_hidden_bias(hidden_bias_deltas):
    for i in range(len(HIDDEN_BIAS)):
        HIDDEN_BIAS[i] += hidden_bias_deltas[i]


def train(training_dataset):
    output_bias = round(random.uniform(-0.5, 0.5), 4)
    cycles = 0
    total_error = EPSILON
    total_error_array = []
    while cycles < MAX_CYCLES and total_error >= EPSILON:
        total_error = 0
        cycles += 1
        for row in training_dataset:
            inputs = row[:-1]
            expected_output = row[-1]

            hidden_inputs = calculate_hidden_inputs(inputs)
            hidden_outputs = calculate_hidden_outputs(hidden_inputs)

            output_layer_input = calculate_output_layer_input(
                hidden_outputs, output_bias)
            calculcated_output = activate_function(output_layer_input)

            output_derivative_delta = calculate_output_derivative_delta(
                expected_output, calculcated_output)
            output_weight_deltas = calculate_output_weight_deltas(
                output_derivative_delta, hidden_outputs)
            output_bias_delta = calculate_output_bias_delta(
                output_derivative_delta)

            hidden_derivative_weight_deltas = calculate_hidden_derivative_weight_deltas(
                output_derivative_delta, hidden_outputs)
            hidden_weight_deltas = calculate_hidden_weight_deltas(
                hidden_derivative_weight_deltas, inputs)
            hidden_bias_delta = calculate_hidden_bias_delta(
                hidden_derivative_weight_deltas)

            update_output_weights(output_weight_deltas)
            output_bias = update_output_bias(output_bias, output_bias_delta)

            update_hidden_weights(hidden_weight_deltas)
            update_hidden_bias(hidden_bias_delta)

            total_error += 0.5 * pow((expected_output - calculcated_output), 2)
        total_error_array.append(total_error)
    plt.figure()
    plt.plot(total_error_array, color='red')
    plt.title('Erro Quadratico X Ciclos')
    plt.xlabel('Ciclos')
    plt.ylabel('Erro Quadratico')
    plt.show()
    return [total_error_array, cycles, output_bias]


def test(testing_dataframe, output_bias):
    write_to_log("\nTest DataFrame:\n {}".format(
        testing_dataframe))
    write_to_log("-----------Started Test-----------")
    for row in testing_dataframe.values:
        inputs = row[:-1]
        write_to_log("\nInputs:\n {}".format(inputs))
        hidden_inputs = calculate_hidden_inputs(inputs)
        hidden_outputs = calculate_hidden_outputs(hidden_inputs)
        output_layer_input = calculate_output_layer_input(
            hidden_outputs, output_bias)
        calculcated_output = activate_function(output_layer_input)
        write_to_log("\nCalculated Output:\n {}".format(calculcated_output))


if __name__ == "__main__":
    write_to_log("\n-----------Started Training Script-----------\n")
    training_dataframe = import_dataframe(TRAINING_DATASET_CSV_FILE_PATH)
    write_to_log("\nImported Training DataFrame:\n {}".format(
        training_dataframe))
    write_to_log("\nAlpha:\n {}".format(ALPHA))
    write_to_log("\nEpsilon:\n {}".format(EPSILON))

    write_to_log("\nInitial Hidden Weights:\n {}".format(HIDDEN_WEIGHTS))
    write_to_log("\nInitial Hidden Bias:\n {}".format(HIDDEN_BIAS))

    write_to_log("\nInitial Output Weights:\n {}".format(OUTPUT_WEIGHTS))

    training_dataset = training_dataframe.values
    [total_error_array, cycles, output_bias] = train(training_dataset)

    inputs = training_dataset.iloc[:,:-1].values
    outputs = training_dataset.iloc[:,-1:].values

    plt.figure()
    plt.plot(inputs, outputs, 'bo')
    plt.title('X vs Y')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()

    write_to_log("\nFinal Hidden Weights:\n {}".format(HIDDEN_WEIGHTS))
    write_to_log("\nFinal Hidden Bias:\n {}".format(HIDDEN_BIAS))

    write_to_log("\nFinal Output Weights:\n {}".format(OUTPUT_WEIGHTS))
    write_to_log("\nFinal Output Bias:\n {}".format(output_bias))

    write_to_log("\nTotal Quadratic Errors:\n {}".format(total_error_array))
    write_to_log("\nCycles:\n {}".format(cycles))
    write_to_log("\n-----------Ended Training Script-----------\n")
    test(training_dataframe, output_bias)