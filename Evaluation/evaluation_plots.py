import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_squared_error


def plot_accuracy_vs_threshold(predictions, labels, increments = 0.1, threshold_value = 2.2):

    predictions = np.array(predictions)
    labels = np.array(labels)

    # Define the range of thresholds to test
    thresholds = np.round(np.arange(increments, threshold_value + increments, increments),1)

    # Initialize a list to store the accuracy values for each threshold
    accuracy_list = []

    # Loop over the range of thresholds and calculate the accuracy for each
    for threshold in thresholds:

        keep_indices = np.where(np.logical_or(predictions > threshold, predictions < -threshold))[0]

        new_predictions  = predictions[keep_indices]
        new_labels = labels[keep_indices]

        count = np.count_nonzero(np.array(new_predictions) * np.array(new_labels) > 0)
        
        if count == 0: 
            accuracy = 0 
        else: 
            accuracy = count / len(new_labels)

        accuracy_list.append(accuracy)


    # Create a grid of subplots
    fig = plt.figure(figsize=(10, 4))
    gs = GridSpec(1, 2, width_ratios=[3, 1])

    # Set the title above both the table and the plot
    fig.suptitle('Accuracy vs. Threshold', fontsize=16, fontweight='bold')

    # Plot the accuracy values as a line
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(thresholds[:len(accuracy_list)], accuracy_list)
    ax0.set_xlabel('Threshold')
    ax0.set_ylabel('Accuracy')

    # Display the accuracy values as a table
    data = {'Threshold': thresholds[:len(accuracy_list)], 'Accuracy': np.round(accuracy_list, 3)}
    df = pd.DataFrame(data)
    n = 10
    step = len(accuracy_list) // n
    df_sampled = df.iloc[::step, :]
    ax1 = fig.add_subplot(gs[1])
    table = ax1.table(cellText=df_sampled.values, colLabels=df_sampled.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    ax1.axis('off')

    plt.show()

def plot_predictions_vs_labels(predictions, labels):
    
    # Calculate accuracy and MSE
    count = np.count_nonzero(np.array(predictions) * np.array(labels) > 0)
    accuracy = count / len(predictions)
    mse = mean_squared_error(labels, predictions)

    # Create a grid of subplots
    fig = plt.figure(figsize=(10, 4))
    gs = GridSpec(1, 2, width_ratios=[3, 1])

    # Set the title above both the table and the plot
    fig.suptitle('Predictions vs Labels', fontsize=16, fontweight='bold')

    # Plot the Predictions vs Labels
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(predictions[-100:], label='predictions')
    ax0.plot(labels[-100:], label='labels')
    ax0.set_xlabel('Time')
    ax0.set_ylabel('Value')

    table_data = [['MSE', 'Accuracy'],
                [np.round(mse,3), np.round(accuracy,3)]]

    ax1 = fig.add_subplot(gs[1])
    table = ax1.table(cellText=table_data, loc='center')
    table.set_fontsize(12)
    table.scale(1, 2)
    ax1.axis('off')

    plt.show()