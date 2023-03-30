import numpy as np
from tsai.all import *
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error


#Calculates a simpel pesudo return by "buying" if the prediction is 1, and keeping it for "lag" time periods before selling
#Add plots immediately
def calculate_pseudo_return(learner, X, y, splits, df, lag, sequence_length, TI, CLF, buckets = -1, transaction_cost=0):
    
    test_x = X[splits[1]] 
    preds, _, y_preds = learner.get_X_preds(test_x)

    predictions = [1 if y_preds[i][0] > y_preds[i-1][0] else -1 for i in range(len(y_preds))] if CLF else [-1 if y < buckets else 1 for y in y_preds]
    
    TI_index = 23 if TI else 0
 
    prices = list(df[sequence_length+TI_index:-lag]['close'][splits[1]])

    investment = 1000
    returns = []
    for i in range(len(predictions)-lag):
        if predictions[i] == 1: #if up, we buy
            buy_price = prices[i]
            sell_price = prices[i+lag]
            transaction_fee = transaction_cost * buy_price
            return_on_trade = (sell_price - buy_price) - transaction_fee
            returns.append(return_on_trade)
            investment += return_on_trade
        else:
            returns.append(0)
    cumulative_return = sum(returns)
    total_return = cumulative_return / investment

    roi = [sum(returns[:i+1]) for i in range(len(returns))]
    plt.plot(roi)
    plt.title("Cumulative returns at various timesteps, total return: "+"{:.2%}".format(total_return))
    plt.ylabel("P/L (USD)")
    plt.xlabel("Timestep")
    plt.show()

    return total_return, investment, returns 

#Plots the actual vs predict curves, as well as an "up-down" confusion matrix
def get_plots_regression(learner, X, y, splits):
    test_x = X[splits[1]] 
    test_y = y[splits[1]]

    preds, _, y_preds = learner.get_X_preds(test_x)

    test_y_converted = [1 if test_y[i] > test_y[i-1] else -1 for i in range(len(test_y))]
    preds_y_converted = [1 if y_preds[i][0] > y_preds[i-1][0] else -1 for i in range(len(y_preds))]

    y_preds_list = [y[0] for y in y_preds]

    accuracy = sum(y1 * y2 > 0 for y1, y2 in zip(preds_y_converted, test_y_converted))/len(preds_y_converted)

    ConfusionMatrixDisplay.from_predictions(test_y_converted, preds_y_converted, display_labels=range(2), normalize = 'true')
    plt.title('Total accuracy: '+str(accuracy))
    plt.show()

    # Calculate the cumulative accuracy at each step
    accuracies = []
    for i in range(len(preds_y_converted)):
        accuracy = sum(y1 * y2 > 0 for y1, y2 in zip(preds_y_converted[:i+1], test_y_converted[:i+1])) / (i+1)
        accuracies.append(accuracy)

    # Plot the accuracies against the step numbers
    plt.plot(range(1, len(accuracies)+1), accuracies)
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Accuracy at different steps')
    plt.show()

    #Plot regression line
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_preds_list, label='Prediction')
    ax.plot(test_y, label='Actual')

    # Set axis labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Prediction vs. Actual')

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_y, y_preds_list))
    plt.text(0.5, 0.95, f"RMSE: {rmse:.5f}", ha="center", va="center", transform=plt.gca().transAxes)
    
    ax.legend()
    plt.show()

#Plots both the actual CM, as well as a binary representation
def get_plots_classification(learner, X, y, splits, buckets):
    test_x = X[splits[1]] 
    test_y = y[splits[1]]

    preds, _, y_preds = learner.get_X_preds(test_x)

    test_binary = [-1 if y < buckets else 1 for y in test_y]
    pred_binary = [-1 if y < buckets else 1 for y in y_preds]

    accuracy = sum(y1 * y2 > 0 for y1, y2 in zip(pred_binary, test_binary))/len(pred_binary)

     # Calculate the cumulative accuracy at each step
    accuracies = []
    for i in range(len(pred_binary)):
        accuracy = sum(y1 * y2 > 0 for y1, y2 in zip(pred_binary[:i+1], test_binary[:i+1])) / (i+1)
        accuracies.append(accuracy)

    # Plot the accuracies against the step numbers
    plt.plot(range(1, len(accuracies)+1), accuracies)
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Accuracy at different steps')
    plt.show()

    ConfusionMatrixDisplay.from_predictions(test_y, y_preds, display_labels=range(2*buckets), normalize = 'true')
    plt.title('Actual classification, Total accuracy: '+str(accuracy))
    plt.show()

    ConfusionMatrixDisplay.from_predictions(test_binary , pred_binary, display_labels=range(2), normalize = 'true')
    plt.title('Binary classification, Total accuracy: '+str(accuracy))
    plt.show()

    plt.hist(y_preds)
    plt.title('Histogram of predictions')
    plt.show()

    predictions = ([1 if y1 * y2 > 0 else -1 for y1, y2 in zip(pred_binary, test_binary)])
    #steps = int(len(predictions) / 10)
    #predictions = [sum(x for x in predictions[i:i+steps] if x == 1)/steps for i in range(0, len(predictions), steps)]
    cumulative_sum = [sum(predictions[i-1:i]/i) for i in range(1, len(predictions)+1)]

    plt.plot(cumulative_sum)
    plt.title('Accuracy over time')
    plt.show()