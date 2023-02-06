
import torch.optim as optim
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


def Evaluator_LSTM(model, data, learning_rate: int, num_epochs :int, visualize:bool, model_text:str):

    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []

    # Train the model
    for epoch in range(num_epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Evaluate the model
    with torch.no_grad():
        y_pred = model(X_test)
        accuracy = sum(y1 * y2 >= 0 for y1, y2 in zip(y_pred, y_test))/len(y_pred)
        print(f'Accuracy: {accuracy[0]}')

    if visualize:

        last_y_pred = y_pred[int(0.8*len(y_pred)):]
        last_y_test = y_test[int(0.8*len(y_pred)):]

        mse = torch.mean((last_y_test - last_y_pred)**2)
        fig, axs = plt.subplots(1, 3, figsize=(25, 8))

        axs[0].plot(last_y_pred, label = 'Prediction', color='red')
        axs[0].plot(last_y_test, label = 'Data', color='green')
        axs[0].set_title('Predictions vs Data')
        axs[0].set_xlabel('Number of predictions')
        axs[0].set_ylabel('Prediction')
        axs[0].legend(title='MSE: {:.4f}'.format(mse))

        axs[1].plot(torch.sign(last_y_pred*last_y_test))
        axs[1].set_title('Correct vs incorrect')
        axs[1].set_xlabel('Number of predictions')
        axs[1].set_ylabel('Prediction')

        axs[2].plot(loss_list[int(0.8*len(loss_list)):], color='red')
        axs[2].set_title('Loss over Epochs')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Loss')

        fig.suptitle(f"LSTM Model: {model_text} Accuracy = {accuracy[0]*100:.2f}%", fontsize=15)
        plt.show()

    return loss.item()
