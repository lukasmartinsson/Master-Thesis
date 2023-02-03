
import torch.optim as optim
import torch
import torch.nn as nn
from torch.autograd import Variable


def Evaluator_Transformer(model, data, learning_rate: int):

    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(100):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Evaluate the model
    with torch.no_grad():
        y_pred = model(X_test)
        accuracy = sum(y1 * y2 >= 0 for y1, y2 in zip(y_pred, y_test))/len(y_pred)
        print(f'Accuracy: {accuracy[0]}')