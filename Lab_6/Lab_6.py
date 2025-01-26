import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def prepare_data(file_name):
    df = pd.read_csv(file_name)
    print("Data preview:")
    print(df.head())

    X = df[['Open', 'High', 'Low']]
    y = df['Close']

    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train.unsqueeze(1), y_test.unsqueeze(1)


def create_hybrid_model(input_size):
    class HybridNeuroFuzzyModel(nn.Module):
        def __init__(self):
            super(HybridNeuroFuzzyModel, self).__init__()
            self.layer1 = nn.Linear(input_size, 64)
            self.layer2 = nn.Linear(64, 128)
            self.layer3 = nn.Linear(128, 64)
            self.output_layer = nn.Linear(64, 1)

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = torch.relu(self.layer3(x))
            x = self.output_layer(x)
            return x

    return HybridNeuroFuzzyModel()


def train_model(model, X_train, y_train, X_test, y_test, epochs, learning_rate):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = []
    test_loss = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test)
            test_loss.append(loss_fn(y_test_pred, y_test).item())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss[-1]:.4f}")

    return train_loss, test_loss


def visualize_results(train_loss, test_loss, y_test, y_test_pred):
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(y_test.numpy(), label='Actual', marker='o')
    plt.plot(y_test_pred.numpy(), label='Predicted', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Close Value')
    plt.title('Predicted vs Actual Close Values')
    plt.legend()
    plt.grid()
    plt.show()


file_name = 'Lab_6\NASDAQ Composite.csv'
X_train, X_test, y_train, y_test = prepare_data(file_name)


model = create_hybrid_model(input_size=3)
train_loss, test_loss = train_model(model, X_train, y_train, X_test, y_test, epochs=500, learning_rate=0.001)


model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)


visualize_results(train_loss, test_loss, y_test, y_test_pred)


last_row = torch.tensor([[13785.3, 13984.5, 13745.6]], dtype=torch.float32)
model.eval()
with torch.no_grad():
    prediction = model(last_row)
    print(f"Predicted Close Value for 2023-11-28: {prediction.item():.2f}")
