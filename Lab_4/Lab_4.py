import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_data(l_bound, h_bound):
    x_values = np.arange(l_bound, h_bound, 0.1)
    y_values = np.sin(np.abs(x_values)) * np.cos(3 * x_values / 2)
    z_values = x_values * np.sin(x_values + y_values)

    X = torch.tensor(np.stack((x_values, y_values), axis=1), dtype=torch.float).to(device)
    y = torch.tensor(z_values, dtype=torch.float).to(device)

    return X, y, x_values, z_values

def train_and_test_model(model, x_train, y_train, x_test, y_test, epochs):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_loss = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train.unsqueeze(1))
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        test_loss = loss_fn(y_pred, y_test.unsqueeze(1)).item()

    return train_loss, test_loss, y_pred

def plot_training_loss(train_loss, title):
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def plot_predictions(x_values, true_values, predicted_values, title):
    plt.figure()
    plt.plot(x_values, true_values, label='True Values', color='blue')
    plt.plot(x_values, predicted_values, label='Predicted Values', color='red')
    plt.title(title)
    plt.xlabel('Input X')
    plt.ylabel('Output Z')
    plt.legend()
    plt.grid()
    plt.show()

def plot_comparison(x_values, true_values, predicted_values, network_type, hidden_units_or_layers):
    plt.figure()
    plt.plot(x_values, true_values, label='True Function', color='blue')
    plt.plot(x_values, predicted_values, label='Predicted Function', color='red', linestyle='dashed')
    plt.title(f'{network_type} - {hidden_units_or_layers}')
    plt.xlabel('Input X')
    plt.ylabel('Output Z')
    plt.legend()
    plt.grid()
    plt.show()

class FFN(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_units)
        self.hidden_layer = nn.Linear(hidden_units, hidden_units)
        self.output_layer = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

class CFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(layers):
            self.hidden_layers.append(nn.Linear(input_dim + i * hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(input_dim + layers * hidden_dim, 1)

    def forward(self, x):
        outputs = [x]
        for layer in self.hidden_layers:
            x = F.relu(layer(torch.cat(outputs, dim=1)))
            outputs.append(x)
        return self.output_layer(torch.cat(outputs, dim=1))

class ElmanRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, nonlinearity='relu', batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(device)
        out, _ = self.rnn(x.unsqueeze(1), h0)
        out = self.output_layer(out[:, -1, :])
        return out

x_train, y_train, train_x_values, train_z_values = generate_data(20, 60)
x_test, y_test, test_x_values, test_z_values = generate_data(10, 40)

print("Task 1: Feed-Forward Networks")
ffn_a = FFN(10).to(device)
train_loss, test_loss, y_pred = train_and_test_model(ffn_a, x_train, y_train, x_test, y_test, epochs=200)
plot_training_loss(train_loss, "FFN: 10 Neurons")
plot_predictions(test_x_values, test_z_values, y_pred.cpu().numpy().flatten(), "FFN: 10 Neurons Predictions")
plot_comparison(test_x_values, test_z_values, y_pred.cpu().numpy().flatten(), "FFN", "10 Neurons")
print("Test Loss (10 neurons):", test_loss)

ffn_b = FFN(20).to(device)
train_loss, test_loss, y_pred = train_and_test_model(ffn_b, x_train, y_train, x_test, y_test, epochs=200)
plot_training_loss(train_loss, "FFN: 20 Neurons")
plot_predictions(test_x_values, test_z_values, y_pred.cpu().numpy().flatten(), "FFN: 20 Neurons Predictions")
plot_comparison(test_x_values, test_z_values, y_pred.cpu().numpy().flatten(), "FFN", "20 Neurons")
print("Test Loss (20 neurons):", test_loss)

print("Task 2: Cascade-Forward Networks")
cfn_a = CFN(2, 20, 1).to(device)
train_loss, test_loss, y_pred = train_and_test_model(cfn_a, x_train, y_train, x_test, y_test, epochs=200)
plot_training_loss(train_loss, "CFN: 1 Layer, 20 Neurons")
plot_predictions(test_x_values, test_z_values, y_pred.cpu().numpy().flatten(), "CFN: 1 Layer, 20 Neurons Predictions")
plot_comparison(test_x_values, test_z_values, y_pred.cpu().numpy().flatten(), "CFN", "1 Layer, 20 Neurons")
print("Test Loss (1 layer, 20 neurons):", test_loss)

cfn_b = CFN(2, 10, 2).to(device)
train_loss, test_loss, y_pred = train_and_test_model(cfn_b, x_train, y_train, x_test, y_test, epochs=200)
plot_training_loss(train_loss, "CFN: 2 Layers, 10 Neurons Each")
plot_predictions(test_x_values, test_z_values, y_pred.cpu().numpy().flatten(), "CFN: 2 Layers, 10 Neurons Each Predictions")
plot_comparison(test_x_values, test_z_values, y_pred.cpu().numpy().flatten(), "CFN", "2 Layers, 10 Neurons Each")
print("Test Loss (2 layers, 10 neurons each):", test_loss)

print("Task 3: Elman Networks")
rnn_a = ElmanRNN(2, 15, 1).to(device)
train_loss, test_loss, y_pred = train_and_test_model(rnn_a, x_train, y_train, x_test, y_test, epochs=200)
plot_training_loss(train_loss, "Elman RNN: 1 Layer, 15 Neurons")
plot_predictions(test_x_values, test_z_values, y_pred.cpu().numpy().flatten(), "Elman RNN: 1 Layer, 15 Neurons Predictions")
plot_comparison(test_x_values, test_z_values, y_pred.cpu().numpy().flatten(), "Elman RNN", "1 Layer, 15 Neurons")
print("Test Loss (1 layer, 15 neurons):", test_loss)

rnn_b = ElmanRNN(2, 5, 3).to(device)
train_loss, test_loss, y_pred = train_and_test_model(rnn_b, x_train, y_train, x_test, y_test, epochs=200)
plot_training_loss(train_loss, "Elman RNN: 3 Layers, 5 Neurons Each")
plot_predictions(test_x_values, test_z_values, y_pred.cpu().numpy().flatten(), "Elman RNN: 3 Layers, 5 Neurons Each Predictions")
plot_comparison(test_x_values, test_z_values, y_pred.cpu().numpy().flatten(), "Elman RNN", "3 Layers, 5 Neurons Each")
print("Test Loss (3 layers, 5 neurons each):", test_loss)
