import numpy as np
import matplotlib.pyplot as plt

def data_x():
    l = [0, 1, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1]

    i = [0,0, 1, 0, 0, 0,
         0, 0, 1, 0, 0, 0,
         0, 0, 1, 0, 0, 0,
         0, 0, 1, 0, 0, 0,
         0, 0, 1, 0, 0, 0]

    z = [1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 1, 1,
         0, 0, 1, 1, 0, 0,
         1, 1, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1]

    a = [0, 0, 1, 1, 0, 0,
         0, 1, 0, 0, 1, 0,
         1, 1, 1, 1, 1, 1,
         1, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 1]

    images = [np.array(l), np.array(i), np.array(z), np.array(a)]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    colors = ['Reds', 'Blues', 'Greens', 'Purples']
    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(5, 6), cmap=colors[i])
        ax.set_title(['L', 'I', 'Z', 'A'][i])
        ax.axis('off')
    plt.show()

    x = [image.reshape(1, 30) for image in images]
    return np.vstack(x)

def data_y():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

class HebbNetwork:
    def __init__(self, input_size, output_size):
        self.weights = np.zeros((output_size, input_size))

    def train(self, x, y):
        for xi, yi in zip(x, y):
            self.weights += np.outer(yi, xi)

    def predict(self, x):
        output = np.dot(self.weights, x.T)
        return np.where(output == np.max(output, axis=0), 1, 0).T

x = data_x()
y = data_y()

network = HebbNetwork(input_size=30, output_size=4)
network.train(x, y)

valid_letters = ['L', 'I', 'Z', 'A']

correct_predictions = []
incorrect_predictions = []
errors = []

for i, test_input in enumerate(x):
    prediction = network.predict(test_input)
    actual_letter = valid_letters[i]
    predicted_index = np.argmax(prediction)

    if predicted_index < len(valid_letters):
        predicted_letter = valid_letters[predicted_index]
    else:
        predicted_letter = "Unknown"

    if predicted_letter == actual_letter:
        correct_predictions.append((actual_letter, predicted_letter))
    else:
        incorrect_predictions.append((actual_letter, predicted_letter))
        errors.append((test_input, actual_letter, predicted_letter))

    print(f"Actual Letter: {actual_letter}, Predicted Letter: {predicted_letter}")
    plt.imshow(test_input.reshape(5, 6), cmap='cool')
    plt.title(f"Actual: {actual_letter}, Predicted: {predicted_letter}")
    plt.axis('off')
    plt.show()

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(network.weights[i].reshape(5, 6), cmap='coolwarm')
    ax.set_title(f'Weight {chr(65 + i)}')
    ax.axis('off')
plt.show()

accuracy = len(correct_predictions) / len(x)
error_rate = len(incorrect_predictions) / len(x)

plt.figure()
plt.bar(['Accuracy', 'Error Rate'], [accuracy, error_rate], color=['green', 'red'])
plt.title('Model Performance')
plt.ylabel('Rate')
plt.show()

if errors:
    fig, axes = plt.subplots(1, len(errors), figsize=(15, 5))
    for i, (error_input, actual, predicted) in enumerate(errors):
        ax = axes[i] if len(errors) > 1 else axes
        ax.imshow(error_input.reshape(5, 6), cmap='cool')
        ax.set_title(f"Actual: {actual}, Predicted: {predicted}")
        ax.axis('off')
    plt.show()
