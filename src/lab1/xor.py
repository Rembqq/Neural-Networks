import numpy as np

# Функція активації (сигмоїда)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Похідна сигмоїди для оновлення ваг
def sigmoid_derivative(x):
    return x * (1 - x)

# Генерація навчальних даних для XOR із 4 змінними
X = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
])

# Правильні відповіді для функції XOR (чітка кількість 1)
y = np.array([[np.sum(row) % 2] for row in X])

# Ініціалізація ваг випадковими значеннями
input_size = 4
hidden_size1 = 6
hidden_size2 = 5
output_size = 1

np.random.seed(1)
weights_input_hidden1 = np.random.uniform(-1, 1, (input_size, hidden_size1))
weights_hidden1_hidden2 = np.random.uniform(-1, 1, (hidden_size1, hidden_size2))
weights_hidden2_output = np.random.uniform(-1, 1, (hidden_size2, output_size))

learning_rate = 0.5
epochs = 10000

# Навчання нейронної мережі
for epoch in range(epochs):
    # Прямий прохід
    hidden_layer1_input = np.dot(X, weights_input_hidden1)
    hidden_layer1_output = sigmoid(hidden_layer1_input)

    hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
    hidden_layer2_output = sigmoid(hidden_layer2_input)

    output_layer_input = np.dot(hidden_layer2_output, weights_hidden2_output)
    output_layer_output = sigmoid(output_layer_input)

    # Помилка
    error = y - output_layer_output

    # Зворотне поширення помилки
    d_output = error * sigmoid_derivative(output_layer_output)
    d_hidden2 = d_output.dot(weights_hidden2_output.T) * sigmoid_derivative(hidden_layer2_output)
    d_hidden1 = d_hidden2.dot(weights_hidden1_hidden2.T) * sigmoid_derivative(hidden_layer1_output)

    # Оновлення ваг
    weights_hidden2_output += hidden_layer2_output.T.dot(d_output) * learning_rate
    weights_hidden1_hidden2 += hidden_layer1_output.T.dot(d_hidden2) * learning_rate
    weights_input_hidden1 += X.T.dot(d_hidden1) * learning_rate

    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f"Епоха {epoch}, помилка: {loss}")

# Тестування
print("Результати після навчання:")
for i in range(len(X)):
    hidden_output1 = sigmoid(np.dot(X[i], weights_input_hidden1))
    hidden_output2 = sigmoid(np.dot(hidden_output1, weights_hidden1_hidden2))
    final_output = sigmoid(np.dot(hidden_output2, weights_hidden2_output))
    binary_output = 1 if final_output[0] >= 0.5 else 0  # Округлення до 0 або 1
    print(f"Вхід: {X[i]} -> Вихід: {round(final_output[0], 2)} ({binary_output})")