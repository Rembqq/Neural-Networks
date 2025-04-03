import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Функція двох змінних
def z_function(x, y):
    return x ** 2 + y ** 2


# Генеруємо дані
x_vals = np.linspace(0, 2.5, 100)
y_vals = np.linspace(0, 2.5, 100)
X1, Y1 = np.meshgrid(x_vals, y_vals)
Z_vals = z_function(X1, Y1)
X = np.column_stack((X1.ravel(), Y1.ravel()))
Z = Z_vals.ravel()

# Розбиваємо дані на train/test
x_train, x_test, y_train, y_test = train_test_split(X, Z, test_size=0.1)

# Масштабуємо
scaler = StandardScaler()
x_train_t = torch.FloatTensor(scaler.fit_transform(x_train))
x_test_t = torch.FloatTensor(scaler.transform(x_test))
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)


# Feedforward Neural Network
class FeedNet(nn.Module):
    def __init__(self, hidden_layers):
        super(FeedNet, self).__init__()
        layers = []
        prev_size = 2
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Cascade Neural Network
class CascadeNet(nn.Module):
    def __init__(self, hidden_layers):
        super(CascadeNet, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_size = 2
        for hidden_size in hidden_layers:
            self.hidden_layers.append(nn.Linear(prev_size + 2, hidden_size))
            prev_size = hidden_size
        self.output = nn.Linear(prev_size + 2, 1)

    def forward(self, x):
        out = x
        for layer in self.hidden_layers:
            out = torch.cat([x, out], dim=1)
            out = torch.tanh(layer(out))
        out = torch.cat([x, out], dim=1)
        return self.output(out)


# Elman Neural Network
class ElmanNet(nn.Module):
    def __init__(self, hidden_sizes):
        super(ElmanNet, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.i2h = nn.ModuleList([nn.Linear(2 + hidden_size, hidden_size) for hidden_size in hidden_sizes])
        self.h2o = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x, hidden_states):
        combined = torch.cat((x, hidden_states[0]), 1)
        hidden_next = []
        for i, layer in enumerate(self.i2h):
            hidden_i = torch.tanh(layer(combined))
            combined = torch.cat((x, hidden_i), 1)
            hidden_next.append(hidden_i)
        return self.h2o(hidden_next[-1]), hidden_next

    def init_hidden(self, batch_size):
        return [torch.zeros(batch_size, hidden_size) for hidden_size in self.hidden_sizes]


# Функція навчання моделі
def train_model(model, epochs=5000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_curve = []
    for epoch in range(epochs):
        model.train()
        if isinstance(model, ElmanNet):
            hidden = model.init_hidden(x_train_t.size(0))
            output, _ = model(x_train_t, hidden)
        else:
            output = model(x_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_curve.append(loss.item())
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        if isinstance(model, ElmanNet):
            hidden = model.init_hidden(x_test_t.size(0))
            predictions, _ = model(x_test_t, hidden)
        else:
            predictions = model(x_test_t)
        epsilon = 1e-8  # Маленькое число, предотвращающее деление на ноль
        mre = torch.mean(torch.abs((y_test_t - predictions) / (y_test_t + epsilon))).item()

    return predictions, mre, loss_curve


# Конфігурація нейромереж
network_config = {
    '1 layer (10) FN': {'model': FeedNet, 'params': {'hidden_layers': [10]}},
    '1 layer (20) FN': {'model': FeedNet, 'params': {'hidden_layers': [20]}},
    '1 layer (20) CN': {'model': CascadeNet, 'params': {'hidden_layers': [20]}},
    '2 layers (10 each) CN': {'model': CascadeNet, 'params': {'hidden_layers': [10, 10]}},
    '1 layer (15) EN': {'model': ElmanNet, 'params': {'hidden_sizes': [15]}},
    '3 layers (5 each) EN': {'model': ElmanNet, 'params': {'hidden_sizes': [5, 5, 5]}},
}

# Навчання та оцінка моделей
for name, config in network_config.items():
    print(f"\nTraining configuration: {name}")
    model_instance = config['model'](**config['params'])
    predictions, mre, loss_curve = train_model(model_instance)
    print(f"MRE: {mre:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, predictions.numpy(), color='green')
    plt.plot([0, 10], [0, 10], color='black', linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{name}')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(loss_curve, color='black')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve: {name}')
    plt.grid()
    plt.show()
