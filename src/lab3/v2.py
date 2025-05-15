import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Параметри
input_size = 28 * 28  # розмір зображення (28x28)
hidden_size = 128     # кількість нейронів у прихованому шарі
num_classes = 10      # цифри від 0 до 9
batch_size = 64
learning_rate = 0.005
epochs = 10

# 2. Завантаження MNIST
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 3. Модель нейронної мережі
class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, input_size)  # розгортаємо 28x28 у 784
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = FeedforwardNN()

# 4. Функція втрат і оптимізатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# 5. Навчання моделі
for epoch in range(epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 6. Тестування
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Точність на тестовому наборі: {100 * correct / total:.2f}%")

# 7. Візуалізація прикладів
examples = iter(test_loader)
images, labels = next(examples)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i][0], cmap='gray')
    plt.title(f"Label: {labels[i]}\nPred: {predicted[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
