import torch.nn as nn


class Net_rand(nn.Module):
    def __init__(self):
        self.input_size = 100
        # Количество узлов на скрытом слое
        self.num_classes = 50  # Число классов на выходе. В этом случае от 0 до 9
        # self.num_epochs = 10**5  # Количество тренировок всего набора данных
        # self.batch_size = 100  # Размер входных данных для одной итерации
        # self.learning_rate = 0.001  # Скорость конвергенции
        # -----------------------------------------------------------
        super(Net_rand, self).__init__()  # Наследуемый родительским классом nn.Module
        self.fc1 = nn.Linear(self.input_size,
                             1000)  # 1й связанный слой: 784 (данные входа) -> 500 (скрытый узел)
        self.fc2 = nn.Linear(1000,
                             5000)
        self.fc3 = nn.Linear(5000,
                             100)
        self.fc4 = nn.Linear(100,
                             self.num_classes)
        self.relu = nn.ReLU()  # Нелинейный слой ReLU max(0,x)

    def out(self):
        return self.num_classes

    def inp(self):
        return self.input_size

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


class Net_detection(nn.Module):
    def __init__(self):
        self.input_size = 50
        # Количество узлов на скрытом слое
        self.num_classes = 1  # Число классов на выходе. В этом случае от 0 до 9
        # self.num_epochs = 10**5  # Количество тренировок всего набора данных
        # self.batch_size = 100  # Размер входных данных для одной итерации
        # self.learning_rate = 0.001  # Скорость конвергенции
        # -----------------------------------------------------------
        super(Net_detection, self).__init__()  # Наследуемый родительским классом nn.Module
        self.fc1 = nn.Linear(self.input_size,
                             500)  # 1й связанный слой: 784 (данные входа) -> 500 (скрытый узел)
        self.fc2 = nn.Linear(500,
                             100)
        self.fc3 = nn.Linear(100,
                             self.num_classes)
        self.relu = nn.ReLU()  # Нелинейный слой ReLU max(0,x)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x
