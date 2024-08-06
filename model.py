import torch.nn as nn
import pytorch_lightning as pl
import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cuda:0"


class Net_rand(nn.Module):
    def __init__(self):
        self.input_size = 500
        # Количество узлов на скрытом слое
        self.num_classes = 40 * 8  # Число классов на выходе. В этом случае от 0 до 9
        # self.num_epochs = 10**5  # Количество тренировок всего набора данных
        # self.batch_size = 100  # Размер входных данных для одной итерации
        # self.learning_rate = 0.001  # Скорость конвергенции
        # -----------------------------------------------------------
        super(Net_rand, self).__init__()  # Наследуемый родительским классом nn.Module
        self.fc1 = nn.Linear(self.input_size,
                             3000)  # 1й связанный слой: 784 (данные входа) -> 500 (скрытый узел)
        self.fc2 = nn.Linear(3000,
                             5000)
        self.fc3 = nn.Linear(5000,
                             5000)
        self.fc4 = nn.Linear(5000,
                             1000)
        self.fc5 = nn.Linear(1000,
                             self.num_classes)
        self.relu = nn.ReLU()  # Нелинейный слой ReLU max(0,x)
        self.sof = nn.Softmax()

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
        x = self.relu(x)
        x = self.fc5(x)
        # x = self.sof(x)
        return x


# class Net_rand(nn.Module):
#     def __init__(self):
#         self.input_size = 100
#         # Количество узлов на скрытом слое
#         self.num_classes = 40*8  # Число классов на выходе. В этом случае от 0 до 9
#         # self.num_epochs = 10**5  # Количество тренировок всего набора данных
#         # self.batch_size = 100  # Размер входных данных для одной итерации
#         # self.learning_rate = 0.001  # Скорость конвергенции
#         # -----------------------------------------------------------
#         super(Net_rand, self).__init__()  # Наследуемый родительским классом nn.Module
#         self.fc1 = nn.Linear(self.input_size,
#                              1000)
#         self.fc2 = nn.Linear(1000,
#                              1000)                 # 1й связанный слой: 784 (данные входа) -> 500 (скрытый узел)
#         self.fc3 = nn.Linear(1000,
#                              self.num_classes)
#         self.relu = nn.ReLU()  # Нелинейный слой ReLU max(0,x)
#     def out(self):
#         return self.num_classes

# def inp(self):
#     return self.input_size
#
# def forward(self, x):
#     x = self.fc1(x)
#     x = self.relu(x)
#     x = self.fc2(x)
#     x = self.relu(x)
#     x = self.fc3(x)
#     return x
class Net_detection(nn.Module):
    def __init__(self):
        self.input_size = 40 * 8
        # Количество узлов на скрытом слое
        self.num_classes = 1  # Число классов на выходе. В этом случае от 0 до 9
        # self.num_epochs = 10**5  # Количество тренировок всего набора данных
        # self.batch_size = 100  # Размер входных данных для одной итерации
        # self.learning_rate = 0.001  # Скорость конвергенции
        # -----------------------------------------------------------
        super(Net_detection, self).__init__()  # Наследуемый родительским классом nn.Module
        self.fc1 = nn.Linear(self.input_size,
                             1000)  # 1й связанный слой: 784 (данные входа) -> 500 (скрытый узел)
        self.fc2 = nn.Linear(1000,
                             3000)
        self.fc3 = nn.Linear(3000,
                             1000)
        self.fc4 = nn.Linear(1000,
                             100)
        self.fc5 = nn.Linear(100,
                             self.num_classes)
        self.relu = nn.ReLU()  # Нелинейный слой ReLU max(0,x)
        self.sof = nn.Softmax()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        # x = self.sof(x)
        # x = self.relu(x)
        return x

# class Net_detection(nn.Module):
#     def __init__(self):
#         self.input_size = 40 * 8
#         # Количество узлов на скрытом слое
#         self.num_classes = 1  # Число классов на выходе. В этом случае от 0 до 9
#         # self.num_epochs = 10**5  # Количество тренировок всего набора данных
#         # self.batch_size = 100  # Размер входных данных для одной итерации
#         # self.learning_rate = 0.001  # Скорость конвергенции
#         # -----------------------------------------------------------
#         super(Net_detection, self).__init__()  # Наследуемый родительским классом nn.Module
#         self.fc1 = nn.Linear(self.input_size,
#                              1000)  # 1й связанный слой: 784 (данные входа) -> 500 (скрытый узел)
#         self.fc2 = nn.Linear(1000,
#                              100)
#         self.fc3 = nn.Linear(100,
#                              self.num_classes)
#         self.relu = nn.ReLU()  # Нелинейный слой ReLU max(0,x)
#         self.sof = nn.Softmax()
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         # x = self.sof(x)
#         # x = self.relu(x)
#         return x


class GAN(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.generator = Net_rand()
        self.discriminator = Net_detection()
        # After each epoch, we generate 100 images using the noise
        # vector here (self.test_noises). We save the output images
        # in a list (self.test_progression) for plotting later.
        self.test_noises = torch.randn(100, 1, 100, device=device)
        self.test_progression = []

    def forward(self, x):
        """
        Generates an image using the generator
        given input noise z
        """
        return self.generator(x)

    def generator_step(self, x):
        """
        Training step for generator
        1. Sample random noise
        2. Pass noise to generator to
           generate images
        3. Classify generated images using
           the discriminator
        4. Backprop loss to the generator
        """

        # Sample noise
        z = torch.randn(x.shape[0], 1, 100, device=device)

        # Generate images
        generated_imgs = self(z)

        # Classify generated images
        # using the discriminator
        # Классифицировать сгенерированные изображения
        # использование дискриминатора
        d_output = torch.squeeze(self.discriminator(generated_imgs))

        # Backprop loss. We want to maximize the discriminator's
        # loss, which is equivalent to minimizing the loss with the true
        # labels flipped (i.e. y_true=1 for fake images). We do this
        # as PyTorch can only minimize a function instead of maximizing
        # Потеря обратной связи. Мы хотим максимизировать эффективность дискриминатора
        # потеря, что эквивалентно минимизации потери с помощью истинного
        # перевернутые метки (т.е. y_true=1 для поддельных изображений). Мы делаем это
        # поскольку PyTorch может только минимизировать функцию вместо того, чтобы максимизировать
        g_loss = nn.BCELoss()(d_output,
                              torch.ones(x.shape[0], device=device))

        return g_loss

    def discriminator_step(self, x):
        """
        Training step for discriminator
        1. Get actual images
        2. Predict probabilities of actual images and get BCE loss
        3. Get fake images from generator
        4. Predict probabilities of fake images and get BCE loss
        5. Combine loss from both and backprop loss to discriminator
        """

        # Real images
        d_output = torch.squeeze(self.discriminator(x))
        loss_real = nn.BCELoss()(d_output,
                                 torch.ones(x.shape[0], device=device))

        # Fake images
        z = torch.randn(x.shape[0], 1, 100, device=device)
        generated_imgs = self(z)
        d_output = torch.squeeze(self.discriminator(generated_imgs))
        loss_fake = nn.BCELoss()(d_output,
                                 torch.zeros(x.shape[0], device=device))

        return loss_real + loss_fake

    def training_step(self, batch, batch_idx, optimizer_idx):
        X, _ = batch

        # train generator
        if optimizer_idx == 0:
            loss = self.generator_step(X)

        # train discriminator
        if optimizer_idx == 1:
            loss = self.discriminator_step(X)

        return loss

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.generator.parameters())
        d_optimizer = torch.optim.Adam(self.discriminator.parameters())
        return [g_optimizer, d_optimizer], []

    def training_epoch_end(self, training_step_outputs):
        epoch_test_images = self(self.test_noises)
        self.test_progression.append(epoch_test_images)
