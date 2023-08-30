
import torch.nn as nn
import pytorch_lightning as pl
import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cuda:0"

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
