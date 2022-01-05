import torch
import torchvision.utils as vutils
import neptune.new as neptune
from neptune.new.types import File
import numpy as np

run = neptune.init(
    project="quan-ml/DCGAN",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NDgzYmY3Zi1kMjZjLTRkNjUtYWY2Ny0wODAwZDBjNjkwNGUifQ==",
)  # your credentials

real_label = 1
fake_label = 0


def train(discriminator,
          generator,
          train_loader,
          criterion,
          optimizer_g,
          optimizer_d,
          device,
          fixed_noise,
          args):
    generator_losses = []
    discriminator_losses = []
    image_list = []
    iters = 0

    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader, 0):
            # train discriminator
            optimizer_d.zero_grad()
            output = discriminator(data.to(device)).view(-1)
            ground_truth = torch.full((output.size(0),), real_label, device=device, dtype=torch.float)
            discriminator_error_real = criterion(output, ground_truth)
            discriminator_error_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(data.size(0), args.noise_dim, 1, 1, device=device)
            fake = generator(noise)
            ground_truth = torch.full((fake.size(0),), fake_label, device=device, dtype=torch.float)

            output = discriminator(fake.detach()).view(-1)
            discriminator_error_fake = criterion(output, ground_truth)
            discriminator_error_fake.backward()
            D_G_z1 = output.mean().item()

            discriminator_error = discriminator_error_real + discriminator_error_fake
            optimizer_d.step()

            # train generator
            optimizer_g.zero_grad()
            ground_truth = torch.full((fake.size(0),), real_label, device=device, dtype=torch.float)
            output = discriminator(fake).view(-1)
            generator_error = criterion(output, ground_truth)
            generator_error.backward()
            D_G_z2 = output.mean().item()
            optimizer_g.step()

            generator_losses.append(generator_error.item())
            discriminator_losses.append(discriminator_error.item())

            run["generator_loss"].log(generator_error.item())
            run["discriminator_loss"].log(discriminator_error.item())

            if i % 100 == 0:
                print('[%d/%d] [%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epochs, i, len(train_loader),
                         discriminator_error.item(), generator_error.item(), D_x, D_G_z1, D_G_z2))
            if iters % 100 == 0 or (epoch == args.epochs - 1 and i == len(train_loader) - 1):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                image_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                run["images"].log(
                    File.as_image(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), [1, 2, 0])))

            iters += 1
        torch.save(generator.state_dict(), './checkpoints/generator_' + str(epoch) + '.pth')
        torch.save(discriminator.state_dict(), './checkpoints/discriminator_' + str(epoch) + '.pth')
        try:
            run["generator"].log(File("./checkpoints/generator_" + str(epoch) + ".pth"))
            run["discriminator"].log(File("./checkpoints/discriminator_" + str(epoch) + ".pth"))
        except Exception as e:
            print(e)
            print("Could not log model")

    return generator_losses, discriminator_losses, image_list
