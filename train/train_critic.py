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
fake_label = -1


def calc_gradient_penalty(critic, real_points, fake_points, device):

    alpha = torch.rand(real_points.size(0), 1, 1, 1, device=device)
    interpolated_points = (alpha * real_points.to(device) + (1 - alpha) * fake_points.to(device)).requires_grad_(True)
    interpolated_output = critic(interpolated_points)

    gradient = torch.autograd.grad(interpolated_output,
                                   interpolated_points,
                                   grad_outputs=torch.ones(interpolated_output.size()).to(device),
                                   create_graph=True)[0]
    gradient = gradient.view(gradient.size(0), -1)
    gradient_l2norm = gradient.norm(2, dim=1)
    gradient_penalty = ((gradient_l2norm - 1) ** 2).mean()
    return gradient_penalty


def train(critic,
          generator,
          train_loader,
          criterion,
          optimizer_g,
          optimizer_c,
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
            # create random noise
            noise = torch.randn(data.size(0), args.noise_dim, 1, 1, device=device)
            fake = generator(noise)
            for _ in range(5):
                optimizer_c.zero_grad()

                real_pred = critic(data.to(device)).view(-1)
                D_x = real_pred.mean().item()

                fake_pred = critic(fake.detach()).view(-1)
                D_G_z1 = fake_pred.mean().item()

                gradient_penalty = calc_gradient_penalty(critic, data, fake.detach(), device)

                discriminator_error = fake_pred.mean() - real_pred.mean() + gradient_penalty * args.gradient_penalty_lambda
                discriminator_error.backward()
                optimizer_c.step()

            # train generator
            optimizer_g.zero_grad()
            gen_output = - critic(fake).view(-1).mean()
            gen_output.backward()
            D_G_z2 = gen_output.item()
            optimizer_g.step()

            generator_losses.append(gen_output.item())
            discriminator_losses.append(discriminator_error.item())

            run["generator_fake_loss"].log(gen_output.item())
            run["critic_real_loss"].log(real_pred.mean().item())
            run["critic_fake_loss"].log(fake_pred.mean().item())
            run["gradient_penalty"].log(gradient_penalty.item())
            run["discriminator_error"].log(discriminator_error.item())

            if i % 100 == 0:
                print('[%d/%d] [%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epochs, i, len(train_loader),
                         discriminator_error.item(), gen_output.item(), D_x, D_G_z1, D_G_z2))
            if iters % 100 == 0 or (epoch == args.epochs - 1 and i == len(train_loader) - 1):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                image_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                run["images"].log(
                    File.as_image(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), [1, 2, 0])))

            iters += 1
        torch.save(generator.state_dict(), './checkpoints/c_generator_' + str(epoch) + '.pth')
        torch.save(critic.state_dict(), './checkpoints/c_discriminator_' + str(epoch) + '.pth')
        try:
            run["generator"].log(File("./checkpoints/generator_" + str(epoch) + ".pth"))
            run["discriminator"].log(File("./checkpoints/discriminator_" + str(epoch) + ".pth"))
        except Exception as e:
            print(e)
            print("Could not log model")

    return generator_losses, discriminator_losses, image_list
