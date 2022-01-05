import argparse
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
from model.discriminator import Discriminator
from loss.loss import wasserstein_loss
from model.critic import Critic
from model.generator import Generator
from dataset.dataset import FaceGan
from torch.utils.data import DataLoader
from train.train_critic import train


def parse_args():
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Parse the arguments.')

    parser.add_argument('--exp_id', type=str, default='exp_0')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--noise_dim', type=int, default=100)
    parser.add_argument('--wasserstein', type=bool, default=True)
    parser.add_argument('--gradient_penalty_lambda', type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    img_transform = Compose([
        Resize(128),
        CenterCrop(128),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    data_dir = 'img_align_celeba'

    face_dataset = FaceGan(data_dir, img_transform)
    train_loader = DataLoader(face_dataset, batch_size=args.batch_size, shuffle=True)

    fixed_noise = torch.randn(args.batch_size, args.noise_dim, 1, 1, device=device)
    generator = Generator(noise_dim=args.noise_dim, output_dim=3).to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    if args.wasserstein:
        criterion = wasserstein_loss
        critic = Critic().to(device)
        optimizer_c = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.999))
        generator_losses, discriminator_losses, image_list = train(critic,
                                                                   generator,
                                                                   train_loader,
                                                                   criterion,
                                                                   optimizer_g,
                                                                   optimizer_c,
                                                                   device,
                                                                   fixed_noise,
                                                                   args)
    else:
        criterion = torch.nn.BCELoss()
        discriminator = Discriminator().to(device)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.1, 0.9))
        generator_losses, discriminator_losses, image_list = train(discriminator,
                                                                   generator,
                                                                   train_loader,
                                                                   criterion,
                                                                   optimizer_g,
                                                                   optimizer_d,
                                                                   device,
                                                                   fixed_noise,
                                                                   args)

    print("Finish training.")
