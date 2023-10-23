import torch.nn as nn
import torch.nn.init as init
import torch
import torch.optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import save_image

device='cuda' if torch.cuda.is_available() else 'cpu'
latent_size = 64
n_channel = 3
n_g_feature = 64
gnet = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 4 * n_g_feature, kernel_size=4, bias=False),
    nn.BatchNorm2d(4 * n_g_feature),
    nn.ReLU(),

    nn.ConvTranspose2d(4 * n_g_feature, 2 * n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * n_g_feature),
    nn.ReLU(),

    nn.ConvTranspose2d(2 * n_g_feature, n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(n_g_feature),
    nn.ReLU(),

    nn.ConvTranspose2d(n_g_feature, n_channel, kernel_size=4, stride=2, padding=1),
    nn.Sigmoid()
)


n_d_feature = 64
dnet = nn.Sequential(
    nn.Conv2d(n_channel, n_d_feature, kernel_size=4, stride=2, padding=1),
    nn.LeakyReLU(0.2),

    nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(2 * n_d_feature),
    nn.LeakyReLU(0.2),

    nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(4 * n_d_feature),
    nn.LeakyReLU(0.2),

    nn.Conv2d(4 * n_d_feature, 1, kernel_size=4)
)


def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0)

gnet.apply(weights_init)
dnet.apply(weights_init)

from torchsummary import summary
gnet.to(device='cuda')
dnet.to(device='cuda')
summary(gnet,(64,1,1))
summary(dnet,(3,32,32))
print(gnet)
print(dnet)

# dataset = CIFAR10(root=r"C:\Users\OYLZ\Desktop\AutoEncoder\CIFAR10", download=False,train=True, transform=transforms.ToTensor())
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# dnet_state_dict=torch.load('dnet.pt')
# gnet_state_dict=torch.load('gnet.pt')
# dnet.load_state_dict(dnet_state_dict.state_dict())
# gnet.load_state_dict(gnet_state_dict.state_dict())

# dnet=dnet.to(device)
# gnet=gnet.to(device)

# criterion = nn.BCEWithLogitsLoss()
# goptimizer = torch.optim.Adam(gnet.parameters(), lr=0.0001, betas=(0.5, 0.999))
# doptimizer = torch.optim.Adam(dnet.parameters(), lr=0.0001, betas=(0.5, 0.999))


# batch_size = 64
# fixed_noises = torch.randn(batch_size, latent_size, 1, 1)

# epoch_num = 300
# for epoch in range(epoch_num):
#     for batch_idx, data in enumerate(dataloader):
#         real_images, _ = data
#         real_images=real_images.to(device)
#         batch_size = real_images.size(0)

#         labels = torch.ones(batch_size).to(device)
#         preds = dnet(real_images)
#         outputs = preds.reshape(-1)
#         dloss_real = criterion(outputs, labels)
#         dmean_real = outputs.sigmoid().mean()

#         noises = torch.randn(batch_size, latent_size, 1, 1).to(device)
#         fake_images = gnet(noises)
#         labels = torch.zeros(batch_size).to(device)
#         fake = fake_images.detach()

#         preds = dnet(fake)
#         outputs = preds.view(-1)
#         dloss_fake = criterion(outputs, labels)
#         dmean_fake = outputs.sigmoid().mean()

#         dloss = dloss_real + dloss_fake
#         dnet.zero_grad()
#         dloss.backward()
#         doptimizer.step()


#         labels = torch.ones(batch_size).to(device)
#         preds = dnet(fake_images)
#         outputs = preds.view(-1)
#         gloss = criterion(outputs, labels)
#         gmean_fake = outputs.sigmoid().mean()
#         gnet.zero_grad()
#         gloss.backward()
#         goptimizer.step()

#         if batch_idx % 100 == 0:
#             fake = gnet(fixed_noises.to(device))
#             save_image(fake, f'GAN_saved04/images_epoch{epoch:02d}_batch{batch_idx:03d}.png')

#             print(f'Epoch index: {epoch}, {epoch_num} epoches in total.')
#             print(f'Batch index: {batch_idx}, the batch size is {batch_size}.')
#             print(f'Discriminator loss is: {dloss}, generator loss is: {gloss}', '\n',
#                   f'Discriminator tells real images real ability: {dmean_real}', '\n',
#                   f'Discriminator tells fake images real ability: {dmean_fake:g}/{gmean_fake:g}')


# gnet_save_path = 'gnet.pt'
# torch.save(gnet, gnet_save_path)
# # gnet = torch.load(gnet_save_path)
# # gnet.eval()

# dnet_save_path = 'dnet.pt'
# torch.save(dnet, dnet_save_path)
# # dnet = torch.load(dnet_save_path)
# # dnet.eval()

# for i in range(100):
#     noises = torch.randn(batch_size, latent_size, 1, 1).to(device)
#     fake_images = gnet(noises)
#     save_image(fake, f'./test_GAN/{i}.png')

# print(gnet, dnet)

