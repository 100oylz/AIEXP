from Generator import Generator
from Discrimintor import Discriminator
from config import *
from utils import trainData,testData
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
import torchvision.utils as vutils

def trainGan():
    trainLoader = DataLoader(trainData, batch_size=DEFAULTGANBATCHSIZE, shuffle=True, num_workers=4)
    generator = Generator()
    discriminator = Discriminator()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator.to(device)
    discriminator.to(device)
    lr = 0.0001
    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr)
    maxepoch = 100
    iterator = iter(trainLoader)
    testImgs = list()
    for i in range(4):
        testImgs.append(testData[i][0])
    testImgs = torch.stack(testImgs, 0)
    dLosses = list()
    gLosses = list()
    max_dLoss = float('-inf')
    min_dLoss = float('inf')
    min_gLoss = float('inf')
    best_dLoss_discriminator = None
    best_gLoss_generator = None
    global_step = 0
    latent_size=64
    # 生成器生成图像与原始图像的对比展示 
    noises = torch.randn(DEFAULTGANBATCHSIZE, 3*32*32)
    for epoch in range(1, maxepoch + 1):
        train_progress = tqdm(trainLoader, desc=f'epoch {epoch}: Training', leave=False)
        for input, _ in train_progress:
            real = input.to(device)
            fake = generator(torch.randn(DEFAULTGANBATCHSIZE,3*32*32).to(device)).to(device)  # 生成与真实样本相同数量的假样本
            # print(fake.shape)
            if global_step % 2 == 0:
                realAndfake = torch.cat((real, fake), 0)
                probs = discriminator(realAndfake)
                # print(probs.shape)
                realProb, fakeProb = torch.chunk(probs, 2)
                # print(realProb.shape,fakeProb.shape)
                dLoss = F.binary_cross_entropy(realProb, torch.ones_like(realProb)) + \
                        F.binary_cross_entropy(fakeProb, torch.zeros_like(fakeProb))
                # print(dLoss.shape)
                dLosses.append(dLoss.item())
                
                if dLoss > max_dLoss:
                    max_dLoss = dLoss
                    best_dLoss_discriminator = discriminator.state_dict()
                
                if dLoss < min_dLoss:
                    min_dLoss = dLoss
                
                optimizerD.zero_grad()
                dLoss.backward()
                optimizerD.step()
            else:
                fakeProb = discriminator(fake).reshape(-1,1)
                gLoss = F.binary_cross_entropy(fakeProb, torch.ones_like(fakeProb))
                gLosses.append(gLoss.item())
                
                if gLoss < min_gLoss:
                    min_gLoss = gLoss
                    best_gLoss_generator = generator.state_dict()
                
                optimizerG.zero_grad()
                gLoss.backward()
                optimizerG.step()
            
            global_step += 1
            
            if global_step % 10 == 0 and global_step != 0:
                print(f"Global Step: {global_step}, Generator Loss: {gLoss.item():.4f}, Discriminator Loss: {dLoss.item():.4f}")
                logger.add_scalar('Generator Loss',gLoss.item(),global_step=global_step)
                logger.add_scalar('Discriminator Loss',dLoss.item(),global_step=global_step)

                # 保存最大dLoss对应的鉴别器
                torch.save(best_dLoss_discriminator, 'checkpoint/best_dLoss_discriminator.pt')
                
                # 保存dLoss最小的鉴别器
                discriminator.load_state_dict(best_dLoss_discriminator)
                torch.save(discriminator.state_dict(), 'checkpoint/min_dLoss_discriminator.pt')
                
                # 保存gLoss最小的生成器
                generator.load_state_dict(best_gLoss_generator)
                torch.save(generator.state_dict(), 'checkpoint/min_gLoss_generator.pt')
            
                if global_step%100==0 and global_step != 0:
                                
                    fake_images = generator(noises.to(device))
                    vutils.save_image(fake_images, f'./png/fake_images_{global_step}.png')

        
def save_dict(
        encoder,
        decoder,# PyTorch模型，包括网络结构和参数
        content: float,  # 要比较的内容值
        threshold: float,  # 阈值，与content比较以确定是否保存模型
        reason: str,  # 原因字符串，用于决定是否保存模型
        epoch: int,  # 当前训练的轮次（epoch）
        journal_list: list,  # 存储日志的列表
        encoder_save_path:str,
        decoder_save_path:str
):
    if reason == "Loss":
        if content < threshold:
            threshold = content
            torch.save(encoder.state_dict(), encoder_save_path)
            torch.save(decoder.state_dict(),decoder_save_path)
            journal = f"epoch {epoch} saved! Because of {reason}"
            print(journal)
            journal_list.append(journal)
    else:
        if content > threshold:
            threshold = content
            torch.save(encoder.state_dict(), encoder_save_path)
            torch.save(decoder.state_dict(),decoder_save_path)
            journal = f"epoch {epoch} saved! Because of {reason}"
            print(journal)
            journal_list.append(journal)
    return threshold
    
if __name__=='__main__':
    time = datetime.now()
    time = time.timestamp()
    logger = SummaryWriter(f'./log/{time}')
    print(time)
    trainGan()