from Encoder import Encoder
from Decoder import Decoder
import torch.optim as optim
import torch.nn as nn
import torch
from tensorboardX import SummaryWriter
from config import *
from utils import loadData
from datetime import datetime
from torch.utils.data import dataloader
from tqdm import tqdm
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def train():
    encoder=Encoder()
    decoder=Decoder()
    max_loss=torch.inf
    optimizer=optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=DEFAULTLEARNINGRATE,eps=1e-18,weight_decay=DEFAULTWEIGHTDEACY,betas=[0.99,0.9])
    lossfn=nn.functional.binary_cross_entropy
    trainData,_=loadData()
    trainLoader=dataloader.DataLoader(trainData,batch_size=32,shuffle=True,num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder.to(device)
    decoder.to(device)
    loss_history=[]
    journal_history=[]
    maxepoch=100
    total_loss=0
    global_iter=0
    test_image=0
    for data,label in trainLoader:
        test_image=data[0,:]
        test_image=test_image.view(1,test_image.size(0),test_image.size(1),test_image.size(2))
        # print(test_image.shape)
        break
    logger.add_images('Source Image',test_image)
    for epoch in range(maxepoch):
        loss=0
        total_loss=0
        total_train_samples=0
        train_progress = tqdm(trainLoader, desc=f'epoch {epoch}: Training', leave=False)
        for input,_ in train_progress:
            input=input.to(device)
            encoded=encoder(input)
            decoded=decoder(encoded)
            loss=lossfn(decoded,input)
            optimizer.zero_grad(None)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            total_train_samples+=input.size(0)
            if(global_iter%100==0):
                logger.add_scalar('loss',loss.item(),global_step=global_iter)
            global_iter+=1
        encoder.eval()
        decoder.eval()
        test_image=test_image.to(device)
        encoded=encoder(test_image)
        decoded=decoder(encoded)
        logger.add_images('Generated Image',decoded,global_step=epoch+1)
        encoder.train()
        decoder.train()
        train_loss = 32*total_loss / total_train_samples
        logger.add_scalar('train_loss',train_loss,global_step=epoch+1)
        encoder_save_path='./checkpoint/encoder_loss_best.pt'
        decoder_save_path='./checkpoint/decoder_loss_best.pt'
        max_loss = save_dict(
            encoder,
            decoder,
            train_loss,
            max_loss,
            "Loss",
            epoch,
            journal_history,
            encoder_save_path=encoder_save_path,
            decoder_save_path=decoder_save_path
        )
        loss_history.append(train_loss)

        echo_string = f"epoch:{epoch} -> loss:{train_loss}"
        print(echo_string)
        loss_history.append(train_loss)
        journal_history.append(echo_string)
    journal = f"The Best Loss:{max_loss}!\n"
    journal_history.append(journal)
        
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
    train()
        