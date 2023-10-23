import torch.nn as nn
from LoadData import MAX_LENGTH, device
from WordVec import WordVec
import torch


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden):
        input_vec = self.embedding(input)
        input_vec = input_vec.view(1, 1, -1)
        output, hidden_t = self.rnn(input_vec, hidden)
        return output, hidden_t

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class GRUDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(GRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.activeFunc = nn.PReLU()

    def forward(self, input, hidden):
        hidden_1 = self.embedding(input)
        hidden_1 = hidden_1.view(1, 1, -1)
        hidden_2 = self.activeFunc(hidden_1)
        output, hidden_state = self.rnn(hidden_2, hidden)
        output = self.softmax(self.out(output.squeeze(0)))

        return output, hidden_state

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def GRUtrain(input_tensor, target_tensor, encoder: GRUEncoder, decoder: GRUDecoder, encoder_optimizer,
             decoder_optimizer, criterion,
             max_length=MAX_LENGTH):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    encoder_hidden = encoder.initHidden()
    target_tensor.to(device)
    input_tensor.to(device)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_outputs, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)


    # 完成训练部分代码
    #######Your Code#######

    # 调用encoder类完成整个编码计算流程

    #######End#######

    decoder_input = torch.tensor([[WordVec.SOS_token]], device=device)
    loss=torch.zeros(1).to(device)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        decoder_output.to(device)
        target_tensor.to(device)
        # 完成训练部分代码
        #######Your Code#######

        # 调用decoder类完成整个解码计算流程

        # 计算解码器每个time step prediction和target的loss，进行求和得到最终loss

        #######End#######

        loss += criterion(decoder_output, target_tensor[di])

        decoder_input = target_tensor[di]
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
