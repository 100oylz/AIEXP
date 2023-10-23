import tensorboardX
from WordVec import WordVec
from LoadData import *
from EncoderAndDecoder import GRUEncoder, GRUDecoder, GRUtrain
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import json

logger = tensorboardX.SummaryWriter(log_dir="data/log")
save_path_format = "./checkpoint/GRU_Seq2Seq_{}_best.pt"
journal_path="./journal/GRU_Seq2Seq_best.json"
def save_dict(
        model: GRUEncoder | GRUDecoder,
        content: float,  # 要比较的内容值
        threshold: float,  # 阈值，与content比较以确定是否保存模型
        reason: str,  # 原因字符串，用于决定是否保存模型
        epoch: int,  # 当前训练的轮次（epoch）
        journal_list: list,  # 存储日志的列表
        save_path: str
):
    if content < threshold:
        threshold = content
        torch.save(model.state_dict(), save_path)
        journal = f"epoch {epoch} saved! Because of {reason}"
        # print(journal)
        journal_list.append(journal)

    return threshold

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    train_loss_history = []
    min_train_loss = torch.inf
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    journal_list = []
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]

    criterion = nn.NLLLoss()
    train_progress = tqdm(range(1, n_iters + 1), desc='Training', leave=False)
    for iter in train_progress:
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = GRUtrain(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)

        train_loss = loss

        save_dict(model=encoder, content=train_loss, threshold=min_train_loss, reason="Loss", journal_list=journal_list,
                  save_path=save_path_format.format(encoder.__class__.__name__),epoch=iter)

        save_dict(model=decoder, content=train_loss, threshold=min_train_loss, reason="Loss", journal_list=journal_list,
                  save_path=save_path_format.format(decoder.__class__.__name__),epoch=iter)
        echo_string=f'Epoch {iter}:-> Loss:{train_loss}'
        journal_list.append(echo_string)
        logger.add_scalar("Plot Loss", train_loss, global_step=iter)
        train_loss_history.append(train_loss)
    train_progress.close()
    journal=f'The Best Loss:{min_train_loss}'
    journal_list.append(journal)
    with open(journal_path, "w", encoding="utf8") as f:
        json.dump(journal_list ,f, ensure_ascii=False)

    print(f"Saved Journal in {journal_path}")

if __name__ == '__main__':
    # input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    hidden_size = 256
    encoder1 = GRUEncoder(input_lang.num_words, hidden_size).to(device)
    decoder1 = GRUDecoder(hidden_size, output_lang.num_words).to(device)
    encoder1.to(device)
    decoder1.to(device)
    trainIters(encoder1, decoder1, 55000)




