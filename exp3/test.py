import torch.cuda

from EncoderAndDecoder import GRUEncoder, GRUDecoder
from LoadData import *


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = 256
    encoder = GRUEncoder(input_lang.num_words, hidden_size)
    decoder = GRUDecoder(hidden_size, output_lang.num_words)
    encoderstateDict = torch.load('checkpoint/GRU_Seq2Seq_GRUEncoder_best.pt')
    decoderstateDict = torch.load('checkpoint/GRU_Seq2Seq_GRUDecoder_best.pt')
    encoder.load_state_dict(encoderstateDict)
    decoder.load_state_dict(decoderstateDict)
    encoder.to(device)
    decoder.to(device)

    evaluateRandomly(encoder, decoder)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    SOS_token=WordVec.SOS_token
    EOS_token=WordVec.EOS_token
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2Count[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

def evaluateRandomly(encoder, decoder, n=100):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

if __name__ == '__main__':
    test()
