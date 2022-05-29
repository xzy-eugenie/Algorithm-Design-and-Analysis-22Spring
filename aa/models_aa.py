import torch
import torch.nn as nn
import torchvision
import torch.optim
import torch.utils.data
import torchvision.models as models
from torch.autograd import Variable
from torch.nn import init
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, hidden_size, embed_size):
        super(Encoder,self).__init__()
        #resnet = torchvision.models.resnet101(pretrained = True)
        resnet = torchvision.models.resnet101(pretrained = True)
        all_modules = list(resnet.children())
        #Remove the last FC layer used for classification and the average pooling layer
        modules = all_modules[:-2]
        #Initialize the modified resnet as the class variable
        self.resnet = nn.Sequential(*modules) 
        self.avgpool = nn.AvgPool2d(7)
        self.fine_tune()    # To fine-tune the CNN, self.fine_tune(status = True)
    
    def forward(self, images):
        """
        The forward propagation function
        input: resized image of shape (batch_size,3,224,224)
        """
        #Run the image through the ResNet
        encoded_image = self.resnet(images)         # (batch_size,2048,7,7)
        batch_size = encoded_image.shape[0]
        features = encoded_image.shape[1]
        num_pixels = encoded_image.shape[2] * encoded_image.shape[3]
        # Get the global features of the image
        global_features = self.avgpool(encoded_image).view(batch_size, -1)   # (batch_size, 2048)
        enc_image = encoded_image.permute(0, 2, 3, 1)  #  (batch_size,7,7,2048)
        enc_image = enc_image.view(batch_size,num_pixels,features)          # (batch_size,num_pixels,2048)
        return enc_image, global_features
    
    def fine_tune(self, status = True):
        
        if not status:
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            for module in list(self.resnet.children())[7:]:    #1 layer only. len(list(resnet.children())) = 8
                for param in module.parameters():
                    param.requires_grad = True 


class AdaptiveLSTMCell(nn.Module):  # 一个LSTM代码块的class，包含哨兵st的计算
    def __init__(self, input_size, hidden_size):
        super(AdaptiveLSTMCell, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.x_gate = nn.Linear(input_size, hidden_size)
        self.h_gate = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, inp, states):
        h_old, c_old = states
        ht, ct = self.lstm_cell(inp, (h_old, c_old))
        sen_gate = torch.sigmoid(self.x_gate(inp) + self.h_gate(h_old))
        st =  sen_gate * torch.tanh(ct)
        return ht, ct, st


class AdaptiveAttention(nn.Module):
    def __init__(self, hidden_size, att_dim):
        super(AdaptiveAttention,self).__init__()
        self.sen_affine = nn.Linear(hidden_size, hidden_size)  
        self.sen_att = nn.Linear(hidden_size, att_dim)
        self.h_affine = nn.Linear(hidden_size, hidden_size)   
        self.h_att = nn.Linear(hidden_size, att_dim)
        self.v_att = nn.Linear(hidden_size, att_dim)
        self.alphas = nn.Linear(att_dim, 1)
        self.context_hidden = nn.Linear(hidden_size, hidden_size)

    def forward(self, spatial_image, decoder_out, st):
        """
        spatial_image: the spatial image of size (batch_size,num_pixels,hidden_size)
        decoder_out: the decoder hidden state of shape (batch_size, hidden_size)
        st: visual sentinal returned by the Sentinal class, of shape: (batch_size, hidden_size)
        """
        num_pixels = spatial_image.shape[1]
        visual_attn = self.v_att(spatial_image)           # (batch_size,num_pixels,att_dim)
        sentinel_affine = F.relu(self.sen_affine(st))     # (batch_size,hidden_size)
        sentinel_attn = self.sen_att(sentinel_affine)     # (batch_size,att_dim)

        hidden_affine = torch.tanh(self.h_affine(decoder_out))    # (batch_size,hidden_size)  # LSTM输出的隐藏层ht
        hidden_attn = self.h_att(hidden_affine)               # (batch_size,att_dim)  # ht也过att  过att并没有attention因素加入，只是为之后引入attention因素所加的一层linear

        hidden_resized = hidden_attn.unsqueeze(1).expand(hidden_attn.size(0), num_pixels + 1, hidden_attn.size(1))  # (batch_size, num_pixels+1, hidden_size)  # 第一维是batch_size, 第三维是att_dim

        concat_features = torch.cat([spatial_image, sentinel_affine.unsqueeze(1)], dim = 1)   # (batch_size, num_pixels+1, hidden_size)  # image和st没过att前的拼接feature
        attended_features = torch.cat([visual_attn, sentinel_attn.unsqueeze(1)], dim = 1)     # (batch_size, num_pixels+1, att_dim)  # image和st过完att之后拼接在一起

        attention = torch.tanh(attended_features + hidden_resized)    # (batch_size, num_pixels+1, att_dim)  # 过完att并拼接的feature和过完att的ht直接相加，过tanh
        
        alpha = self.alphas(attention).squeeze(2)                   # (batch_size, num_pixels+1)  # 这一步得到的是z_t
        att_weights = F.softmax(alpha, dim=1)                              # (batch_size, num_pixels+1)  # 这一步得到的是alpha_t

        context = (concat_features * att_weights.unsqueeze(2)).sum(dim=1)       # (batch_size, hidden_size)  # image和st没过att前的拼接feature，用att得到的alpha系数加权，是原论文的ct_hat
        beta_value = att_weights[:,-1].unsqueeze(1)                       # (batch_size, 1)  # alpha是att系数，其最后一维作为gate value: beta

        out_l = torch.tanh(self.context_hidden(context + hidden_affine))  # 这是什么输出？我们接着看！

        return out_l, att_weights, beta_value


class DecoderWithAttention(nn.Module):
    
    def __init__(self,hidden_size, vocab_size, att_dim, embed_size, encoded_dim, att_mode=None):
        super(DecoderWithAttention,self).__init__()
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.encoded_to_hidden = nn.Linear(encoded_dim, hidden_size)  # encoded_dim是encoder输出的feature_num
        self.global_features = nn.Linear(encoded_dim, embed_size)
        self.LSTM = AdaptiveLSTMCell(embed_size * 2,hidden_size)  # embed_size * 2是lstmCell中的input_size
        self.adaptive_attention = AdaptiveAttention(hidden_size, att_dim)
        # input to the LSTMCell should be of shape (batch, input_size). Remember we are concatenating the word with
        # the global image features, therefore out input features should be embed_size * 2
        self.embedding = nn.Embedding(vocab_size, embed_size)  
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()
        self.att_mode = att_mode
        
    def init_weights(self):
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
    
    def init_hidden_state(self, enc_image):
        h = torch.zeros(enc_image.shape[0], 512).to(device)  # (batch_size, hidden_size=512)
        c = torch.zeros(enc_image.shape[0], 512).to(device)
        return h, c

    def forward(self, enc_image, global_features, encoded_captions, caption_lengths):
        
        """
        enc_image: the encoded images from the encoder, of shape (batch_size, 49, 2048) 2048: feature_num 49: pixels_num
        global_features: the global image features returned by the Encoder, of shape: (batch_size, 2048)
        encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        """
        global_image = F.relu(self.global_features(global_features))      # (batch_size,embed_size)  # 第三维由feature_num变成embed_size
        spatial_image = F.relu(self.encoded_to_hidden(enc_image))  # (batch_size, num_pixels, hidden_size)  # 第三维由feature_num变成 hidden_size
        batch_size = enc_image.shape[0]
        num_pixels = enc_image.shape[1]
        # Sort input data by decreasing lengths
        # caption_lengths will contain the sorted lengths, and sort_ind contains the sorted elements indices 
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        # sort_ind contains elements of the batch index of the tensor encoder_out. For example, if sort_ind is [3,2,0],
        # then that means the descending order starts with batch number 3,then batch number 2, and finally batch number 0. 
        global_image = global_image[sort_ind]             # (batch_size, embed_size) with sorted batches
        encoded_captions = encoded_captions[sort_ind]     # (batch_size, max_caption_length) with sorted batches 
        spatial_image = spatial_image[sort_ind]                   # (batch_size, num_pixels, 2048)

        # Embedding. Each batch contains a caption. All batches have the same number of rows (words), since we previously
        # padded the ones shorter than max_caption_length, as well as the same number of columns (embed_dim)
        embeddings = self.embedding(encoded_captions)     # (batch_size, max_caption_length, embed_dim)

        # Initialize the LSTM state
        h,c = self.init_hidden_state(enc_image)          # (batch_size, hidden_size)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        decode_lengths = (caption_lengths - 1).tolist()  # 把最后一个token <end> 占用的长度统一剪掉

        # Create tensors to hold word predicion scores,alphas and betas
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)  # 第三维的值表示选哪个词
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels+1).to(device)
        betas = torch.zeros(batch_size, max(decode_lengths),1).to(device) 
        
        # Concatenate the embeddings and global image features for input to LSTM 
        global_image = global_image.unsqueeze(1).expand_as(embeddings)
        inputs = torch.cat((embeddings,global_image), dim = 2)    # (batch_size, max_caption_length, embed_dim * 2)

        #Start decoding 产生一个句子
        for timestep in range(max(decode_lengths)):            
            batch_size_t = sum([l > timestep for l in decode_lengths])  # 句子长度比当前时间步长，才有意义继续产生单词
            # Create a Packed Padded Sequence manually, to process only the effective batch size N_t at that timestep.
            # We cannot use the pack_padded_seq provided by torch.util because we are using an LSTMCell, and not an LSTM        
            current_input = inputs[:batch_size_t, timestep, :]             # (batch_size_t, embed_dim * 2)  # 句子从长到短排序
            h, c, st = self.LSTM(current_input, (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, hidden_size)
            # Run the adaptive attention model
            out_l, alpha_t, beta_t = self.adaptive_attention(spatial_image[:batch_size_t],h,st)
            # Compute the probability over the vocabulary
            pt = self.fc(self.dropout(out_l))                  # (batch_size, vocab_size) fc将hidden_size转成vocab_size
            predictions[:batch_size_t, timestep, :] = pt
            alphas[:batch_size_t, timestep, :] = alpha_t
            betas[:batch_size_t, timestep, :] = beta_t
        return predictions, alphas, betas, encoded_captions, decode_lengths, sort_ind  
