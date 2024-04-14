import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline, BertModel
from torchvision import models


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TimeDistributed(nn.Module):
    # Takes any module and stacks the time dimension with the batch dimenison of inputs before applying the module
    # Insipired from https://keras.io/api/layers/recurrent_layers/time_distributed/
    # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module  # Can be any layer we wish to apply like Linear, Conv etc
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class FusionNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, use_img, use_text, dropout=0.2):
        super(FusionNetwork, self).__init__()

        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_linear = nn.Linear(2048, embedding_dim)
        self.use_img = use_img
        self.use_text = use_text
        input_dim = embedding_dim + (embedding_dim * use_img) + (embedding_dim * use_text)
        self.feature_fusion = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim)
        )

    def forward(self, introduction_encoding, img_encoding, dummy_encoding):
        # Fuse static features together
        pooled_img = self.img_pool(img_encoding)
        condensed_img = self.img_linear(pooled_img.flatten(1)) # condensed_img (706,32)

        # Build input
        decoder_inputs = []
        if self.use_text == 1:
            decoder_inputs.append(introduction_encoding)
        if self.use_img == 1:
            decoder_inputs.append(condensed_img)
        decoder_inputs.append(dummy_encoding)
        concat_features = torch.cat(decoder_inputs, dim=1)
        # concat_features (706, 96)

        final = self.feature_fusion(concat_features) # (706, 64)
        # final = self.feature_fusion(dummy_encoding)

        return final


class CommentEmbedder(nn.Module):
    def __init__(self, output_dim, embedding_dim,hidden_dim, comment_len):
        super().__init__()
        self.output_dim = output_dim
        self.input_linear = TimeDistributed(nn.Linear(embedding_dim, hidden_dim))
        self.pos_embedding = PositionalEncoding(hidden_dim, max_len=comment_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, comments_text_encoding):
        comments_encoding = self.input_linear(comments_text_encoding) # (706, 100, 64)
        comments_encoding = self.pos_embedding(comments_encoding.permute(1, 0, 2)) # (100, 706, 64)
        comments_encoding = self.encoder(comments_encoding)
        # return (100, 706, 64)
        return comments_encoding

class TextEmbedder(nn.Module):
    def __init__(self, embedding_dim, gpu_num):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.word_embedder = pipeline('feature-extraction',model='../bert-base-uncased',device=0)
        self.fc = nn.Linear(768, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.gpu_num = gpu_num

    def forward(self, text):
        textual_description = text

        # Use BERT to extract features
        word_embeddings = self.word_embedder(textual_description)

        # BERT gives us embeddings for [CLS] ..  [EOS], which is why we only average the embeddings in the range [1:-1]
        # We're not fine tuning BERT and we don't want the noise coming from [CLS] or [EOS]
        word_embeddings = [torch.FloatTensor(x[0][1:-1]).mean(axis=0) for x in word_embeddings]
        word_embeddings = torch.stack(word_embeddings).to('cuda:' + str(self.gpu_num))

        # Embed to our embedding space
        word_embeddings = self.dropout(self.fc(word_embeddings))

        return word_embeddings

class ImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        # Img feature extraction
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Fine tune resnet
        # for c in list(self.resnet.children())[6:]:
        #     for p in c.parameters():
        #         p.requires_grad = True

    def forward(self, images):
        img_embeddings = self.resnet(images)
        size = img_embeddings.size()
        out = img_embeddings.view(*size[:2], -1)

        # batch_size, 2048, image_size[0]/32, image_size[1]/32
        # return (data_len, batch_size, image_size[0]/32, image_size[1]/32)
        # (706, 2048, 8, 8)
        return out.view(*size).contiguous()


class DummyEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.dummy_fusion = nn.Linear(embedding_dim * 4, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, temporal_features):
        # Temporal dummy variables (day, week, month, year)
        d, w, m, y = temporal_features[:, 0].unsqueeze(1), temporal_features[:, 1].unsqueeze(1), \
            temporal_features[:, 2].unsqueeze(1), temporal_features[:, 3].unsqueeze(1)
        d_emb, w_emb, m_emb, y_emb = self.day_embedding(d), self.week_embedding(w), self.month_embedding(
            m), self.year_embedding(y)
        temporal_embeddings = self.dummy_fusion(torch.cat([d_emb, w_emb, m_emb, y_emb], dim=1))
        temporal_embeddings = self.dropout(temporal_embeddings)

        return temporal_embeddings


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None, tgt_key_padding_mask = None,
            memory_key_padding_mask = None):

        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_weights