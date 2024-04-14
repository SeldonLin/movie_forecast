import os
import torch
import argparse
import torch.nn as nn
import pandas as pd
from pathlib import Path
from transformers import pipeline, BertModel
from tqdm import tqdm
import ast
import os

print(os.path.abspath(__file__))
class TextEmbedder(nn.Module):
    def __init__(self, embedding_dim, gpu_num):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.word_embedder = pipeline('feature-extraction',model='/root/autodl-tmp/GTM-Transformer-main/bert-base-uncased')
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
        word_embeddings = torch.stack(word_embeddings)

        # Embed to our embedding space
        word_embeddings = self.dropout(self.fc(word_embeddings))

        return word_embeddings

    
def get_comment_text_encoding(data,comments, comment_embedding):
    # get comment 
    comments_text_encoding = torch.zeros(len(data), 100, 32)

    # Read the descriptions and the images
    for (idx, row) in tqdm(data.iterrows(), total=len(data), ascii=True):
        name = row['name']

        # get comment
        comment = comments[comments['name'] == name].head(100)
        comment_list = list(comment['recommend'])
        comment_text_encoding = comment_embedding(comment_list)
        comments_text_encoding[idx] = comment_text_encoding
    return comments_text_encoding


def get_introduction_encoding(data, introduction_embedding):
    movies_description = []
    for (idx, row) in tqdm(data.iterrows(), total=len(data), ascii=True):
        id, name, director, actor, introduction = row['id'], row['name'], row['director'], row['actor'], row['introduction']

        # get movies description
        director = ' '.join(ast.literal_eval(director))
        actor = ' '.join(ast.literal_eval(actor))
        # description_tmp = director + actor + introduction
        description_tmp = 'director:' + director +'actor:' + actor
        movies_description.append(description_tmp)
    introduction_encoding = introduction_embedding(movies_description)
    return introduction_encoding

def run(args):

    comment_embedding = TextEmbedder(args.embedding_dim, args.gpu_num)
    introduction_embedding = TextEmbedder(args.embedding_dim, args.gpu_num)
    train_df = pd.read_excel('dataset/train.xlsx')
    print('get train_df\n')
    test_df = pd.read_excel('dataset/test.xlsx')
    print('get test_df\n')
    comments = pd.read_excel('dataset/comment.xlsx')
    print('get comments\n')


    train_comment_text_encoding = get_comment_text_encoding(train_df, comments, comment_embedding)
    torch.save(train_comment_text_encoding,Path(args.data_folder + 'train_comment_text_encoding.pth'))
    print('get train_comment_text_encoding\n')

    test_comment_text_encoding = get_comment_text_encoding(test_df, comments, comment_embedding)
    torch.save(test_comment_text_encoding,Path(args.data_folder + 'test_comment_text_encoding.pth'))
    print('get test_comment_text_encoding\n')

    train_introduction_emcoding = get_introduction_encoding(train_df,introduction_embedding)
    torch.save(train_introduction_emcoding,Path(args.data_folder + 'train_introduction_emcoding.pth'))
    print('get train_introduction_emcoding\n')

    test_introduction_emcoding = get_introduction_encoding(test_df,introduction_embedding)
    torch.save(test_introduction_emcoding,Path(args.data_folder + 'test_introduction_emcoding.pth'))
    print('get test_introduction_emcoding\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Box office forecasting')

    # General arguments
    parser.add_argument('--model_name', type=str, default='multimodal')
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--ckpt_path', type=str, default='ckpt')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpu_num', type=int, default=0)

    # Model specific arguments
    parser.add_argument('--comment_len', type=int, default=100)
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=20)
    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=int, default=0)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=1)


    args = parser.parse_args()
    run(args)
