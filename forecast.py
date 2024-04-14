import argparse
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from model import multimodal
from encoder_decoder import TextEmbedder,TransformerDecoderLayer,ImageEmbedder,TimeDistributed,PositionalEncoding,DummyEmbedder,FusionNetwork
from data_process import DatasetPrepare
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from pathlib import Path


def cal_error_metrics(gt, forecasts):
    # Absolute errors
    mae = mean_absolute_error(gt, forecasts)
    wape = 100 * np.sum(np.sum(np.abs(gt - forecasts), axis=-1)) / np.sum(gt)

    return round(mae, 3), round(wape, 3)


def print_error_metrics(y_test, y_hat):
    mae, wape = cal_error_metrics(y_test, y_hat)
    print(mae, wape)


def run(args):
    print(args)

    # Set up CUDA
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

    # Seeds for reproducibility
    pl.seed_everything(args.seed)

    # Load box office data
    test_df = pd.read_excel(Path(args.data_folder + 'test.xlsx'), parse_dates=['start_time'])

    movies_id = test_df['id'].values

    # Load comment
    test_comment_text_encoding = torch.load(Path(args.data_folder + 'test_comment_text_encoding.pth'))
                                            
    test_introduction_encoding = torch.load(Path(args.data_folder + 'test_comment_text_encoding.pth'))

    # pic root and photos root
    pic_root = Path(args.data_folder + '/pic')
    photos_root = Path(args.data_folder + '/photos')

    # get loader
    test_loader = DatasetPrepare(test_df, test_comment_text_encoding, test_introduction_encoding, pic_root,photos_root,
                                  args.embedding_dim,args.gpu_num).get_loader(batch_size=1, train=False)


    # Create model
    model = multimodal(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        comment_len = args.comment_len,
        num_heads=args.num_attn_heads,
        num_layers=args.num_hidden_layers,
        use_text=args.use_text,
        use_img=args.use_img,
        use_encoder_mask=args.use_encoder_mask,
        autoregressive=args.autoregressive,
        gpu_num=args.gpu_num
    )

    model.load_state_dict(torch.load(args.best_model_path)['state_dict'], strict=False)

    # Forecast the testing set
    model.to(device)
    model.eval()
    origin_data, forecasts, attns = [], [], []
    for test_data in tqdm(test_loader, total=len(test_loader), ascii=True):
        with torch.no_grad():
            test_data = [tensor.to(device) for tensor in test_data]
            box_office, introduction_encoding, comments_text_encoding, images, temporal_features = test_data
            y_pred, att = model(introduction_encoding, comments_text_encoding, images, temporal_features)
            forecasts.append(y_pred.detach().cpu().numpy().flatten()[:args.output_dim])
            origin_data.append(box_office.detach().cpu().numpy().flatten()[:args.output_dim])
            attns.append(att.detach().cpu().numpy())

    attns = np.stack(attns)
    forecasts = np.array(forecasts)
    origin_data = np.array(origin_data)
    print_error_metrics(origin_data, forecasts)

    torch.save({'results': forecasts, 'origin_data': origin_data, 'id': movies_id.tolist()},
               Path('results/best_model.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Box office forecasting')

    # General arguments
    parser.add_argument('--model_name', type=str, default='multimodal')
    # parser.add_argument('--best_model_path', type=str)
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--ckpt_path', type=str, default='ckpt')
    parser.add_argument('--best_model_path', type=str, default='ckpt/multimodal/multimodal_epoch=9_model_save.ckpt')
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--seed', type=int, default=21)

    # Model specific arguments
    parser.add_argument('--comment_len', type=int, default=100)
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=20)
    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=int, default=0)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=1)

    # wandb arguments
    # parser.add_argument('--wandb_run', type=str, default='Run1')

    args = parser.parse_args()
    run(args)
