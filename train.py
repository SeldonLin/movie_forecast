import os
import argparse
import wandb
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime
from model import multimodal
from data_process import DatasetPrepare
from tqdm import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(args):
    print(args)
    # Seeds for reproducibility (By default we use the number 21)
    pl.seed_everything(args.seed)

    # Load box office data and comment_text_encoding
    train_df = pd.read_excel(Path(args.data_folder + 'train.xlsx'))
    train_introduction_encoding = torch.load(Path(args.data_folder + 'train_introduction_emcoding.pth'))
    train_comment_text_encoding = torch.load(Path(args.data_folder + 'train_comment_text_encoding.pth'))

    test_df = pd.read_excel(Path(args.data_folder + 'test.xlsx'))
    test_introduction_encoding = torch.load(Path(args.data_folder + 'test_introduction_emcoding.pth'))
    test_comment_text_encoding = torch.load(Path(args.data_folder + 'test_comment_text_encoding.pth'))
                                          

    # pic root and photos root
    pic_root = Path(args.data_folder + '/pic')
    photos_root = Path(args.data_folder + '/photos')

    # get loader
    train_loader = DatasetPrepare(train_df,train_comment_text_encoding, train_introduction_encoding, pic_root, photos_root, args.embedding_dim, args.gpu_num).get_loader(batch_size=args.batch_size, train=True)
    test_loader = DatasetPrepare(test_df, test_comment_text_encoding, test_introduction_encoding, pic_root, photos_root, args.embedding_dim, args.gpu_num).get_loader(batch_size=1, train=False)

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

    # Model Training
    # Define model saving procedure
    model_savename = args.model_name + '_model_save'

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.ckpt_path + '/'+args.model_name,
        filename=args.model_name+'_{epoch}_model_save',
        monitor='val_mae',
        mode='min',
        save_top_k=1
    )

    '''
    # use wandb
    wandb.init(project=args.wandb_proj, name=args.wandb_run)
    wandb_logger = pl_loggers.WandbLogger()
    wandb_logger.watch(model)
    '''

    # use Tensorboard:
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.log_dir + '/', name=args.model_name + '_log')

    trainer = pl.Trainer(gpus=0, max_epochs=args.epochs, check_val_every_n_epoch=5,
                         logger=tb_logger, callbacks=[checkpoint_callback])

    # Fit model
    trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=test_loader)

    # Print out path of best model
    print('best_model_path:',checkpoint_callback.best_model_path)


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

    '''
    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='username-here')
    parser.add_argument('--wandb_proj', type=str, default='multimodal')
    parser.add_argument('--wandb_run', type=str, default='Run1')
    '''

    args = parser.parse_args()
    run(args)
