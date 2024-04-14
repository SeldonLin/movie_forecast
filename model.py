import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from fairseq.optim.adafactor import Adafactor
from encoder_decoder import DummyEmbedder,ImageEmbedder,FusionNetwork,TransformerDecoderLayer,TimeDistributed,PositionalEncoding,CommentEmbedder


class multimodal(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, output_dim, comment_len, num_heads, num_layers, use_text, use_img, \
                 gpu_num, use_encoder_mask=1,
                 autoregressive=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_len = output_dim
        self.comment_len = comment_len
        self.use_encoder_mask = use_encoder_mask
        self.autoregressive = autoregressive
        self.gpu_num = gpu_num
        self.save_hyperparameters()

        # Encoder
        self.dummy_encoder = DummyEmbedder(embedding_dim)
        self.image_encoder = ImageEmbedder()
        self.static_feature_encoder = FusionNetwork(embedding_dim, hidden_dim, use_img, use_text)
        self.comment_encoder = CommentEmbedder(output_dim,embedding_dim, hidden_dim, comment_len)

        # Decoder
        self.decoder_linear = TimeDistributed(nn.Linear(1, hidden_dim))
        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_dim, nhead=num_heads, \
                                                dim_feedforward=self.hidden_dim * 4, dropout=0.1)

        if self.autoregressive: self.pos_encoder = PositionalEncoding(hidden_dim, max_len=20)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, self.output_len if not self.autoregressive else 1),
            nn.Dropout(0.2)
        )

    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:' + str(self.gpu_num))
        return mask

    def forward(self, introduction_encoding, comments_text_encoding, images, temporal_features):
        # Encode features and get inputs
        img_encoding = self.image_encoder(images) # (706, 32)
        dummy_encoding = self.dummy_encoder(temporal_features)  # (706, 32)
        # introduction_encoding (706, 32)
        introduction_encoding.to('cuda:'+str(self.gpu_num))
        comments_encoding = self.comment_encoder(comments_text_encoding)
        # comments_encoding (100, 706, 64)
        

        # Fuse static features together
        static_feature_fusion = self.static_feature_encoder(introduction_encoding, img_encoding, dummy_encoding)
        # static_feature_fusion (706, 64)
        if self.autoregressive == 1:
            # Decode
            tgt = torch.zeros(self.output_len, comments_encoding.shape[1], comments_encoding.shape[-1]).to('cuda:' + str(self.gpu_num))
            tgt[0] = static_feature_fusion
            tgt = self.pos_encoder(tgt)
            tgt_mask = self._generate_square_subsequent_mask(self.output_len)
            memory = comments_encoding
            decoder_out, attn_weights = self.decoder(tgt, memory, tgt_mask)
            forecast = self.decoder_fc(decoder_out)
        else:
            # Decode (generatively/non-autoregressively)
            tgt = static_feature_fusion.unsqueeze(0)
            memory = comments_encoding
            decoder_out, attn_weights = self.decoder(tgt, memory)
            forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len), attn_weights

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        return [optimizer]

    def training_step(self, train_batch, batch_idx):
        box_office, introduction_encoding, comments_text_encoding, images, temporal_features = train_batch
        forecasted_sales, _ = self.forward(introduction_encoding, comments_text_encoding, images, temporal_features)
        loss = F.mse_loss(box_office, forecasted_sales.squeeze())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, test_batch, batch_idx):
        box_office, introduction_encoding, comments_text_encoding, images, temporal_features = test_batch
        forecasted_sales, _ = self.forward(introduction_encoding, comments_text_encoding, images, temporal_features)
        return box_office.squeeze(), forecasted_sales.squeeze()

    def validation_epoch_end(self, val_step_outputs):
        box_office, forecasted_sales = [x[0] for x in val_step_outputs], [x[1] for x in val_step_outputs]
        box_office, forecasted_sales = torch.stack(box_office), torch.stack(forecasted_sales)
        loss = F.mse_loss(box_office, forecasted_sales.squeeze())
        mae = F.l1_loss(box_office, forecasted_sales.squeeze())
        self.log('val_mae', mae)
        self.log('val_loss', loss)
        print('Validation MAE:', mae.detach().cpu().numpy(), 'LR:', self.optimizers().param_groups[0]['lr'])