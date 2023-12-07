from einops import rearrange, reduce, repeat
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class Transformer(nn.Module):

    def __init__(self, channel=2, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048,
                #  HR_conv_channel=64, HR_conv_num=3, HR_kernel_size=5,
                 dropout=0.1, activation="gelu"):
        super().__init__()

        # self.num_HRdecoder_layers = num_HRdecoder_layers

        self.encoder_embedding_layer = nn.Sequential(
            nn.Linear(channel, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )

        self.pe_layer = PositionalEncoding(d_model, position_dim=4,magnify=250.0)

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer,
                                              num_layers=num_decoder_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, channel)
        )

    def forward(self, src, src_pos, out_pos):
        """
        Args:
          src: [bs, src_len, c] intensity of sampled points
          lr_pos: [bs, lh*lw, 2] normalized coordinates of LR query points
          src_pos: [bs, src_len, 2] normalized coordinates of sampled points
          hr_pos: [bs, query_len, 2] normalized coordinates of unsampled points
          k_us: [bs, h, w, c] zero-filled specturm
          mask: [bs, h, w, c] undersampling mask, 1 means unsampled and 0 means sampled
          unsampled_pos: [bs, query_len, 2] coordinates of unsampled points(unnormalized)
          up_scale: LR upsample ratio to HR
        Returns:
        """
        # encoder
        src_embed = self.encoder_embedding_layer(src)
        src_pe = self.pe_layer(src_pos)      # [bs, src_len, d]
        hidden_state = self.encoder(
            src_embed, pos=src_pe)  # [bs, src_len, d]
        # decoder
        out_pe = self.pe_layer(out_pos)        # [bs, out_len, d]
        hidden_state = self.decoder(hidden_state, out_pe) # [bs, out_len, d]
        return self.head(hidden_state)  # [bs, out_len, c]

class PositionalEncoding(nn.Module):

    def __init__(self, dim=128, temperature=10000, position_dim=4, magnify = 250):
        super().__init__()
        # Compute the division term
        # note that the positional dim for all axis are equal to dim/position_axis_num
        self.dim = dim
        self.position_dim = position_dim
        self.dim_of_each_position_dim= self.dim // position_dim
        # self.div_term = nn.Parameter(torch.exp(torch.arange(0, self.dim/position_axis_num, 2) * -(2 * math.log(10000.0) / self.dim)), requires_grad=False)      # [32]
        omega = torch.arange(0, self.dim_of_each_position_dim,2) / self.dim_of_each_position_dim
        self.freqs = 1.0 / (temperature ** omega)
        self.magnify = magnify

    def forward(self, position_norm):
        """
        given position_norm:[bs, token_num, position_axis_num]
        return pe [bs, token_num, dim]
        """
        positional_embeddings = []
        for i in range(self.position_dim):
            inside = torch.einsum("bi,j -> bij", position_norm[...,i]*self.magnify, self.freqs)
            sin = torch.stack([inside.sin(), inside.cos()], dim=-1)
            positional_embedding = torch.flatten(sin,-2,-1)
            positional_embeddings.append(positional_embedding)
        positional_embedding = torch.cat(positional_embeddings, dim=-1)
        return positional_embedding


if __name__ == "__main__":
    # test transformer
    bs = 1
    src_len = 9600
    out_len = 9600
    c = 2
    d_model = 64
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 256
    dropout = 0.1
    activation = "gelu"

    src = torch.randn(bs, src_len, c)
    src_pos = torch.randn(bs, src_len, 2)
    out_pos = torch.randn(bs, out_len, 2)

    transformer = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                              num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                              dropout=dropout, activation=activation)
    out = transformer(src, src_pos, out_pos)
    print(out.shape)
    print(out)
    # test positional encoding
    # bs = 2
    # token_num = 100
    # position_axis_num = 4
    # dim = 128
    # temperature = 10000
    # position_encoding = PositionalEncoding(dim, temperature, position_axis_num)
    # position_norm = torch.randn(bs, token_num, position_axis_num)
    # pe = position_encoding(position_norm)
    # print(pe.shape)
    # print(pe)