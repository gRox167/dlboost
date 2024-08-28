# %%
import math
from typing import Tuple

import einx
import torch
import torch_dct as dct
from torch import Tensor, nn
from torch.nn import (
    GELU,
    Dropout,
    LayerNorm,
    Linear,
    MultiheadAttention,
)


def percentile(t: torch.tensor, l, h):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    l_ = 1 + round(0.01 * float(l) * (t.numel() - 1))
    h_ = 1 + round(0.01 * float(h) * (t.numel() - 1))
    l_th = t.kthvalue(l_).values
    h_th = t.kthvalue(h_).values
    return l_th, h_th


# percentile_v = vmap(percentile, in_dims=(0, None, None), out_dims=0)
# clamp_v = vmap(torch.clamp, in_dims=(0, 0, 0), out_dims=0)
def complex_normalize_abs_95(x, start_dim=0):
    x_abs = x.abs()
    min_95, max_95 = percentile(x_abs.flatten(start_dim), 2.5, 97.5)
    x_abs_clamped = torch.clamp(x_abs, min_95, max_95)
    mean = torch.mean(x_abs_clamped)
    std = torch.std(x_abs_clamped, unbiased=False)
    return (x - mean) / std, mean, std


def complex_normalize_abs(x, start_dim=0):
    x_abs = x.abs()
    x_abs = x_abs.flatten(start_dim)
    mean = torch.mean(x_abs)
    std = torch.std(x_abs, unbiased=False)
    return (x - mean) / std, mean, std


complex_normalize_abs_95_v = torch.vmap(
    complex_normalize_abs_95, in_dims=0, out_dims=(0, 0, 0)
)
complex_normalize_abs_v = torch.vmap(
    complex_normalize_abs, in_dims=0, out_dims=(0, 0, 0)
)


class FourierTokenMixing(nn.Module):
    def __init__(
        self,
        d_model,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.conv_filters = nn.ModuleList(
            [
                nn.Conv1d(d_model, d_model, 3, padding=1),
                nn.Conv1d(d_model, d_model, 5, padding=2),
                nn.Conv1d(d_model, d_model, 7, padding=3),
                nn.Conv1d(d_model, d_model, 11, padding=5),
            ]
        )
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None, need_weights=False):
        x = self.norm(x)
        x = einx.rearrange("b len c -> b c len", x)
        x = dct.dct(x, norm="ortho")
        x = einx.sum(
            "[num] b c l", torch.stack([conv(x) for conv in self.conv_filters])
        )
        x = dct.idct(x, norm="ortho")
        x = einx.rearrange("b c len -> b len c", x)

        return self.dropout(x)


class SelfAttentionTokenMixing(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout: float = 0.1,
        batch_first: bool = True,
        layer_norm_eps: float = 1e-5,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.self_attention = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        # return x
        x = self.norm(x)
        x = self.self_attention(
            x,
            x,
            x,
            need_weights=False,
        )[0]
        return self.dropout(x)


class EncoderDecoderTokenMixing(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout: float = 0.1,
        batch_first: bool = True,
        layer_norm_eps: float = 1e-5,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.attention = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout)

    def forward(self, x, memory):
        x = self.norm(x)
        x = self.attention(
            x,
            memory,
            memory,
            need_weights=False,
        )[0]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.

    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    TransformerEncoderLayer can handle either traditional torch.tensor inputs,
    or Nested Tensor inputs.  Derived classes are expected to similarly accept
    both input formats.  (Not all combinations of inputs are currently
    supported by TransformerEncoderLayer while Nested Tensor is in prototype
    state.)

    If you are implementing a custom layer, you may derive it either from
    the Module or TransformerEncoderLayer class.  If your custom layer
    supports both torch.Tensors and Nested Tensors inputs, make its
    implementation a derived class of TransformerEncoderLayer. If your custom
    Layer supports only torch.Tensor inputs, derive its implementation from
    Module.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.fourier_sa = FourierTokenMixing(
            d_model, dropout=dropout, layer_norm_eps=layer_norm_eps
        )
        self.self_attn = SelfAttentionTokenMixing(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            layer_norm_eps=layer_norm_eps,
        )
        self.feed_forward = nn.Sequential(
            LayerNorm(d_model, eps=layer_norm_eps, bias=bias),
            Linear(d_model, dim_feedforward, bias=bias),
            GELU(),
            Dropout(dropout),
            Linear(dim_feedforward, d_model, bias=bias),
            Dropout(dropout),
        )

    def forward(
        self,
        src: Tensor,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        x = x + self._alt_sa_block(x)
        x = x + self.feed_forward(x)
        return x

    def _alt_sa_block(
        self,
        x: Tensor,
    ) -> Tensor:
        b = x.shape[0]
        x = einx.rearrange("b sp l c -> (b sp) l c", x)
        x = x + self.self_attn(x)
        x = einx.rearrange("(b sp) l c -> (b l) sp c", x, b=b)
        x = x + self.self_attn(x)
        return einx.rearrange("(b l) sp c -> b sp l c", x, b=b)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.fourier_sa = FourierTokenMixing(
            d_model, dropout=dropout, layer_norm_eps=layer_norm_eps
        )
        self.self_attn = SelfAttentionTokenMixing(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            layer_norm_eps=layer_norm_eps,
        )
        self.mha = EncoderDecoderTokenMixing(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            layer_norm_eps=layer_norm_eps,
        )
        self.feed_forward = nn.Sequential(
            LayerNorm(d_model, eps=layer_norm_eps, bias=bias),
            Linear(d_model, dim_feedforward, bias=bias),
            GELU(),
            Dropout(dropout),
            Linear(dim_feedforward, d_model, bias=bias),
            Dropout(dropout),
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
    ) -> Tensor:
        # return memory
        out_sp = tgt.shape[1]
        x = einx.rearrange("b out_sp l c -> b (out_sp l) c", tgt, out_sp=out_sp)
        # memory = einx.rearrange(
        #     "b src_sp l c -> b (src_sp l) c ",
        #     memory,  # , out_sp=out_sp
        # )
        memory = einx.mean(
            "b src_sp [l] c",
            memory,  # , out_sp=out_sp
        )
        x = x + self.mha(x, memory)
        x = einx.rearrange("b (out_sp l) c -> b out_sp l c ", x, out_sp=out_sp)
        x = x + self._alt_sa_block(tgt)
        x = x + self.feed_forward(x)
        return x

    def _alt_sa_block(
        self,
        x: Tensor,
    ) -> Tensor:
        b = x.shape[0]
        x = einx.rearrange("b sp l c -> (b sp) l c", x)
        x = x + self.self_attn(x)
        x = einx.rearrange("(b sp) l c -> (b l) sp c", x, b=b)
        x = x + self.self_attn(
            x,
        )
        return einx.rearrange("(b l) sp c -> b sp l c", x, b=b)


class KSpaceTransformer(nn.Module):
    def __init__(
        self,
        channel=2,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=512,
        #  HR_conv_channel=64, HR_conv_num=3, HR_kernel_size=5,
        position_dim=2,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder_embedding_layer = nn.Sequential(
            nn.Linear(channel, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # self.pe_layer = PositionalEncoding(
        #     d_model, position_dim=position_dim, magnify=250.0
        # )
        self.pe_layer = PositionalEncoding(d_model)
        # self.pe_layer = LearnablePositionalEncoding(
        #     in_dim=position_dim, out_dim=d_model
        # )

        self.encoder_list = nn.ModuleList(
            [
                EncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder_list = nn.ModuleList(
            [
                DecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
                for _ in range(num_decoder_layers)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            GELU(),
            nn.Linear(d_model, channel),
        )

    def forward(self, src, src_pos, out_pos):
        """
        Args:
          src: [bs, sp, len, 2] intensity of sampled points
          src_pos: [bs, sp, len, 2] normalized coordinates of sampled points
          out_pos: [bs, query_len, 2] normalized coordinates of unsampled points
        Returns:
        """
        # normalization

        # b, sp, l = src.shape
        # src = torch.fft.ifftshift(torch.fft.ifft(src, dim=2, norm="ortho"), dim=2)
        src, mean, std = complex_normalize_abs_95_v(src)
        src = torch.view_as_real(src.contiguous())

        b, _, sp, l = src_pos.size()
        src_pos = (
            (einx.rearrange("b c sp l -> b (sp l) c", src_pos) + torch.pi)
            / 2
            / torch.pi
        )
        src_pe = einx.rearrange("b (sp l) c -> b sp l c", self.pe_layer(src_pos), sp=sp)
        # encoder
        src_embed = self.encoder_embedding_layer(src)
        hidden_state = src_embed + src_pe  # [bs, src_sp, src_len, d]
        for encoder in self.encoder_list:
            hidden_state = encoder(hidden_state)

        # decoder
        b, _, out_sp, l = out_pos.size()
        out_pos = (
            (einx.rearrange("b c out_sp l -> b (out_sp l) c", out_pos) + torch.pi)
            / 2
            / torch.pi
        )
        out_pe = einx.rearrange(
            "b (out_sp l) c -> b out_sp l c", self.pe_layer(out_pos), out_sp=out_sp
        )

        output: torch.Any | Tuple[torch.Any] = out_pe
        for decoder in self.decoder_list:
            output = decoder(output, hidden_state)  # [bs, out_sp, out_len, d]
        x = self.head(output)

        # re-normalize
        x = torch.view_as_complex(x.contiguous())
        x = einx.multiply("b sp l , b -> b sp l", x, std)
        x = einx.add("b sp l , b -> b sp l", x, mean)

        return x  # [bs, out_sp, out_len]


class SpokeTransformer(nn.Module):
    def __init__(
        self,
        channel=2,
        d_model=1024,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=12,
        dim_feedforward=2048,
        #  HR_conv_channel=64, HR_conv_num=3, HR_kernel_size=5,
        position_dim=2,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder_embedding_layer = nn.Sequential(
            nn.Linear(channel, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # self.pe_layer = PositionalEncoding(
        #     d_model, position_dim=position_dim, magnify=250.0
        # )
        self.pe_layer = PositionalEncoding(d_model)
        # self.pe_layer = LearnablePositionalEncoding(
        #     in_dim=position_dim, out_dim=d_model
        # )

        self.encoder_list = nn.ModuleList(
            [
                EncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder_list = nn.ModuleList(
            [
                DecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
                for _ in range(num_decoder_layers)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            GELU(),
            nn.Linear(d_model, channel),
        )

    def forward(self, src, src_pos, out_pos):
        """
        Args:
          src: [bs, sp, len, 2] intensity of sampled points
          src_pos: [bs, sp, len, 2] normalized coordinates of sampled points
          out_pos: [bs, query_len, 2] normalized coordinates of unsampled points
        Returns:
        """
        # normalization
        src, mean, std = complex_normalize_abs_95_v(src)

        src = torch.view_as_real(src.contiguous())
        src_embed = self.encoder_embedding_layer(src)

        b, _, sp, l = src_pos.size()
        src_pos = einx.rearrange("b c sp l -> b (sp l) c", src_pos)
        src_pe = einx.rearrange("b (sp l) c -> b sp l c", self.pe_layer(src_pos), sp=sp)
        # encoder
        hidden_state = src_embed + src_pe  # [bs, src_sp, src_len, d]
        for encoder in self.encoder_list:
            hidden_state = encoder(hidden_state)

        # decoder
        b, _, out_sp, l = out_pos.size()
        out_pos = einx.rearrange("b c out_sp l -> b (out_sp l) c", out_pos)
        out_pe = einx.rearrange(
            "b (out_sp l) c -> b out_sp l c", self.pe_layer(out_pos), out_sp=out_sp
        )

        output: torch.Any | Tuple[torch.Any] = out_pe
        for decoder in self.decoder_list:
            output = decoder(output, hidden_state)  # [bs, out_sp, out_len, d]
        x = self.head(output)

        # re-normalize
        x = torch.view_as_complex(x.contiguous())
        x = einx.multiply("b sp l , b -> b sp l", x, std)
        x = einx.add("b sp l , b -> b sp l", x, mean)

        return x  # [bs, out_sp, out_len]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, in_dim=2, out_dim=128):
        super().__init__()
        self.pos_embedding = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.pos_embedding(x)


# class PositionalEncoding(nn.Module):
#     def __init__(self, dim=128, temperature=10000, position_dim=4, magnify=250):
#         super().__init__()
#         # Compute the division term
#         # note that the positional dim for all axis are equal to dim/position_axis_num
#         self.dim = dim
#         self.position_dim = position_dim
#         self.dim_of_each_position_dim = self.dim // position_dim
#         # self.div_term = nn.Parameter(torch.exp(torch.arange(0, self.dim/position_axis_num, 2) * -(2 * math.log(10000.0) / self.dim)), requires_grad=False)      # [32]
#         omega = (
#             torch.arange(0, self.dim_of_each_position_dim, 2)
#             / self.dim_of_each_position_dim
#         )
#         self.register_buffer("freqs", 1.0 / (temperature**omega))
#         self.magnify = magnify

#     def forward(self, position_norm):
#         """
#         given position_norm:[bs, token_num, position_axis_num]
#         return pe [bs, token_num, dim]
#         """
#         positional_embeddings = []
#         for i in range(self.position_dim):
#             inside = torch.einsum(
#                 "bi,j -> bij", position_norm[..., i] * self.magnify, self.freqs
#             )
#             sin = torch.stack([inside.sin(), inside.cos()], dim=-1)
#             positional_embedding = torch.flatten(sin, -2, -1)
#             positional_embeddings.append(positional_embedding)
#         positional_embedding = torch.cat(positional_embeddings, dim=-1)
#         return positional_embedding


# not decisive
class PositionalEncoding(nn.Module):
    def __init__(self, pe_dim=128, magnify=100.0):
        super().__init__()
        # Compute the division term
        # note that the positional dim for x and y is equal to dim_of_pe/2
        self.dim = pe_dim
        self.div_term = nn.Parameter(
            torch.exp(
                torch.arange(0, self.dim / 2, 2) * -(2 * math.log(10000.0) / self.dim)
            ),
            requires_grad=False,
        )  # [32]
        self.magnify = magnify

    def forward(self, p_norm):
        """
        given position:[bs, h*w*0.2, 2]

        return pe
        """

        p = p_norm * self.magnify

        no_batch = False
        if p.dim() == 2:  # no batch size
            no_batch = True
            p = p.unsqueeze(0)

        p_x = p[:, :, 0].unsqueeze(2)  # [bs, h*w*0.2, 1]
        p_y = p[:, :, 1].unsqueeze(2)
        # assert p_x.shape[1] == p_y.shape[1]
        pe_x = torch.zeros(p_x.shape[0], p_x.shape[1], self.dim // 2).to(
            torch.device("cuda")
        )  # [bs, h*w*0.2, 64]
        pe_x[:, :, 0::2] = torch.sin(p_x * self.div_term)  # [bs, h*w*0.2, 32]
        pe_x[:, :, 1::2] = torch.cos(p_x * self.div_term)

        pe_y = torch.zeros(p_x.shape[0], p_x.shape[1], self.dim // 2).to(
            torch.device("cuda")
        )  # [bs, h*w*0.2, 64]
        pe_y[:, :, 0::2] = torch.sin(p_y * self.div_term)
        pe_y[:, :, 1::2] = torch.cos(p_y * self.div_term)

        pe = torch.cat([pe_x, pe_y], dim=2)  # [bs, h*w*0.2, 128]

        if no_batch:
            pe = pe.squeeze(0)

        # [len, dim]
        return pe


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


# %%
if __name__ == "__main__":
    # test transformer
    bs = 1
    sp1 = 10
    sp2 = 40
    src_len = 640
    out_len = 640
    c = 2
    # d_model = 64
    # nhead = 8
    # num_encoder_layers = 6
    # num_decoder_layers = 6
    # dim_feedforward = 256
    # dropout = 0.1
    # activation = "gelu"

    src = torch.randn(bs, sp1, src_len, c)
    src_pos = torch.randn(bs, 2, sp1, src_len)
    out_pos = torch.randn(bs, 2, sp2, out_len)

    transformer = KSpaceTransformer().cuda()
    out = transformer(src.cuda(), src_pos.cuda(), out_pos.cuda())
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

# %%
