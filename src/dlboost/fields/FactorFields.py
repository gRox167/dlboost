import math
import time

import numpy as np
import skimage
import torch
import torch.nn
import torch.nn.functional as F
from icecream import ic

from utils import N_to_reso, N_to_vm_reso


def grid_mapping(positions, freq_bands, bbox, basis_mapping="sawtooth"):
    # ic(bbox)
    # ic(positions.shape)
    bbox_size = torch.max(bbox[1] - bbox[0])
    # ic(aabbSize.shape, freq_bands.shape)
    scale = bbox_size / freq_bands
    # ic(scale)
    # if basis_mapping == "triangle":
    #     pts_local = (positions - bbox[0]).unsqueeze(-1) % scale
    #     pts_local_int = ((positions - bbox[0]).unsqueeze(-1) // scale) % 2
    #     pts_local = pts_local / (scale / 2) - 1
    #     pts_local = torch.where(pts_local_int == 1, -pts_local, pts_local)
    # elif basis_mapping == "sawtooth":
    # ic(scale, bbox[0], positions[1:2])
    pts_local = (positions - bbox[0])[..., None] % scale
    pts_local = pts_local / (scale / 2) - 1
    pts_local = pts_local.clamp(-1.0, 1.0)
    # elif basis_mapping == "sinc":
    #     pts_local = torch.sin(
    #         (positions - bbox[0])[..., None] / (scale / np.pi) - np.pi / 2
    #     )
    # elif basis_mapping == "trigonometric":
    #     pts_local = (positions - bbox[0])[..., None] / scale * 2 * np.pi
    #     pts_local = torch.cat((torch.sin(pts_local), torch.cos(pts_local)), dim=-1)
    # elif basis_mapping == "x":
    #     pts_local = (positions - bbox[0]).unsqueeze(-1) / scale
    # elif basis_mapping=='hash':
    #     pts_local = (positions - bbox[0])/max(aabbSize)

    return pts_local


def dct_dict(n_atoms_fre, size, n_selete, dim=2):
    """
    Create a dictionary using the Discrete Cosine Transform (DCT) basis. If n_atoms is
    not a perfect square, the returned dictionary will have ceil(sqrt(n_atoms))**2 atoms
    :param n_atoms:
        Number of atoms in dict
    :param size:
        Size of first patch dim
    :return:
        DCT dictionary, shape (size*size, ceil(sqrt(n_atoms))**2)
    """
    # todo flip arguments to match random_dictionary
    p = n_atoms_fre  # int(math.ceil(math.sqrt(n_atoms)))
    dct = np.zeros((p, size))

    for k in range(p):
        basis = np.cos(np.arange(size) * k * math.pi / p)
        if k > 0:
            basis = basis - np.mean(basis)

        dct[k] = basis

    kron = np.kron(dct, dct)
    if 3 == dim:
        kron = np.kron(kron, dct)

    if n_selete < kron.shape[0]:
        idx = [x[0] for x in np.array_split(np.arange(kron.shape[0]), n_selete)]
        kron = kron[idx]

    for col in range(kron.shape[0]):
        norm = np.linalg.norm(kron[col]) or 1
        kron[col] /= norm

    kron = torch.FloatTensor(kron)
    # print(kron.shape)
    return kron


def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


class MLPMixer(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim=16, num_layers=2, hidden_dim=64, pe=0, with_dropout=False
    ):
        super().__init__()

        self.with_dropout = with_dropout
        self.in_dim = in_dim + 2 * in_dim * pe
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pe = pe

        backbone = []
        for l in range(num_layers):
            if l == 0:
                layer_in_dim = self.in_dim
            else:
                layer_in_dim = self.hidden_dim

            if l == num_layers - 1:
                layer_out_dim, bias = out_dim, False
            else:
                layer_out_dim, bias = self.hidden_dim, True

            backbone.append(torch.nn.Linear(layer_in_dim, layer_out_dim, bias=bias))

        self.backbone = torch.nn.ModuleList(backbone)
        # torch.nn.init.constant_(backbone[0].weight.data, 1.0/self.in_dim)

    def forward(self, x, is_train=False):
        # x: [B, 3]
        h = x
        if self.pe > 0:
            h = torch.cat([h, positional_encoding(h, self.pe)], dim=-1)

        if self.with_dropout and is_train:
            h = F.dropout(h, p=0.1)

        for l in range(self.num_layers):
            h = self.backbone[l](h)
            if l != self.num_layers - 1:  # l!=0 and
                h = F.relu(h, inplace=True)
                # h = torch.sin(h)
        # sigma, feat = h[...,0], h[...,1:]
        return h


class DictField(torch.nn.Module):
    def __init__(self, cfg, im_size, device):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.im_size = im_size
        # self.matMode = [[0, 1], [0, 2], [1, 2]]
        # self.vecMode = [2, 1, 0]
        # self.n_scene, self.scene_idx = 1, 0

        # self.alphaMask = None
        self.coeff_type, self.basis_type = cfg.model.coeff_type, cfg.model.basis_type

        self.bbox = [[0.0, 0.0], list(im_size)]
        self.setup_params(self.bbox)

        if self.cfg.model.coeff_type != "none":
            self.coeffs = self.init_coef()

        if self.cfg.model.basis_type != "none":
            self.basises = self.init_basis()

        out_dim = cfg.model.out_dim
        in_dim = sum(cfg.model.basis_dims)
        self.linear_mat = MLPMixer(
            in_dim,
            out_dim,
            num_layers=cfg.model.num_layers,
            hidden_dim=cfg.model.hidden_dim,
            with_dropout=cfg.model.with_dropout,
        ).to(device)

        """     
        if 'reconstruction' in cfg.defaults.mode:
            # self.cur_volumeSize = N_to_reso(cfg.training.volume_resoInit, self.aabb)
            # self.update_renderParams(self.cur_volumeSize)

            view_pe, fea_pe = cfg.renderer.view_pe, cfg.renderer.fea_pe
            num_layers, hidden_dim = cfg.renderer.num_layers, cfg.renderer.hidden_dim
            self.renderModule = MLPRender_Fea(inChanel=out_dim - 1, num_layers=num_layers, hidden_dim=hidden_dim,
                                              viewpe=view_pe, feape=fea_pe).to(device)

            self.is_unbound = self.cfg.dataset.is_unbound
            if self.is_unbound:
                self.bg_len = 0.2
                self.inward_aabb = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]).to(device)
                self.aabb = self.inward_aabb * (1 + self.bg_len)
            else:
                self.inward_aabb = self.aabb

            # self.freq_bands = torch.FloatTensor(cfg.model.freq_bands).to(device)
            self.cur_volumeSize = N_to_reso(cfg.training.volume_resoInit ** self.in_dim, self.aabb)
            self.update_renderParams(self.cur_volumeSize)
            """
        print("=====> total parameters: ", self.n_parameters())

    def setup_params(self, bbox):
        # self.in_dim = len(aabb[0]) - 1
        self.in_dim = 2

        self.basis_dims = np.array(self.cfg.model.basis_dims)  # [32,32,32,16,16,16]
        self.basis_reso = np.array(self.cfg.model.basis_resos)  # [32,51,70,89,108,128]
        # self.basis_reso_y = np.round(
        #     self.basis_reso / self.im_size[0] * self.im_size[1]
        # )
        ic(self.basis_dims)
        ic(self.basis_reso)
        # ic(self.basis_reso_y)

        self.freq_bands = max(bbox[1]) / torch.FloatTensor(self.basis_reso).to(
            self.device
        )

        self.bbox = torch.FloatTensor(bbox).to(self.device)
        ic(self.bbox)
        self.coeff_reso = self.cfg.model.coeff_reso
        ic(self.coeff_reso)

    def init_coef(self):
        coeffs = self.cfg.model.coef_init * torch.ones(
            (1, sum(self.basis_dims), *self.coeff_reso), device=self.device
        )
        # size([1,sum([32,32,32,16,16,16]), H/4, W/4])
        # return torch.nn.ParameterList(coeffs)
        ic(coeffs.shape)
        return torch.nn.Parameter(coeffs)

    def init_basis(self):
        basises = []
        for i, (basis_dim, reso) in enumerate(zip(self.basis_dims, self.basis_reso)):
            ic([1, basis_dim] + [reso] * self.in_dim)
            basises.append(
                torch.nn.Parameter(
                    dct_dict(
                        int(np.power(basis_dim, 1.0 / self.in_dim) + 1),
                        reso,
                        n_selete=basis_dim,
                        dim=self.in_dim,
                    )
                    .reshape([1, basis_dim] + [reso] * self.in_dim)
                    .to(self.device)
                )
            )

        return torch.nn.ParameterList(basises)

    def get_coeff(self, xyz_sampled):
        N_points, dim = xyz_sampled.shape
        pts = self.normalize_coord(xyz_sampled).view([1, -1] + [1] * (dim - 1) + [dim])
        # normalize to -1, 1
        coeffs = (
            F.grid_sample(
                self.coeffs,
                pts,
                mode=self.cfg.model.coef_mode,
                align_corners=False,
                # align_corners=True,
                padding_mode="border",
                # padding_mode="zeros",
            )
            .view(-1, N_points)
            .t()
        )
        return coeffs

    def get_basis(self, x):
        # ic(x.shape)
        N_points = x.shape[0]
        # x = x[..., :-1]
        # ic(self.freq_bands)
        freq_len = len(self.freq_bands)
        xyz = grid_mapping(
            x,
            self.freq_bands,
            self.bbox,
            self.cfg.model.basis_mapping,
        ).view(1, *([1] * (self.in_dim - 1)), -1, self.in_dim, freq_len)
        # ic(xyz.shape)
        basises = []
        for i in range(freq_len):
            basises.append(
                F.grid_sample(
                    self.basises[i],
                    xyz[..., i],
                    mode=self.cfg.model.basis_mode,
                    align_corners=True,
                )
                .view(-1, N_points)
                .T
            )
        if isinstance(basises, list):
            basises = torch.cat(basises, dim=-1)
        return basises

    @torch.no_grad()
    def normalize_basis(self):
        for basis in self.basises:
            basis.data = basis.data / torch.norm(basis.data, dim=(2, 3), keepdim=True)

    def get_coding(self, x):
        if self.cfg.model.coeff_type != "none" and self.cfg.model.basis_type != "none":
            coeff = self.get_coeff(x)
            basises = self.get_basis(x)
            return basises * coeff, coeff
        elif self.cfg.model.coeff_type != "none":
            coeff = self.get_coeff(x)
            return coeff, coeff
        elif self.cfg.model.basis_type != "none":
            basises = self.get_basis(x)
            return basises, basises

    def n_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        if "fix" in self.cfg.model.basis_type:
            total -= self.T_basis
        return total

    def get_optparam_groups(self, lr_small=0.001, lr_large=0.02):
        grad_vars = []
        if self.cfg.training.linear_mat:
            grad_vars += [{"params": self.linear_mat.parameters(), "lr": lr_small}]

        if "none" != self.coeff_type and self.cfg.training.coeff:
            grad_vars += [{"params": self.coeffs.parameters(), "lr": lr_large}]

        if (
            "fix" not in self.cfg.model.basis_type
            and "none" != self.cfg.model.basis_type
            and self.cfg.training.basis
        ):
            grad_vars += [{"params": self.basises.parameters(), "lr": lr_large}]

        return grad_vars

    def set_optimizable(self, items, statue):
        for item in items:
            if item == "basis" and self.cfg.model.basis_type != "none":
                for item in self.basises:
                    item.requires_grad = statue
            elif item == "coeff" and self.cfg.model.coeff_type != "none":
                for item in self.basises:
                    item.requires_grad = statue
            elif item == "proj":
                self.linear_mat.requires_grad = statue

    def TV_loss(self, reg):
        total = 0
        for idx in range(len(self.basises)):
            total = total + reg(self.basises[idx]) * 1e-2
        return total

    def forward(self, coordinates):
        feats, coeffs = self.get_coding(coordinates)
        # ic(feats.shape)
        output = self.linear_mat(feats).reshape(*[640, 368], 2)
        return output

    def normalize_coord(self, xyz_sampled):
        invaabbSize = 2.0 / (self.bbox[1] - self.bbox[0])
        return (xyz_sampled - self.bbox[0]) * invaabbSize - 1

    def basis2density(self, density_features):
        if self.cfg.renderer.fea2denseAct == "softplus":
            return F.softplus(density_features + self.cfg.renderer.density_shift)
        elif self.cfg.renderer.fea2denseAct == "relu":
            return F.relu(density_features + self.cfg.renderer.density_shift)

    """
    @torch.no_grad()
    def cal_mean_coef(self, state_dict):
        if 'grid' in self.coeff_type or 'mlp' in self.coeff_type:
            key_list = []
            for item in state_dict.keys():
                if 'coeffs.0' in item:
                    key_list.append(item)

            for key in key_list:
                average = torch.zeros_like(state_dict[key])
                for i in range(self.n_scene):
                    item = key.replace('0', f'{i}', 1)
                    average += state_dict[item]
                    state_dict.pop(item, None)
                average /= self.n_scene
                state_dict[key] = average
        elif 'vec' in self.coeff_type:
            state_dict['coeffs.0'] = torch.mean(state_dict['coeffs.0'], dim=-1, keepdim=True)
        elif 'cp' in self.coeff_type or 'vm' in self.coeff_type:
            for i in range(3):
                state_dict[f'coeffs.{i}'] = torch.mean(state_dict[f'coeffs.{i}'], dim=-1, keepdim=True)

        return state_dict
        """

    def save(self, path):
        ckpt = {"state_dict": self.state_dict(), "cfg": self.cfg}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({"alphaMask.shape": alpha_volume.shape})
            ckpt.update({"alphaMask.mask": np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({"alphaMask.aabb": self.alphaMask.aabb.cpu()})

        # average the coeff for saving if batch training
        if "reconstruction" in self.cfg.defaults.mode:
            ckpt["state_dict"] = self.cal_mean_coef(ckpt["state_dict"])
        torch.save(ckpt, path)

    def load(self, ckpt):
        if "alphaMask.aabb" in ckpt.keys():
            length = np.prod(ckpt["alphaMask.shape"])
            alpha_volume = torch.from_numpy(
                np.unpackbits(ckpt["alphaMask.mask"])[:length].reshape(
                    ckpt["alphaMask.shape"]
                )
            )
            self.alphaMask = AlphaGridMask(
                self.device,
                ckpt["alphaMask.aabb"].to(self.device),
                alpha_volume.float().to(self.device),
            )
        self.load_state_dict(ckpt["state_dict"])
        volumeSize = N_to_reso(
            self.cfg.training.volume_resoFinal**self.in_dim, self.aabb
        )
        self.update_renderParams(volumeSize)


def get_coordinates(size):
    H, W = size[0], size[1]
    y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing="ij")
    coordinate = torch.stack((y, x), -1).float() + 0.5
    coordinate = coordinate.reshape(-1, 2)

    return coordinate
