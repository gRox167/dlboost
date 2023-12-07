from meerkat import image
import torch
import torch.nn.functional as f
import zarr
import torchkbnufft as tkbn
from einops import rearrange, reduce, repeat
from monai.inferers import sliding_window_inference
from dlboost.utils import to_png


def predict_step(batch, nufft_adj, predictor=None, patch_size=(1, 320, 320), device=torch.device("cuda"), ch_reduce_fn=torch.sum):
    b, t, ch, ph, z, sp = batch["kspace_data"].shape
    # kspace_traj torch.Size([1, 2, 1, 5, 2, 9600])
    kspace_traj = batch["kspace_traj"].expand(
        -1, -1, ch, -1, -1, -1)
    #  cse torch.Size([b, ch, z, 320, 320])
    cse = rearrange(
        batch["cse"], "b ch z x y -> b () ch () z x y").expand(-1, t, -1, ph, -1, -1, -1)
    # cse torch.Size([b, t, ch, ph, z, 320, 320])
    image_recon, image_init = multi_contrast_predict_v(
        batch["kspace_data_compensated"],
        kspace_traj, cse,
        nufft_adj, predictor=predictor, patch_size=patch_size,
        sw_device=device,
        ch_reduce_fn=ch_reduce_fn)
    return image_recon.abs(), image_init.abs()  # .mT.flip(-2)


def validation_step(self, batch, batch_idx, predictor=None, ch_reduce_fn=torch.sum):
    predictor = self if predictor is None else predictor
    b, t, ch, ph, z, sp = batch["kspace_data"].shape
    # kspace_traj torch.Size([1, 2, 1, 5, 2, 9600])
    batch["kspace_traj"] = batch["kspace_traj"].expand(
        -1, -1, ch, -1, -1, -1)
    # cse torch.Size([b, ch, z, 320, 320])
    batch["cse"] = rearrange(
        batch["cse"], "b ch z x y -> b () ch () z x y").expand(-1, t, -1, ph, -1, -1, -1)
    # cse torch.Size([b, t, ch, ph, z, 320, 320])
    image_recon, image_init = multi_contrast_predict_v(
        batch["kspace_data_compensated"][:, :, :, :, 40:50],
        batch["kspace_traj"], batch["cse"][:, :, :, :, 40:50],
        self.nufft_adj, predictor=predictor, patch_size=self.patch_size,
        sw_device=self.device, sum_reduce_fn=torch.sum)
    # print(image_recon.shape, image_init.shape)
    zarr.save(self.trainer.default_root_dir +
              f'/epoch_{self.trainer.current_epoch}_recon.zarr',
              image_recon[0, 0].abs().mT.flip(-2).numpy(force=True))
    zarr.save(self.trainer.default_root_dir +
              f'/epoch_{self.trainer.current_epoch}_init.zarr',
              image_init[0, 0].abs().mT.flip(-2).numpy(force=True))
    to_png(self.trainer.default_root_dir +
           f'/epoch_{self.trainer.current_epoch}_image_recon.png',
           image_recon[0, 0, 0, 0].mT.flip(-2))
    print("saved to "+f'/epoch_{self.trainer.current_epoch}_image_recon.png')
    to_png(self.trainer.default_root_dir +
           f'/epoch_{self.trainer.current_epoch}_image_init.png',
           image_init[0, 0, 0, 0].mT.flip(-2))


def multi_contrast_predict_v(kspace_data_compensated, kspace_traj, cse, nufft_adj_op, predictor, patch_size, sw_device, ch_reduce_fn=torch.sum):
    r = [forward_contrast(kd, kt, c, nufft_adj_op, predictor, patch_size, sw_device, ch_reduce_fn) for kd, kt, c in zip(
        kspace_data_compensated.unbind(0), kspace_traj.unbind(0), cse.unbind(0))]
    image_recon, image_init = zip(*r)
    image_recon = torch.stack(image_recon, dim=0)
    image_init = torch.stack(image_init, dim=0)
    return image_recon, image_init


def forward_contrast(kspace_data_compensated, kspace_traj, cse, nufft_adj_op, predictor, patch_size, sw_device, ch_reduce_fn=torch.sum):
    r = [forward_ch(kd, kt, c, nufft_adj_op, predictor, patch_size, sw_device, ch_reduce_fn) for kd, kt, c in zip(
        kspace_data_compensated.unbind(0), kspace_traj.unbind(0), cse.unbind(0))]
    print(r)
    image_recon, image_init = zip(*r)
    image_recon = torch.stack(image_recon, dim=0)
    image_init = torch.stack(image_init, dim=0)
    return image_recon, image_init


def forward_ch(kspace_data_compensated, kspace_traj, cse, nufft_adj_op, predictor, patch_size, sw_device, ch_reduce_fn=torch.sum):
    # kspace_traj_ch = kspace_traj[0] if kspace_traj.shape[0] == 1 else kspace_traj
    image_recon, image_init = forward_step(
        kspace_data_compensated=kspace_data_compensated, kspace_traj=kspace_traj, cse=cse, nufft_adj_op=nufft_adj_op, predictor=predictor, patch_size=patch_size, sw_device=sw_device)
    return ch_reduce_fn(image_recon, 0), ch_reduce_fn(image_init, 0)


def forward_step(kspace_data_compensated, kspace_traj, cse, nufft_adj_op, predictor, patch_size, sw_device):
    image_init = nufft_adj_fn(kspace_data_compensated,
                              kspace_traj, nufft_adj_op.to(torch.device("cpu")))
    image_recon = sliding_window_inference(
        image_init, roi_size=patch_size,
        sw_batch_size=1, overlap=0, predictor=predictor.to(sw_device), device=torch.device("cpu"), sw_device=sw_device)  # , mode='gaussian')
    # image_recon = image_init.clone()
    return image_recon * cse.flip(-3).cpu().conj(), image_init.cpu() * cse.flip(-3).cpu().conj()


def nufft_fn(image, omega, nufft_op, norm="ortho"):
    """do nufft on image

    Args:
        image (_type_): b ph z x y
        omega (_type_): b ph complex_2ch l
        nufft_op (_type_): tkbn operator
        norm (str, optional): Defaults to "ortho".
    """
    b, ph, c, l = omega.shape
    image_kx_ky_z = nufft_op(  # torch.squeeze(image, dim=1)
        rearrange(image, "b ph z x y -> (b ph) z x y"),
        rearrange(omega, "b ph c l -> (b ph) c l"), norm=norm)
    image_kx_ky_z = rearrange(
        image_kx_ky_z, "(b ph) z l -> b ph z l", b=b)
    # image_kx_ky_z.unsqueeze_(dim=1)
    return image_kx_ky_z


def nufft_adj_fn(kdata, omega, nufft_adj_op, norm="ortho"):
    """do adjoint nufft on kdata  

    Args:
        kdata (_type_): b ph z l
        omega (_type_): b ph complex_2ch l
        nufft_adj_op (_type_): tkbn operator
        norm (str, optional): Defaults to "ortho".

    Returns:
        _type_: _description_
    """
    b, ph, c, l = omega.shape
    image = nufft_adj_op(rearrange(kdata, "b ph z l -> (b ph) z l"),
                         rearrange(omega, "b ph c l -> (b ph) c l"), norm=norm)
    return rearrange(image, "(b ph) z x y -> b ph z x y", b=b, ph=ph)
