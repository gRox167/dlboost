
from ctypes import Union
import lightning.pytorch as pl
import torch
from torch import nn
from torch.nn import functional as f
from utils import losses
from typing import Optional, Sequence
from models.SpatialTransformNetwork import SpatialTransformNetwork


class ReconRegis(pl.LightningModule):
    def __init__(
        self,
        recon_module: nn.Module,
        regis_module: nn.Module,
        STN_size: Union[int,Sequence[int]] = [64,64,64],
        patch_size = [64,64,64],
        is_optimize_regis: bool = False,
        lambda_: float = 6.0,
        loss_regis_mse_COEFF: float = 0.0,
        recon_loss_fn=nn.MSELoss,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()
        self.automatic_optimization=False
        self.save_hyperparameters()
        self.recon_module = recon_module
        self.regis_module = regis_module
        self.stn = SpatialTransformNetwork(STN_size)
        self.is_optimize_regis = is_optimize_regis
        self.lambda_ = lambda_
        self.loss_regis_mse_COEFF = loss_regis_mse_COEFF
        self.recon_loss_fn = recon_loss_fn

    def forward(self, x):
        return self.recon_module(x)

    def training_step(self, batch, batch_idx):
        recon_opt, regis_opt = self.optimizers()

        regis_loss, batch = self.training_step_regis(batch, batch_idx)
        regis_opt.zero_grad()
        self.manual_backward(regis_loss)
        regis_opt.step()

        recon_loss, batch = self.training_step_recon(batch, batch_idx)
        recon_opt.zero_grad()
        self.manual_backward(recon_loss)
        recon_opt.step()

        return regis_loss, recon_loss

    def training_step_regis(self, batch, batch_idx):
        moved_x, moved_y_tran, moved_y, moved_mask, fixed_x, fixed_y_tran, fixed_y, fixed_mask = batch
        if self.is_optimize_regis:
            fixed_y_tran_recon = self.recon_module(fixed_y_tran).detach()
            moved_y_tran_recon = self.recon_module(moved_y_tran).detach()

            fixed_y_tran_recon = f.pad(
                torch.sqrt(torch.sum(fixed_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])
            moved_y_tran_recon = f.pad(
                torch.sqrt(torch.sum(moved_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])

            wrap_m2f, flow_m2f = self.regis_module(
                moved_y_tran_recon, fixed_y_tran_recon)

            regis_recon_loss_m2f, regis_grad_loss_m2f, regis_mse_loss_m2f = self.get_regis_losses(
                wrap_m2f, fixed_y_tran_recon, flow_m2f)

            wrap_f2m, flow_f2m = self.regis_module(
                fixed_y_tran_recon, moved_y_tran_recon)
            
            regis_recon_loss_f2m, regis_grad_loss_f2m, regis_mse_loss_f2m = self.get_regis_losses(
                wrap_f2m, fixed_y_tran_recon, flow_f2m)

            regis_loss = regis_recon_loss_m2f + regis_recon_loss_f2m

            if self.lambda_ > 0:
                regis_loss += self.lambda_ * \
                    (regis_grad_loss_m2f + regis_grad_loss_f2m)

            if self.loss_regis_mse_COEFF > 0:
                regis_loss += self.loss_regis_mse_COEFF * \
                    (regis_mse_loss_m2f + regis_mse_loss_f2m)
        return regis_loss, batch

    def training_step_recon(self, batch, batch_idx):
        moved_x, moved_y_tran, moved_y, moved_mask, fixed_x, fixed_y_tran, fixed_y, fixed_mask = batch
        fixed_y_tran_recon = self.recon_module(fixed_y_tran)
        moved_y_tran_recon = self.recon_module(moved_y_tran)

        if self.is_optimize_regis:
            fixed_y_tran_recon_abs = f.pad(
                torch.sqrt(torch.sum(fixed_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])
            moved_y_tran_recon_abs = f.pad(
                torch.sqrt(torch.sum(moved_y_tran_recon ** 2, dim=1, keepdim=True)), [4, 4])

            _, flow_m2f = self.regis_module(
                moved_y_tran_recon_abs, fixed_y_tran_recon_abs)
            flow_m2f = flow_m2f[..., 4:-4] # padded before by [4,4], cut it here
            wrap_m2f = torch.cat([self.stn(tmp, flow_m2f) for tmp in [
                torch.unsqueeze(moved_y_tran_recon[:, 0], 1), torch.unsqueeze(
                    moved_y_tran_recon[:, 1], 1)
            ]], 1)

            wrap_y_m2f = fixed_mask * torch.view_as_real(torch.fft.fft2(
                torch.view_as_complex(wrap_m2f.permute([0, 2, 3, 1]).contiguous())))

            _, flow_f2m = self.regis_module(
                fixed_y_tran_recon_abs, moved_y_tran_recon_abs)
            flow_f2m = flow_f2m[..., 4:-4]

            wrap_f2m = torch.cat([self.stn(tmp, flow_f2m) for tmp in [
                torch.unsqueeze(fixed_y_tran_recon[:, 0], 1), torch.unsqueeze(
                    fixed_y_tran_recon[:, 1], 1)
            ]], 1)

            wrap_y_f2m = moved_mask * torch.view_as_real(torch.fft.fft2(
                torch.view_as_complex(wrap_f2m.permute([0, 2, 3, 1]).contiguous())))

        else:
            wrap_y_m2f = fixed_mask * torch.view_as_real(torch.fft.fft2(
                torch.view_as_complex(moved_y_tran_recon.permute([0, 2, 3, 1]).contiguous())))
            wrap_y_f2m = moved_mask * torch.view_as_real(torch.fft.fft2(
                torch.view_as_complex(fixed_y_tran_recon.permute([0, 2, 3, 1]).contiguous())))

        recon_loss_m2f = self.recon_loss_fn(wrap_y_m2f, fixed_y)
        recon_loss_f2m = self.recon_loss_fn(wrap_y_f2m, moved_y)

        recon_loss = recon_loss_f2m + recon_loss_m2f

        recon_loss_consensus_fixed = self.recon_loss_fn(
            fixed_mask * torch.view_as_real(torch.fft.fft2(torch.view_as_complex(fixed_y_tran_recon.permute([0, 2, 3, 1]).contiguous()))), fixed_y)
        recon_loss_consensus_moved = self.recon_loss_fn(
            moved_mask * torch.view_as_real(torch.fft.fft2(torch.view_as_complex(moved_y_tran_recon.permute([0, 2, 3, 1]).contiguous()))), moved_y)

        if self.loss_recon_consensus_COEFF > 0:
            recon_loss += self.loss_recon_consensus_COEFF * \
                (recon_loss_consensus_fixed + recon_loss_consensus_moved)

    def get_regis_losses(self, wrap, fixed, flow):
        regis_recon_loss = losses.sim_loss_fn(wrap, fixed)
        regis_grad_loss = losses.grad_loss_fn(flow)
        regis_mse_loss = losses.mse_loss_fn(wrap, fixed)
        return regis_recon_loss, regis_grad_loss, regis_mse_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"]
                                  for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"]
                                   for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"]
                           for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(
            predictions=preds, references=labels), prog_bar=True)
        return loss

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * \
            float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = self.optimizer(self.model.parameters())
        return [optimizer]
