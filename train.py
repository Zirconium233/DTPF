import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"]='0' # default use one GPU to prevent bugs
import pytorch_lightning as pl
import argparse
import numpy as np
from Datasets.datasets import *
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torchvision
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything
from torch.utils import data
import json
import shutil
import os
import copy
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from easydict import EasyDict
from models.models import MODELS
from models.models import LOSSES
from utils.optimizer import Lion
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
import collections
from utils.ema import EMA as EMACallback
import clip
import traceback
from utils.text_utils import text_to_task_list
from Metric_Python import EN_function, MI_function, SF_function, SD_function, AG_function, PSNR_function, VIF_function, SSIM_function, MS_SSIM_function, CC_function, SCD_function, Qabf_function, Nabf_function
from utils.text_utils import truncate_text_batch
import warnings
warnings.filterwarnings("ignore")
# create floder
def ensure_dir(file_path):
    # directory = os.path.dirname(file_path)
    if not os.path.exists(file_path):
        os.makedirs(file_path)


# select dataset type
__dataset__ = {
    "MSRS": PromptFusionDataset,
    "M3FD": PromptFusionDataset,
    "RS": PromptFusionDataset,
    "CT_MRI": PromptFusionDataset,
    "PET_MRI": PromptFusionDataset,
    "SPECT_MRI": PromptFusionDataset,
}

class CoolSystem(pl.LightningModule):
    def __init__(self):
        """init parameters"""
        super(CoolSystem, self).__init__()
        # train datasets
        self.train_datasets = __dataset__[config["train_dataset"]](config, is_train=True, is_val=False, is_grayB=True, is_with_text=True)
        self.train_batchsize = config["train_batch_size"]
        # val datasets
        self.validation_datasets = __dataset__[config["val_dataset"]](config, is_train=False, is_val=True, is_label=False, is_grayB=True, is_with_text=True)
        self.test_datasets = __dataset__[config["test_dataset"]](config, is_train=False, is_val=False, is_label=False, is_grayB=True, is_with_text=True)
        self.val_batchsize = config["val_batch_size"]
        self.num_workers = config["num_workers"]
        self.save_path = os.path.join(config["save_path"], config["tags"])
        self.drop_last = config["drop_last"]
        self.parent_models = []
        self.distill_models = []
        self.best_parent_loaded = False
        self.save_top_k = config["trainer"]["save_top_k"]
        # self.eval_freq = config["trainer"]["test_freq"]
        ensure_dir(self.save_path)
        # set mode stype
        model_clip, preprocess = clip.load("ViT-B/32", device="cuda:0")
        # freeze model_clip
        for param in model_clip.parameters():
            param.requires_grad = False
        self.text_xrestormer = MODELS[config["net_parent"]](model_clip, **config["text_xrestormer_params"])
        self.distilled_xrestormer = MODELS[config["net_distill"]](**config["distill_params"])

        # loss
        self.loss = LOSSES[config["loss"]](**config["loss_weights"])
        if "loss_configs" in config and "task_configs" in config["loss_configs"]:
            self.loss.task_configs = config["loss_configs"]["task_configs"]
            print("task_configs has been loaded")

        print(PATH)
        # print model summary.txt
        import sys
        original_stdout = sys.stdout
        with open(PATH + "/" + "model_summary.txt", 'w+') as f:
            sys.stdout = f
            print(f'\n{self.text_xrestormer}\n')
            print(f'\n{self.distilled_xrestormer}\n')
            sys.stdout = original_stdout
            # shutil.copy(f'./models/{config["model"]}.py',PATH+"/"+"model.py")
        self.automatic_optimization = False

    def train_dataloader(self):
        train_loader = data.DataLoader(
            self.train_datasets,
            batch_size=self.train_batchsize,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=self.drop_last,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = data.DataLoader(
            self.validation_datasets,
            batch_size=self.val_batchsize,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=self.drop_last,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = data.DataLoader(
            self.test_datasets,
            batch_size=self.val_batchsize,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=self.drop_last,
        )
        return test_loader

    def configure_optimizers(self):
        # Setting up optimizer.
        self.initlr = config["optimizer"]["args"]["lr"]
        self.weight_decay = config["optimizer"]["args"]["weight_decay"]
        self.momentum = config["optimizer"]["args"]["momentum"]
        
        if config["optimizer"]["type"] == "ADAMW":
            optimizer_0 = optim.AdamW(
                self.text_xrestormer.parameters(),
                lr=self.initlr,
                weight_decay=self.weight_decay,
                betas=[0.9, 0.999]
            )
            optimizer_1 = optim.AdamW(
                self.distilled_xrestormer.parameters(),
                lr=self.initlr,
                weight_decay=self.weight_decay,
                betas=[0.9, 0.999]
            )
        elif config["optimizer"]["type"] == "SGD":
            optimizer_0 = optim.SGD(
                self.text_xrestormer.parameters(),
                lr=self.initlr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
            optimizer_1 = optim.SGD(
                self.distilled_xrestormer.parameters(),
                lr=self.initlr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif config["optimizer"]["type"] == "ADAM":
            optimizer_0 = optim.Adam(
                self.text_xrestormer.parameters(),
                lr=self.initlr,
                weight_decay=self.weight_decay,
                betas=[0.9, 0.999]
            )
            optimizer_1 = optim.Adam(
                self.distilled_xrestormer.parameters(),
                lr=self.initlr,
                weight_decay=self.weight_decay,
                betas=[0.9, 0.999]
            )

        elif config["optimizer"]["type"] == "ADAMW":
            optimizer_0 = optim.AdamW(
                self.text_xrestormer.parameters(),
                lr=self.initlr,
                weight_decay=self.weight_decay,
                betas=[0.9, 0.999]
            )
            optimizer_1 = optim.AdamW(
                self.distilled_xrestormer.parameters(),
                lr=self.initlr,
                weight_decay=self.weight_decay,
                betas=[0.9, 0.999]
            )
        elif config["optimizer"]["type"] == "Lion":
            optimizer_0 = Lion(filter(lambda p: p.requires_grad, self.text_xrestormer.parameters()),
                             lr=self.initlr,
                             betas=[0.9, 0.99],
                             weight_decay=0)
            optimizer_1 = Lion(filter(lambda p: p.requires_grad, self.distilled_xrestormer.parameters()),
                             lr=self.initlr,
                             betas=[0.9, 0.99],
                             weight_decay=0)

        else:
            exit("Undefined optimizer type")
        if config["optimizer"]["parent_scheduler"]['type'] == "CosineAnnealingLR":
        # two schedulers
            parent_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_0,
                T_max=config["optimizer"]["parent_scheduler"]["args"]["T_max"],
                eta_min=config["optimizer"]["parent_scheduler"]["args"]["eta_min"]
            )
        else:
            exit("Undefined scheduler type")
        if config["optimizer"]["distill_scheduler"]['type'] == "CosineAnnealingLR":
            distill_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_1,
                T_max=config["optimizer"]["distill_scheduler"]["args"]["T_max"],
                eta_min=config["optimizer"]["distill_scheduler"]["args"]["eta_min"]
            )
        else:
            exit("Undefined scheduler type")
        
        return [
            {
                "optimizer": optimizer_0,
                "lr_scheduler": {
                    "scheduler": parent_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "train_loss"
                }
            },
            {
                "optimizer": optimizer_1,
                "lr_scheduler": {
                    "scheduler": distill_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "train_loss"
                }
            }
        ]

    def on_train_epoch_start(self):
        # only load parent model in the first epoch in main process
        if self.trainer.is_global_zero:
            if self.current_epoch == config["trainer"]["parent_epochs"] and not self.best_parent_loaded:
                if len(self.parent_models) > 0:
                    best_parent_path = self.parent_models[0][1]
                    state_dict = torch.load(best_parent_path)
                    self.text_xrestormer.load_state_dict(state_dict)
                    self.best_parent_loaded = True
                    print(f"\nLoaded best parent model from {best_parent_path}\n")
                else:
                    print("\nNo parent model found\n")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def training_step(self, data):
        """optimize the training"""
        opt = self.optimizers()
        stage = "parent" if self.current_epoch < config["trainer"]["parent_epochs"] else "distill"
        opt_idx = 0 if stage == "parent" else 1
        opt[opt_idx].zero_grad()
        train_stage = 'parent' if stage == "parent" else 'distill'
        device = next(self.text_xrestormer.parameters()).device

        """training step"""
        # for ivf dataset they are vi, ir, text;but for medical datasets, they are CT, MRI, text or PET, MRI, text
        vi, ir, text = data 
        if isinstance(text, str):
            text = [text]
        text = truncate_text_batch(text)
        
        if train_stage == "parent":
            fusion = self.text_xrestormer(vi, ir, text)
            total_loss, loss_dict = self.loss(
                image_A=vi,
                image_B=ir, 
                image_fused=fusion,
                stage='parent',
                text=text,
            )
        
        else:  # distill stage
            with torch.no_grad():
                fusion_parent, teacher_feats = self.text_xrestormer(vi, ir, text, return_features=True)
            fusion_student, student_feats = self.distilled_xrestormer(vi, ir, return_features=True)
            
            total_loss, loss_dict = self.loss(
                image_A=vi,
                image_B=ir,
                image_fused=fusion_parent,
                image_fused_distilled=fusion_student,
                stage='distill',
                text=text,
                teacher_feats=teacher_feats,
                student_feats=student_feats
            )

        self.log('train/train_loss', total_loss, prog_bar=True)
        for loss_name, loss_value in loss_dict.items():
            self.log(f'train/train_{loss_name}', loss_value)

        self.manual_backward(total_loss)
        # nn.utils.clip_grad_norm_(self.dtpf_trainer.parameters(), config["trainer"]["clip_grad_norm"])
        opt[opt_idx].step()

        schedulers = self.lr_schedulers()
        if train_stage == 'parent':
            scheduler = schedulers[0]  # parent_scheduler
        else:
            scheduler = schedulers[1]  # distill_scheduler
        scheduler.step()
        # if torch.distributed.is_initialized():
        #     torch.distributed.barrier()
        return {'loss': total_loss}

    # def on_train_epoch_end(self):
    #     if self.current_epoch % self.eval_freq == 0 and self.current_epoch > 0:
    #         self.validation_epoch()
    def _val_or_test_step(self, data, batch_idx, mode="val"):
        ensure_dir(os.path.join(self.save_path, mode))
        vi, ir, text, files = data
        B, C, H, W = vi.shape
        assert B == 1, "val batch size must be 1"
        
        window_size = 128  # patch大小
        stride = 64        # 步长
        stage = "parent" if self.current_epoch < config["trainer"]["parent_epochs"] else "distill"
        stage = "distill" if mode == "test" else stage
        fusion_result = torch.zeros_like(vi)
        weight_mask = torch.zeros_like(vi)
        fusion_result_parent = torch.zeros_like(vi) if mode == "test" else None
        gaussian_weights = torch.from_numpy(
            np.tile(self.fspecial_gaussian((window_size, window_size), window_size/4), (C,1,1))
        ).float().to(vi.device)
        
        with torch.no_grad():
            if isinstance(text, str):
                text = [text]
            text = truncate_text_batch(text)
            
            for h in range(0, H-window_size+1, stride):
                for w in range(0, W-window_size+1, stride):
                    h, w = int(h), int(w)
                    vi_patch = vi[:, :, h:h+window_size, w:w+window_size]
                    ir_patch = ir[:, :, h:h+window_size, w:w+window_size]
                    
                    if stage == "parent":
                        fusion_patch = self.text_xrestormer(vi_patch, ir_patch, text)
                        fusion_result[:, :, h:h+window_size, w:w+window_size] += fusion_patch * gaussian_weights
                    else:
                        fusion_student = self.distilled_xrestormer(vi_patch, ir_patch)
                        fusion_result[:, :, h:h+window_size, w:w+window_size] += fusion_student * gaussian_weights
                        if mode == "test":
                            fusion_parent = self.text_xrestormer(vi_patch, ir_patch, text)
                            fusion_result_parent[:, :, h:h+window_size, w:w+window_size] += fusion_parent * gaussian_weights
                    
                    weight_mask[:, :, h:h+window_size, w:w+window_size] += gaussian_weights
            
            if H % stride != 0:
                h = H - window_size
                for w in range(0, W-window_size+1, stride):
                    vi_patch = vi[:, :, h:h+window_size, w:w+window_size]
                    ir_patch = ir[:, :, h:h+window_size, w:w+window_size]
                    
                    if stage == "parent":
                        fusion_patch = self.text_xrestormer(vi_patch, ir_patch, text)
                        fusion_result[:, :, h:h+window_size, w:w+window_size] += fusion_patch * gaussian_weights
                    else:
                        fusion_student = self.distilled_xrestormer(vi_patch, ir_patch)
                        fusion_result[:, :, h:h+window_size, w:w+window_size] += fusion_student * gaussian_weights
                        if mode == "test":
                            fusion_parent = self.text_xrestormer(vi_patch, ir_patch, text)
                            fusion_result_parent[:, :, h:h+window_size, w:w+window_size] += fusion_parent * gaussian_weights
                    
                    weight_mask[:, :, h:h+window_size, w:w+window_size] += gaussian_weights
            
            
        fusion_result = fusion_result / (weight_mask + 1e-6)
        if mode == "test":
            fusion_result_parent = fusion_result_parent / (weight_mask + 1e-6)
        
        f_img = fusion_result[0].cpu().numpy()
        ir_img = ir[0].cpu().numpy()
        vi_img = vi[0].cpu().numpy()
        
        f_img = np.clip(f_img * 255, 0, 255).astype(np.int32)
        ir_img = np.clip(ir_img * 255, 0, 255).astype(np.int32)
        vi_img = np.clip(vi_img * 255, 0, 255).astype(np.int32)
        
        ir_img = np.repeat(ir_img, 3, axis=0)
        metrics = self.compute_metrics(vi_img, ir_img, f_img)
        quality = self.calculate_quality(metrics)
        prefix = "distilled" if stage == "distill" else "parent"
        self.log(f'{mode}/{prefix}/quality', quality, sync_dist=True, prog_bar=True)
        for metric_name, metric_value in metrics.items():
            self.log(f'{mode}/{prefix}/{metric_name}', metric_value, sync_dist=True)
        save_image(fusion_result[0], os.path.join(self.save_path, f"{mode}/{prefix}_epoch{self.current_epoch}_{files[0]}"))
        metrics['quality'] = quality
        if mode == "test":
            fusion_result_parent = fusion_result_parent[0].cpu().numpy()
            fusion_result_parent = np.clip(fusion_result_parent * 255, 0, 255).astype(np.int32)
            metrics_parent = self.compute_metrics(vi_img, ir_img, fusion_result_parent)
            quality_parent = self.calculate_quality(metrics_parent)
            self.log(f'{mode}/parent/quality', quality_parent, sync_dist=True, prog_bar=True)
            for metric_name, metric_value in metrics_parent.items():
                self.log(f'{mode}/parent/{metric_name}', metric_value, sync_dist=True)
            metrics_parent['quality'] = quality_parent
            metrics_all = {}
            for metric_name, metric_value in metrics.items():
                metrics_all[f'{prefix}_{metric_name}'] = metric_value
            for metric_name, metric_value in metrics_parent.items():
                metrics_all[f'parent_{metric_name}'] = metric_value
            metrics = metrics_all
        return metrics

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            stage = "parent" if self.current_epoch < config["trainer"]["parent_epochs"] else "distill"
            metrics = self.trainer.callback_metrics
            prefix = "distilled" if stage == "distill" else "parent"
            quality = metrics[f'val/{prefix}/quality'].item()
            
            if stage == "parent":
                save_path = f"{PATH}/parent_epoch{self.current_epoch}_quality{quality:.4f}.pth"
                torch.save(self.text_xrestormer.state_dict(), save_path)
                best_path = self.save_model_with_top_k(quality, save_path, is_parent=True)
                print(f"\nNow best parent model is {best_path}\n")
            else:
                save_path = f"{PATH}/distill_epoch{self.current_epoch}_quality{quality:.4f}.pth" 
                torch.save(self.distilled_xrestormer.state_dict(), save_path)
                best_path = self.save_model_with_top_k(quality, save_path, is_parent=False)
                print(f"\nNow best distilled model is {best_path}\n")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def validation_step(self, data, batch_idx):
        metrics = self._val_or_test_step(data, batch_idx, mode="val")
        return metrics

    def on_test_epoch_start(self):
        if self.trainer.is_global_zero:
            if len(self.parent_models) > 0:
                best_parent_path = self.parent_models[0][1]
                state_dict = torch.load(best_parent_path)
                self.text_xrestormer.load_state_dict(state_dict)
                print(f"\nLoaded best parent model from {best_parent_path}\n")
            else:
                print("\nNo parent model found\n")
            
            if len(self.distill_models) > 0:
                best_distill_path = self.distill_models[0][1]
                state_dict = torch.load(best_distill_path)
                self.distilled_xrestormer.load_state_dict(state_dict)
                print(f"\nLoaded best distilled model from {best_distill_path}\n")
            else:
                print("\nNo distilled model found\n")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def test_step(self, data, batch_idx):
        metrics = self._val_or_test_step(data, batch_idx, mode="test")     
        return metrics

    def calculate_quality(self, metrics):
        """compute quality"""
        ssim = metrics['SSIM']
        vim = metrics['VIF'] 
        sd = metrics['SD']
        qabf = metrics['QABF']
        en = metrics['EN']
        return (ssim + vim + (sd / 45) + (qabf / 0.70) + (en / 6.75)) / 5

    def save_model_with_top_k(self, quality, path, is_parent=True):
        """top k"""
        model_list = self.parent_models if is_parent else self.distill_models
        
        model_list.append((quality, path))
        model_list.sort(key=lambda x: x[0], reverse=True)
        
        if len(model_list) > self.save_top_k:
            _, old_path = model_list.pop()
            if os.path.exists(old_path):
                os.remove(old_path)
                
        return model_list[0][1]
    
    def compute_metrics(self, vi_img, ir_img, f_img):
        return {
            "SSIM": SSIM_function(vi_img, ir_img, f_img),
            "EN": EN_function(f_img),
            "MI": MI_function(vi_img, ir_img, f_img),
            "SF": SF_function(f_img),
            "SD": SD_function(f_img),
            "AG": AG_function(f_img),
            "CC": CC_function(vi_img, ir_img, f_img),
            "SCD": SCD_function(vi_img, ir_img, f_img),
            "PSNR": PSNR_function(vi_img, ir_img, f_img),
            "VIF": VIF_function(vi_img, ir_img, f_img),
            "NABF": Nabf_function(vi_img, ir_img, f_img),
            "QABF": Qabf_function(vi_img, ir_img, f_img),
        }

    def fspecial_gaussian(self, shape, sigma):
        """
        Generate 2D Gaussian weights
        """
        m, n = [(ss-1.)/2. for ss in shape]
        y, x = np.ogrid[-m:m+1, -n:n+1]
        h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
        h[h < np.finfo(h.dtype).eps*h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='./configs/Train_text_xrestormer.yaml', type=str,
                        help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default='-1', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-v', '--val', default=False, type=bool,
                        help='Valdation')
    parser.add_argument('--val_path',
                        default='',
                        type=str, help='Path to the val path')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    global args
    args = parser.parse_args()
    # set resmue
    global config
    # config = yaml.safe_load(open(args.config))
    # select config file type
    if args.config.endswith('.json'):
        config = json.load(open(args.config))
    elif args.config.endswith('.yaml'):
        config = yaml.safe_load(open(args.config))
    else:
        raise ValueError("Unsupported config file format. Please use .json or .yaml.")

    # Set seeds.
    seed = 3407  # Global seed set to 3407
    seed_everything(seed)

    output_dir = './TensorBoardLogs'
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=config['name'] + "_" + config["train_dataset"],
        default_hp_metric=False
    )
    
    # Setting up path
    global PATH
    PATH = "./" + config["experim_name"] + "/" + config["train_dataset"] + "/" + str(config["tags"])
    ensure_dir(PATH + "/")
    shutil.copy2(args.config, PATH)

    # init pytorch-litening
    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True, gradient_as_bucket_view=True)
    model = CoolSystem()

    # set checkpoint mode and init ModelCheckpointHook
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val/quality',
    #     dirpath=PATH,
    #     filename='best_model-epoch:{epoch:02d}-ssim:{val/quality:.4f}',
    #     auto_insert_metric_name=False,
    #     every_n_epochs=config["trainer"]["test_freq"],
    #     save_on_train_epoch_end=True,
    #     save_top_k=config["trainer"]["save_top_k"],
    #     mode="max"
    # )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    ema_callback = EMACallback(decay=0.995, every_n_steps=1)
    callback = [lr_monitor_callback, ema_callback] if config["trainer"]["ema"] else [lr_monitor_callback]

    trainer = pl.Trainer(
        strategy=ddp,
        max_epochs=config["trainer"]["total_epochs"],
        accelerator='gpu', devices=args.device,
        logger=logger,
        # amp_backend="apex",
        # amp_level='01',
        # accelerator='ddp',
        # precision='16-mixed',
        callbacks=callback,
        check_val_every_n_epoch=config["trainer"]["test_freq"],
        log_every_n_steps=10,
        fast_dev_run=args.debug,
        # detect_anomaly=True,
    )

    if args.val == True:
        trainer.validate(model, ckpt_path=args.val_path)
    else:
        # resume from ckpt pytorch lightening
        # trainer.fit(model,ckpt_path=resume_checkpoint_path)
        # resume from pth pytorch
        try:
            trainer.fit(model)
        except Exception as e:
            print(f"An error occurred: {e}")
            print(traceback.format_exc())
            if trainer.current_epoch > 100:
                trainer.save_checkpoint(PATH + "/lastest_lightning_model.pth") # for resume training
            exit()
        trainer.test(model)


if __name__ == '__main__':
    print('-----------------------------------------train.py trainning-----------------------------------------')
    main()
