import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from tqdm import tqdm
from pathlib import Path
import clip
from models.text_xrestormer import Text_XRestormer, XRestormer
from Datasets.datasets import PromptFusionDataset
from torchvision.utils import save_image
from utils.text_utils import truncate_text_batch
import random
from Metric_Python import (
    EN_function, MI_function, SF_function, SD_function, 
    AG_function, PSNR_function, VIF_function, SSIM_function, 
    MS_SSIM_function, CC_function, SCD_function, 
    Qabf_function, Nabf_function
)
import warnings
warnings.filterwarnings("ignore")

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def compute_metrics(vi_img, ir_img, f_img):
    """compute metrics"""
    return {
        "SSIM": SSIM_function(ir_img, vi_img, f_img),
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

def fspecial_gaussian(shape, sigma):
    """generate 2D Gaussian kernel"""
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def inference(model, vi, ir, text, window_size=128, stride=64, inference_type="full_image"):
    """inference function"""
    model.eval()
    B, C, H, W = vi.shape
    assert B == 1, "batch size must be 1"
    if isinstance(text, str):
        text = [text]
    text = truncate_text_batch(text)

    if inference_type == "sliding_window":
        fusion_result = torch.zeros_like(vi)
        weight_mask = torch.zeros_like(vi)
        
        gaussian_weights = torch.from_numpy(
            np.tile(fspecial_gaussian((window_size, window_size), window_size/4), (C,1,1))
        ).float().to(vi.device)
        with torch.no_grad():
            for h in range(0, H-window_size+1, stride):
                for w in range(0, W-window_size+1, stride):
                    vi_patch = vi[:, :, h:h+window_size, w:w+window_size]
                    ir_patch = ir[:, :, h:h+window_size, w:w+window_size]
                    
                    if isinstance(model, Text_XRestormer):
                        fusion_patch = model(vi_patch, ir_patch, text)
                    else:
                        fusion_patch = model(vi_patch, ir_patch)
                    
                    fusion_result[:, :, h:h+window_size, w:w+window_size] += fusion_patch * gaussian_weights
                    weight_mask[:, :, h:h+window_size, w:w+window_size] += gaussian_weights
            
            if H % stride != 0:
                h = H - window_size
                for w in range(0, W-window_size+1, stride):
                    vi_patch = vi[:, :, h:h+window_size, w:w+window_size]
                    ir_patch = ir[:, :, h:h+window_size, w:w+window_size]
                    
                    if isinstance(model, Text_XRestormer):
                        fusion_patch = model(vi_patch, ir_patch, text)
                    else:
                        fusion_patch = model(vi_patch, ir_patch)
                    
                    fusion_result[:, :, h:h+window_size, w:w+window_size] += fusion_patch * gaussian_weights
                    weight_mask[:, :, h:h+window_size, w:w+window_size] += gaussian_weights
            
            if W % stride != 0:
                w = W - window_size
                for h in range(0, H-window_size+1, stride):
                    vi_patch = vi[:, :, h:h+window_size, w:w+window_size]
                    ir_patch = ir[:, :, h:h+window_size, w:w+window_size]
                    
                    if isinstance(model, Text_XRestormer):
                        fusion_patch = model(vi_patch, ir_patch, text)
                    else:
                        fusion_patch = model(vi_patch, ir_patch)
                    
                    fusion_result[:, :, h:h+window_size, w:w+window_size] += fusion_patch * gaussian_weights
                    weight_mask[:, :, h:h+window_size, w:w+window_size] += gaussian_weights
        
        fusion_result = fusion_result / (weight_mask + 1e-6)
    elif inference_type == "full_image":
        # Note that full_image mode uses a lot of memory for large images.
        new_width = (W // 64) * 64
        new_height = (H // 64) * 64
        vi = torch.nn.functional.interpolate(vi, size=(new_height, new_width), mode='bilinear', align_corners=False)
        ir = torch.nn.functional.interpolate(ir, size=(new_height, new_width), mode='bilinear', align_corners=False)
        
        if isinstance(model, Text_XRestormer):
            fusion_result = model(vi, ir, text)
        else:
            fusion_result = model(vi, ir)
            
        # resize to original size
        fusion_result = torch.nn.functional.interpolate(fusion_result, size=(H, W), mode='bilinear', align_corners=False)

    return fusion_result

def main():
    # parser
    parser = argparse.ArgumentParser(description='DTPF Testing')
    parser.add_argument('-m', '--model_dir', required=True, help='model directory')
    parser.add_argument('-t', '--type', choices=['parent', 'distilled', 'all'], default='all', 
                        help='parent/distilled/all')
    parser.add_argument('-o', '--output', default='./save_images/test_ivf', help='folder to save images')
    parser.add_argument('--device', default='cuda:0', help='devices, recommend to use CUDA_VISIBLE_DEVICES')
    parser.add_argument('-i', '--inference_type', choices=['sliding_window', 'full_image'], default='sliding_window', 
                        help='sliding_window/full_image')
    parser.add_argument('--test_all', action='store_true', help='test on msrs、m3fd and road-scene')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    args = parser.parse_args()

    # random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # best model
    model_dir = Path(args.model_dir)
    yaml_file = model_dir / 'Train_text_xrestormer.yaml'
    print(f"using config: {yaml_file}")

    # config
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    parent_models = list(model_dir.glob('parent_epoch*quality*.pth'))
    distill_models = list(model_dir.glob('distill_epoch*quality*.pth'))

    best_parent = None
    best_parent_quality = -1
    for model in parent_models:
        quality = float(str(model).split('quality')[-1].split('.pth')[0])
        if quality > best_parent_quality:
            best_parent_quality = quality
            best_parent = model

    best_distill = None 
    best_distill_quality = -1
    for model in distill_models:
        quality = float(str(model).split('quality')[-1].split('.pth')[0])
        if quality > best_distill_quality:
            best_distill_quality = quality
            best_distill = model

    print(f"\nbest parent: {best_parent}, quality: {best_parent_quality}")
    print(f"best distill: {best_distill}, quality: {best_distill_quality}")

    ensure_dir(args.output)
    if args.type == 'all':
        ensure_dir(f"{args.output}/parent")
        ensure_dir(f"{args.output}/distilled")

    clip_model, _ = clip.load("ViT-B/32", device=args.device)
    for param in clip_model.parameters():
        param.requires_grad = False
    if not args.test_all:
        test_dataset = PromptFusionDataset(config, is_train=False, is_val=False, is_grayB=True, is_with_text=True)
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=config.get('num_workers', 4),
            shuffle=False,
            pin_memory=True
        )

        models_to_test = []
        if args.type in ['parent', 'all'] and best_parent:
            parent_model = Text_XRestormer(clip_model, **config['text_xrestormer_params']).to(args.device)
            checkpoint = torch.load(best_parent, map_location=args.device)
            parent_model.load_state_dict(checkpoint)
            models_to_test.append(('parent', parent_model))
        
        if args.type in ['distilled', 'all'] and best_distill:
            distilled_model = XRestormer(**config['distill_params']).to(args.device)
            checkpoint = torch.load(best_distill, map_location=args.device)
            distilled_model.load_state_dict(checkpoint)
            models_to_test.append(('distilled', distilled_model))

        metrics_dict = {model_type: [] for model_type, _ in models_to_test}
        
        for vi, ir, text, files in tqdm(test_loader, desc='Testing'):
            vi = vi.to(args.device)
            ir = ir.to(args.device)
            
            for model_type, model in models_to_test:
                fusion_result = inference(model, vi, ir, text, inference_type=args.inference_type)
                
                f_img = fusion_result[0].cpu().numpy()
                ir_img = ir[0].cpu().numpy()
                vi_img = vi[0].cpu().numpy()
                
                f_img = np.clip(f_img * 255, 0, 255).astype(np.int32)
                ir_img = np.clip(ir_img * 255, 0, 255).astype(np.int32)
                vi_img = np.clip(vi_img * 255, 0, 255).astype(np.int32)
                
                ir_img = np.repeat(ir_img, 3, axis=0)
                metrics = compute_metrics(vi_img, ir_img, f_img)
                metrics_dict[model_type].append(metrics)
                
                save_path = f"{args.output}/{config['test_dataset']}/{model_type}/{files[0]}"
                save_image(fusion_result[0], save_path)

        print("\n===== Testing Results =====")
        for model_type in metrics_dict:
            print(f"\n{model_type.upper()} Model Metrics:")
            avg_metrics = {}
            for metric in metrics_dict[model_type][0].keys():
                avg_metrics[metric] = np.mean([m[metric] for m in metrics_dict[model_type]])
                print(f"{metric}: {avg_metrics[metric]:.4f}")
    else:
        print("Inference all datasets")
        config['test_dataset'] = 'MSRS'
        config[config['test_dataset']]['data_dir']['test_dir']['range'] = [0, 1] # 全部testset
        test_dataset_MSRS = PromptFusionDataset(config, is_train=False, is_val=False, is_grayB=True, is_with_text=True)
        test_loader_MSRS = data.DataLoader(
            test_dataset_MSRS,
            batch_size=1,
            num_workers=config.get('num_workers', 4),
            shuffle=False,
            pin_memory=True
        )
        config['test_dataset'] = 'M3FD'
        config[config['test_dataset']]['data_dir']['test_dir']['range'] = [0, 1]
        test_dataset_M3FD = PromptFusionDataset(config, is_train=False, is_val=False, is_grayB=True, is_with_text=True)
        test_loader_M3FD = data.DataLoader(
            test_dataset_M3FD,
            batch_size=1,
            num_workers=config.get('num_workers', 4),
            shuffle=False,
            pin_memory=True
        )
        config['test_dataset'] = 'RS'
        config[config['test_dataset']]['data_dir']['test_dir']['range'] = [0, 1]
        test_dataset_RoadScene = PromptFusionDataset(config, is_train=False, is_val=False, is_grayB=True, is_with_text=True)
        test_loader_RoadScene = data.DataLoader(
            test_dataset_RoadScene,
            batch_size=1,
            num_workers=config.get('num_workers', 4),
            shuffle=False,
            pin_memory=True
        )

        
        models_to_test = []
        if args.type in ['parent', 'all'] and best_parent:
            parent_model = Text_XRestormer(clip_model, **config['text_xrestormer_params']).to(args.device)
            checkpoint = torch.load(best_parent, map_location=args.device)
            parent_model.load_state_dict(checkpoint)
            models_to_test.append(('parent', parent_model))
        
        if args.type in ['distilled', 'all'] and best_distill:
            distilled_model = XRestormer(**config['distill_params']).to(args.device)
            checkpoint = torch.load(best_distill, map_location=args.device)
            distilled_model.load_state_dict(checkpoint)
            models_to_test.append(('distilled', distilled_model))

        for dataset_name, test_loader in [
            ("MSRS", test_loader_MSRS),
            ("M3FD", test_loader_M3FD),
            ("RoadScene", test_loader_RoadScene)
        ]:
            print(f"\n===== Testing on {dataset_name} Dataset =====")
            metrics_dict = {model_type: [] for model_type, _ in models_to_test}
            
            dataset_output = os.path.join(args.output, dataset_name)
            if args.type == 'all':
                ensure_dir(f"{dataset_output}/parent")
                ensure_dir(f"{dataset_output}/distilled")
            
            for vi, ir, text, files in tqdm(test_loader, desc=f'Testing {dataset_name}'):
                vi = vi.to(args.device)
                ir = ir.to(args.device)
                
                for model_type, model in models_to_test:
                    fusion_result = inference(model, vi, ir, text, inference_type=args.inference_type)
                    
                    f_img = fusion_result[0].cpu().numpy()
                    ir_img = ir[0].cpu().numpy()
                    vi_img = vi[0].cpu().numpy()
                    
                    f_img = np.clip(f_img * 255, 0, 255).astype(np.int32)
                    ir_img = np.clip(ir_img * 255, 0, 255).astype(np.int32)
                    vi_img = np.clip(vi_img * 255, 0, 255).astype(np.int32)
                    
                    ir_img = np.repeat(ir_img, 3, axis=0)
                    metrics = compute_metrics(vi_img, ir_img, f_img)
                    metrics_dict[model_type].append(metrics)
                    
                    save_path = f"{dataset_output}/{model_type}/{files[0]}"
                    save_image(fusion_result[0], save_path)

            print(f"\n{dataset_name} Dataset Results:")
            for model_type in metrics_dict:
                print(f"\n{model_type.upper()} Model Metrics:")
                avg_metrics = {}
                for metric in metrics_dict[model_type][0].keys():
                    avg_metrics[metric] = np.mean([m[metric] for m in metrics_dict[model_type]])
                    print(f"{metric}: {avg_metrics[metric]:.4f}")

if __name__ == '__main__':
    main()
