# -*- coding: utf-8 -*-
"""
CIFAR-10-P Evaluation Script
Evaluates model stability on CIFAR-10-P perturbations
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from scipy.stats import rankdata

from models import *


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CIFAR-10-P stability')
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='Model name (e.g., ResNet18, ResNet50, vit)')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str,
                       default='/data/RobustART/datasets/imagenet-p/CIFAR-10-P',
                       help='Path to CIFAR-10-P dataset')
    parser.add_argument('--batch-size', type=int, default=25,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    return parser.parse_args()


def load_model(model_name, checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    model_name_lower = model_name.lower()
    
    # Map model names to functions
    if model_name_lower in ['resnet18', 'resnet_18']:
        from models.resnet import ResNet18
        model = ResNet18(num_classes=10)
    elif model_name_lower in ['resnet34', 'resnet_34']:
        from models.resnet import ResNet34
        model = ResNet34(num_classes=10)
    elif model_name_lower in ['resnet50', 'resnet_50']:
        from models.resnet import ResNet50
        model = ResNet50(num_classes=10)
    elif model_name_lower in ['resnet101', 'resnet_101']:
        from models.resnet import ResNet101
        model = ResNet101(num_classes=10)
    elif model_name_lower in ['resnet152', 'resnet_152']:
        from models.resnet import ResNet152
        model = ResNet152(num_classes=10)
    elif model_name_lower in ['vit', 'vision_transformer']:
        from models.vit import vit
        model = vit()  # vit() already has num_classes=10 hardcoded
    elif model_name_lower in ['mobilenet', 'mobilenet_v2']:
        # Try to import MobileNet if available
        try:
            from models.mobilenet import MobileNetV2
            model = MobileNetV2(num_classes=10)
        except ImportError:
            raise ValueError(f"MobileNet not available. Available models: ResNet18/34/50/101/152, ViT, MobileNetV3")
    elif model_name_lower in ['mobilenet_v3_large', 'mobilenetv3_large', 'mobilenet_v3-large']:
        from models.mobilenet_v3 import mobilenet_v3_large
        model = mobilenet_v3_large(num_classes=10)
    elif model_name_lower in ['mobilenet_v3_small', 'mobilenetv3_small', 'mobilenet_v3-small']:
        from models.mobilenet_v3 import mobilenet_v3_small
        model = mobilenet_v3_small(num_classes=10)
    else:
        # Try to get from models module directly
        try:
            model_fn = globals().get(model_name) or locals().get(model_name)
            if model_fn is None:
                # Try importing from models
                import models
                model_fn = getattr(models, model_name, None)
            if model_fn is None:
                raise ValueError(f"Unknown model: {model_name}")
            if callable(model_fn):
                model = model_fn(num_classes=10)
            else:
                model = model_fn
        except Exception as e:
            raise ValueError(f"Failed to load model {model_name}: {e}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'net' in checkpoint:
        state_dict = checkpoint['net']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (for DataParallel models)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def dist(sigma, mode='top5', num_classes=10):
    """Compute distance metric for ranking"""
    if mode == 'top5':
        identity = np.asarray(range(1, num_classes + 1))
        cum_sum_top5 = np.cumsum(np.asarray([0] + [1] * 5 + [0] * (num_classes - 1 - 5)))
        return np.sum(np.abs(cum_sum_top5[:5] - cum_sum_top5[sigma - 1][:5]))
    elif mode == 'zipf':
        identity = np.asarray(range(1, num_classes + 1))
        recip = 1. / identity
        return np.sum(np.abs(recip - recip[sigma - 1]) * recip)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def ranking_dist(ranks, noise_perturbation=False, mode='top5', num_classes=10):
    """Compute ranking distance"""
    result = 0
    step_size = 1 if noise_perturbation else 1
    
    for vid_ranks in ranks:
        result_for_vid = []
        
        for i in range(step_size):
            perm1 = vid_ranks[i]
            perm1_inv = np.argsort(perm1)
            
            for rank in vid_ranks[i::step_size][1:]:
                perm2 = rank
                result_for_vid.append(dist(perm2[perm1_inv], mode, num_classes))
                if not noise_perturbation:
                    perm1 = perm2
                    perm1_inv = np.argsort(perm1)
        
        if len(result_for_vid) > 0:
            result += np.mean(result_for_vid) / len(ranks)
    
    return result


def flip_prob(predictions, noise_perturbation=False):
    """Compute flip probability"""
    result = 0
    step_size = 1 if noise_perturbation else 1
    
    for vid_preds in predictions:
        result_for_vid = []
        
        for i in range(step_size):
            prev_pred = vid_preds[i]
            
            for pred in vid_preds[i::step_size][1:]:
                result_for_vid.append(int(prev_pred != pred))
                if not noise_perturbation:
                    prev_pred = pred
        
        if len(result_for_vid) > 0:
            result += np.mean(result_for_vid) / len(predictions)
    
    return result


def evaluate_perturbation(model, data, batch_size, device='cuda'):
    """Evaluate model on a perturbation sequence"""
    # data shape: (num_videos, num_frames, 32, 32, 3)
    num_videos = data.shape[0]
    num_frames = data.shape[1]
    
    predictions = []
    ranks = []
    
    with torch.no_grad():
        # Process in batches
        for i in tqdm(range(0, num_videos, batch_size), desc='Processing videos'):
            batch_videos = data[i:i+batch_size]  # (batch_size, num_frames, 32, 32, 3)
            batch_size_actual = batch_videos.shape[0]
            
            # Reshape to (batch_size * num_frames, 32, 32, 3)
            batch_data = batch_videos.reshape(-1, 32, 32, 3)
            
            # Convert to tensor and normalize
            # Input shape: (batch * num_frames, 32, 32, 3) -> (batch * num_frames, 3, 32, 32)
            batch_data = torch.from_numpy(batch_data).float().to(device)
            batch_data = batch_data / 255.0  # Normalize to [0, 1]
            batch_data = batch_data.permute(0, 3, 1, 2)  # (batch * num_frames, 3, 32, 32)
            
            # Manual normalization (CIFAR-10 mean and std)
            mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)
            batch_data = (batch_data - mean) / std
            
            # Forward pass
            outputs = model(batch_data)  # (batch_size * num_frames, num_classes)
            
            # Reshape back to (batch_size, num_frames, num_classes)
            outputs = outputs.view(batch_size_actual, num_frames, -1)
            
            # Get predictions and rankings for each video
            for vid_idx in range(batch_size_actual):
                vid_output = outputs[vid_idx]  # (num_frames, num_classes)
                vid_preds = vid_output.argmax(dim=1).cpu().numpy()
                predictions.append(vid_preds)
                
                # Compute rankings
                vid_ranks = [np.uint16(rankdata(-frame, method='ordinal')) 
                            for frame in vid_output.cpu().numpy()]
                ranks.append(vid_ranks)
    
    return predictions, ranks


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    
    # Load model
    print(f'Loading model: {args.model}')
    model = load_model(args.model, args.checkpoint, device)
    print('Model loaded successfully')
    
    # CIFAR-10-P perturbations
    perturbations = [
        'gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur',
        'brightness', 'translate', 'rotate', 'tilt', 'scale', 'snow'
    ]
    
    # Extra perturbations (if available)
    extra_perturbations = [
        'gaussian_blur', 'spatter', 'speckle_noise', 'shear'
    ]
    
    flip_list = []
    zipf_list = []
    results = {}
    
    print('\n' + '='*60)
    print('Evaluating on CIFAR-10-P')
    print('='*60)
    
    # Evaluate on each perturbation
    for p in perturbations:
        p_path = os.path.join(args.data_dir, f'{p}.npy')
        
        if not os.path.exists(p_path):
            print(f'Warning: {p}.npy not found, skipping...')
            continue
        
        print(f'\nEvaluating {p}...')
        
        # Load perturbation data
        # CIFAR-P format: (num_frames, num_videos, 32, 32, 3) -> transpose to (num_videos, num_frames, 32, 32, 3)
        data = np.load(p_path).astype(np.float32)
        
        # Transpose to (num_videos, num_frames, 32, 32, 3)
        # Original format is typically (num_frames, num_videos, 32, 32, 3)
        if len(data.shape) == 5:
            # Check if first dimension is num_frames (usually 10) or num_videos (10000)
            if data.shape[0] < 100:  # Likely num_frames
                # Transpose: (num_frames, num_videos, 32, 32, 3) -> (num_videos, num_frames, 32, 32, 3)
                data = data.transpose((1, 0, 2, 3, 4))
            # If shape[0] == 10000, already in correct format
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}, expected 5D array")
        
        # Evaluate
        predictions, ranks = evaluate_perturbation(
            model, data, args.batch_size, device
        )
        
        # Compute metrics
        is_noise = 'noise' in p
        current_flip = flip_prob(predictions, noise_perturbation=is_noise)
        current_zipf = ranking_dist(ranks, noise_perturbation=is_noise, mode='zipf', num_classes=10)
        
        flip_list.append(current_flip)
        zipf_list.append(current_zipf)
        results[p] = {
            'flip_prob': current_flip,
            'zipf_dist': current_zipf
        }
        
        print(f'  Flipping Probability: {current_flip:.5f}')
        print(f'  Zipf Distance: {current_zipf:.5f}')
    
    # Evaluate extra perturbations if available
    extra_dir = os.path.join(args.data_dir, 'extra')
    if os.path.exists(extra_dir):
        for p in extra_perturbations:
            p_path = os.path.join(extra_dir, f'{p}.npy')
            
            if not os.path.exists(p_path):
                continue
            
            print(f'\nEvaluating {p} (extra)...')
            
            data = np.load(p_path).astype(np.float32)
            
            # Transpose if needed: (num_frames, num_videos, 32, 32, 3) -> (num_videos, num_frames, 32, 32, 3)
            if len(data.shape) == 5:
                if data.shape[0] < 100:  # Likely num_frames
                    data = data.transpose((1, 0, 2, 3, 4))
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}, expected 5D array")
            
            predictions, ranks = evaluate_perturbation(
                model, data, args.batch_size, device
            )
            
            is_noise = 'noise' in p
            current_flip = flip_prob(predictions, noise_perturbation=is_noise)
            current_zipf = ranking_dist(ranks, noise_perturbation=is_noise, mode='zipf', num_classes=10)
            
            flip_list.append(current_flip)
            zipf_list.append(current_zipf)
            results[p] = {
                'flip_prob': current_flip,
                'zipf_dist': current_zipf
            }
            
            print(f'  Flipping Probability: {current_flip:.5f}')
            print(f'  Zipf Distance: {current_zipf:.5f}')
    
    # Print summary
    print('\n' + '='*60)
    print('Summary Results')
    print('='*60)
    for p, result in results.items():
        print(f'{p:20s}: Flip Prob = {result["flip_prob"]:.5f}, Zipf Dist = {result["zipf_dist"]:.5f}')
    
    if flip_list:
        mean_flip = np.mean(flip_list)
        print(f'\nMean Flipping Probability: {mean_flip:.5f}')
    
    if zipf_list:
        mean_zipf = np.mean(zipf_list)
        print(f'Mean Zipf Distance: {mean_zipf:.5f}')
    
    print('='*60)


if __name__ == '__main__':
    main()

