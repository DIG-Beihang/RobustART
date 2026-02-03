# -*- coding: utf-8 -*-
"""
CIFAR-10-C Evaluation Script
Evaluates model robustness on CIFAR-10-C corruptions
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from models import *


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CIFAR-10-C robustness')
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='Model name (e.g., ResNet18, ResNet50, vit)')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, 
                       default='/data/RobustART/datasets/imagenet-c/CIFAR-10-C',
                       help='Path to CIFAR-10-C dataset')
    parser.add_argument('--batch-size', type=int, default=100,
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


def evaluate_corruption(model, data, labels, batch_size, device='cuda'):
    """Evaluate model on a corruption"""
    num_samples = len(data)
    num_correct = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc='Evaluating'):
            batch_data = data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # Convert to tensor and normalize
            # Input shape: (batch, 32, 32, 3) -> (batch, 3, 32, 32)
            batch_data = torch.from_numpy(batch_data).float().to(device)
            batch_data = batch_data / 255.0  # Normalize to [0, 1]
            batch_data = batch_data.permute(0, 3, 1, 2)  # (batch, 3, 32, 32)
            
            # Manual normalization (CIFAR-10 mean and std)
            mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)
            batch_data = (batch_data - mean) / std
            
            batch_labels = torch.from_numpy(batch_labels).long().to(device)
            
            # Forward pass
            outputs = model(batch_data)
            preds = outputs.argmax(dim=1)
            
            num_correct += (preds == batch_labels).sum().item()
    
    accuracy = num_correct / num_samples
    error_rate = 1.0 - accuracy
    return error_rate, accuracy


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    
    # Load model
    print(f'Loading model: {args.model}')
    model = load_model(args.model, args.checkpoint, device)
    print('Model loaded successfully')
    
    # Load labels
    labels_path = os.path.join(args.data_dir, 'labels.npy')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    labels = np.load(labels_path)
    print(f'Loaded labels: {len(labels)} samples')
    
    # CIFAR-10-C corruptions
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    # Extra corruptions (if available)
    extra_corruptions = [
        'gaussian_blur', 'saturate', 'spatter', 'speckle_noise'
    ]
    
    error_rates = []
    results = {}
    
    print('\n' + '='*60)
    print('Evaluating on CIFAR-10-C')
    print('='*60)
    
    # Evaluate on each corruption
    for corruption in corruptions:
        corruption_path = os.path.join(args.data_dir, f'{corruption}.npy')
        
        if not os.path.exists(corruption_path):
            print(f'Warning: {corruption}.npy not found, skipping...')
            continue
        
        print(f'\nEvaluating {corruption}...')
        data = np.load(corruption_path)  # Shape: (50000, 32, 32, 3) for 5 severities
        
        # CIFAR-10-C has 5 severity levels, each with 10000 samples
        # Data shape: (50000, 32, 32, 3)
        severities = [1, 2, 3, 4, 5]
        severity_errors = []
        
        for severity in severities:
            start_idx = (severity - 1) * 10000
            end_idx = severity * 10000
            severity_data = data[start_idx:end_idx]
            severity_labels = labels[start_idx:end_idx]
            
            error_rate, accuracy = evaluate_corruption(
                model, severity_data, severity_labels, args.batch_size, device
            )
            severity_errors.append(error_rate)
            print(f'  Severity {severity}: Error Rate = {error_rate*100:.2f}%')
        
        # Average error rate across severities
        avg_error = np.mean(severity_errors)
        error_rates.append(avg_error)
        results[corruption] = {
            'severity_errors': severity_errors,
            'avg_error': avg_error
        }
        print(f'  Average Error Rate: {avg_error*100:.2f}%')
    
    # Evaluate extra corruptions if available
    extra_dir = os.path.join(args.data_dir, 'extra')
    if os.path.exists(extra_dir):
        for corruption in extra_corruptions:
            corruption_path = os.path.join(extra_dir, f'{corruption}.npy')
            
            if not os.path.exists(corruption_path):
                continue
            
            print(f'\nEvaluating {corruption} (extra)...')
            data = np.load(corruption_path)
            
            severities = [1, 2, 3, 4, 5]
            severity_errors = []
            
            for severity in severities:
                start_idx = (severity - 1) * 10000
                end_idx = severity * 10000
                severity_data = data[start_idx:end_idx]
                severity_labels = labels[start_idx:end_idx]
                
                error_rate, accuracy = evaluate_corruption(
                    model, severity_data, severity_labels, args.batch_size, device
                )
                severity_errors.append(error_rate)
                print(f'  Severity {severity}: Error Rate = {error_rate*100:.2f}%')
            
            avg_error = np.mean(severity_errors)
            error_rates.append(avg_error)
            results[corruption] = {
                'severity_errors': severity_errors,
                'avg_error': avg_error
            }
            print(f'  Average Error Rate: {avg_error*100:.2f}%')
    
    # Print summary
    print('\n' + '='*60)
    print('Summary Results')
    print('='*60)
    for corruption, result in results.items():
        print(f'{corruption:20s}: {result["avg_error"]*100:6.2f}%')
    
    mean_error = np.mean(error_rates)
    print(f'\nMean Corruption Error (mCE): {mean_error*100:.2f}%')
    print('='*60)


if __name__ == '__main__':
    main()

