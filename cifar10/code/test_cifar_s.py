# -*- coding: utf-8 -*-
"""
CIFAR-10-S Evaluation Script
Evaluates model stability on CIFAR-10 with different image processing methods
Similar to ImageNet-S evaluation
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from scipy.stats import rankdata


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CIFAR-10-S stability')
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='Model name (e.g., ResNet18, ResNet50, vit)')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str,
                       default='/data/RobustART/datasets/cifar-10-batches-py',
                       help='Path to CIFAR-10 dataset')
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


def ranking_dist(ranks, mode='top5', num_classes=10):
    """Compute ranking distance"""
    result = 0
    
    for vid_ranks in ranks:
        result_for_vid = []
        
        # Compare first frame with all subsequent frames
        perm1 = vid_ranks[0]
        perm1_inv = np.argsort(perm1)
        
        for rank in vid_ranks[1:]:
            perm2 = rank
            result_for_vid.append(dist(perm2[perm1_inv], mode, num_classes))
        
        if len(result_for_vid) > 0:
            result += np.mean(result_for_vid) / len(ranks)
    
    return result


def flip_prob(predictions):
    """Compute flip probability"""
    result = 0
    
    for vid_preds in predictions:
        result_for_vid = []
        prev_pred = vid_preds[0]
        
        for pred in vid_preds[1:]:
            result_for_vid.append(int(prev_pred != pred))
            prev_pred = pred
        
        if len(result_for_vid) > 0:
            result += np.mean(result_for_vid) / len(predictions)
    
    return result


def evaluate_processing_method(model, test_loader, transform, device='cuda'):
    """Evaluate model on test set with a specific transform"""
    num_correct = 0
    total = 0
    all_predictions = []
    all_ranks = []
    
    # CIFAR-10 mean and std
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)
    
    from PIL import Image
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating', leave=False):
            # Data is in [0, 1] range (from ToTensor)
            # Convert to PIL, apply transform, convert back
            data_transformed = []
            for img in data:
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                img_transformed = transform(img_pil)
                data_transformed.append(transforms.ToTensor()(img_transformed))
            
            data_tensor = torch.stack(data_transformed).to(device)
            # Normalize
            data_tensor = (data_tensor - mean) / std
            
            target = target.to(device)
            
            # Forward pass
            outputs = model(data_tensor)
            preds = outputs.argmax(dim=1)
            
            num_correct += (preds == target).sum().item()
            total += target.size(0)
            
            # Store predictions and rankings for stability metrics
            all_predictions.extend(preds.cpu().numpy().tolist())
            all_ranks.extend([np.uint16(rankdata(-frame, method='ordinal')) 
                            for frame in F.softmax(outputs, dim=1).cpu().numpy()])
    
    accuracy = num_correct / total
    return accuracy, all_predictions, all_ranks


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    
    # Load model
    print(f'Loading model: {args.model}')
    model = load_model(args.model, args.checkpoint, device)
    print('Model loaded successfully')
    
    # Load CIFAR-10 test set (without normalization, we'll apply it after transform)
    print('Loading CIFAR-10 test set...')
    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir, 
        train=False, 
        download=True, 
        transform=transforms.ToTensor()  # Only convert to tensor, no normalization yet
    )
    test_loader = torch.utils.data.DataLoader(
        testset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    print(f'Loaded {len(testset)} test samples')
    
    # Define different processing methods (similar to ImageNet-S)
    # Test different resize operations with different interpolation methods
    # For CIFAR-10, we resize to different sizes and then back to 32x32
    processing_methods = {}
    
    # Different interpolation methods at original size (32x32)
    # Resize to 40 then back to 32 to test interpolation
    resize_sizes = [24, 28, 40, 48]
    interpolation_modes = [
        ('nearest', transforms.InterpolationMode.NEAREST),
        ('bilinear', transforms.InterpolationMode.BILINEAR),
        ('bicubic', transforms.InterpolationMode.BICUBIC),
    ]
    
    for interp_name, interp_mode in interpolation_modes:
        for size in resize_sizes:
            processing_methods[f'{interp_name}_{size}'] = transforms.Compose([
                transforms.Resize(size, interpolation=interp_mode),
                transforms.Resize(32, interpolation=interp_mode),
            ])
    
    # Also test identity (no transform) as baseline
    processing_methods['identity'] = transforms.Lambda(lambda x: x)
    
    results = {}
    all_predictions_list = []
    all_ranks_list = []
    
    print('\n' + '='*60)
    print('Evaluating on CIFAR-10-S (different processing methods)')
    print('='*60)
    
    # Evaluate on each processing method
    for method_name, transform in processing_methods.items():
        print(f'\nEvaluating {method_name}...')
        
        accuracy, predictions, ranks = evaluate_processing_method(
            model, test_loader, transform, device
        )
        
        results[method_name] = {
            'accuracy': accuracy,
            'predictions': predictions,
            'ranks': ranks
        }
        
        all_predictions_list.append(predictions)
        all_ranks_list.append(ranks)
        
        print(f'  Accuracy: {accuracy*100:.2f}%')
    
    # Compute stability metrics across all methods
    print('\n' + '='*60)
    print('Computing Stability Metrics')
    print('='*60)
    
    # Group predictions by sample across all methods
    num_samples = len(testset)
    sample_predictions = [[] for _ in range(num_samples)]
    sample_ranks = [[] for _ in range(num_samples)]
    
    # Collect all predictions and ranks
    for method_name, result in results.items():
        preds = result['predictions']
        ranks = result['ranks']
        for idx, (pred, rank) in enumerate(zip(preds, ranks)):
            if idx < num_samples:
                sample_predictions[idx].append(pred)
                sample_ranks[idx].append(rank)
    
    # Compute flip probability and ranking distance
    flip_list = []
    zipf_list = []
    
    for sample_preds, sample_rank_list in zip(sample_predictions, sample_ranks):
        if len(sample_preds) > 1:
            # Flip probability: compare first prediction with all others
            flips = sum(1 for pred in sample_preds[1:] if pred != sample_preds[0])
            flip_list.append(flips / (len(sample_preds) - 1))
            
            # Ranking distance: compare first ranking with all others
            if len(sample_rank_list) > 1:
                perm1 = sample_rank_list[0]
                perm1_inv = np.argsort(perm1)
                zipf_dists = []
                for rank in sample_rank_list[1:]:
                    zipf_dists.append(dist(rank[perm1_inv], mode='zipf', num_classes=10))
                if zipf_dists:
                    zipf_list.append(np.mean(zipf_dists))
    
    mean_flip = np.mean(flip_list) if flip_list else 0.0
    mean_zipf = np.mean(zipf_list) if zipf_list else 0.0
    
    # Print summary
    print('\n' + '='*60)
    print('Summary Results')
    print('='*60)
    for method_name, result in results.items():
        print(f'{method_name:20s}: Accuracy = {result["accuracy"]*100:6.2f}%')
    
    print(f'\nMean Accuracy: {np.mean([r["accuracy"] for r in results.values()])*100:.2f}%')
    print(f'Std Accuracy: {np.std([r["accuracy"] for r in results.values()])*100:.2f}%')
    print(f'\nMean Flipping Probability: {mean_flip:.5f}')
    print(f'Mean Zipf Distance: {mean_zipf:.5f}')
    print('='*60)


if __name__ == '__main__':
    main()

