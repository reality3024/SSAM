import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network import ResBase, feat_bootleneck, feat_classifier
from data_list import ImageList_idx
from loss import Entropy

def image_test(resize_size=256, crop_size=224):
    """Image preprocessing for testing"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def load_model(model_path, net_name='resnet101', class_num=79):
    """Load pretrained model from checkpoint"""
    # Initialize networks
    netF = ResBase(res_name=net_name)
    netB = feat_bootleneck(netF.in_features, bottleneck_dim=256, type="bn")
    netC = feat_classifier(class_num, bottleneck_dim=256, type="wn")
    
    # Load model weights
    if os.path.exists(os.path.join(model_path, "source_F.pt")):
        netF.load_state_dict(torch.load(os.path.join(model_path, "source_F.pt")))
        netB.load_state_dict(torch.load(os.path.join(model_path, "source_B.pt")))
        netC.load_state_dict(torch.load(os.path.join(model_path, "source_C.pt")))
    else:
        raise FileNotFoundError(f"Model files not found in {model_path}")
    
    # Set to evaluation mode and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netF.to(device).eval()
    netB.to(device).eval()
    netC.to(device).eval()
    
    return netF, netB, netC, device

def load_data(data_path, batch_size=64, num_workers=4):
    """Load M58 dataset"""
    # Load data list
    with open(data_path, 'r') as f:
        data_list = f.readlines()
    # print(f'data path: {data_path}')
    # print(f'dir name: {os.path.dirname(data_path)}')
    # Create dataset and dataloader
    root_path = os.path.dirname(data_path) + '/'
    dataset = ImageList_idx(data_list, transform=image_test(), root=root_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, drop_last=False)
    
    return dataloader

def evaluate_and_get_misclassified(netF, netB, netC, dataloader, device):
    """Evaluate model and collect misclassified samples with their entropy"""
    all_predictions = []
    all_labels = []
    all_entropies = []
    all_confidences = []
    misclassified_entropies = []
    correctly_classified_entropies = []
    
    print("Evaluating model and collecting predictions...")
    
    with torch.no_grad():
        for i, (inputs, labels, _, _) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            features = netF(inputs)
            features = netB(features)
            outputs = netC(features)
            
            # Get predictions and probabilities
            softmax_out = F.softmax(outputs, dim=1)
            predictions = torch.max(softmax_out, 1)[1]
            
            # Calculate entropy for each sample
            entropy = Entropy(softmax_out)
            
            # Get maximum confidence (probability of predicted class)
            max_confidence = torch.max(softmax_out, 1)[0]
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_entropies.extend(entropy.cpu().numpy())
            all_confidences.extend(max_confidence.cpu().numpy())
            
            # Separate misclassified and correctly classified samples
            correct_mask = (predictions == labels)
            misclassified_entropies.extend(entropy[~correct_mask].cpu().numpy())
            correctly_classified_entropies.extend(entropy[correct_mask].cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_entropies = np.array(all_entropies)
    all_confidences = np.array(all_confidences)
    misclassified_entropies = np.array(misclassified_entropies)
    correctly_classified_entropies = np.array(correctly_classified_entropies)
    
    # Calculate accuracy
    accuracy = np.mean(all_predictions == all_labels)
    num_misclassified = len(misclassified_entropies)
    num_correctly_classified = len(correctly_classified_entropies)
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Number of misclassified samples: {num_misclassified}")
    print(f"Number of correctly classified samples: {num_correctly_classified}")
    print(f"Total samples: {len(all_predictions)}")
    
    return {
        'all_entropies': all_entropies,
        'all_predictions': all_predictions,
        'all_labels': all_labels,
        'all_confidences': all_confidences,
        'misclassified_entropies': misclassified_entropies,
        'correctly_classified_entropies': correctly_classified_entropies,
        'accuracy': accuracy
    }

def plot_entropy_histograms(results, save_dir='./'):
    """Plot entropy histograms for misclassified vs correctly classified samples"""
    plt.style.use('default')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Entropy histogram for misclassified samples
    axes[0, 0].hist(results['misclassified_entropies'], bins=50, alpha=0.7, 
                    color='red', edgecolor='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Entropy')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].set_title(f'Entropy Distribution - Misclassified Samples\n(n={len(results["misclassified_entropies"])})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Entropy histogram for correctly classified samples
    axes[0, 1].hist(results['correctly_classified_entropies'], bins=50, alpha=0.7, 
                    color='green', edgecolor='black', linewidth=0.5)
    axes[0, 1].set_xlabel('Entropy')
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].set_title(f'Entropy Distribution - Correctly Classified Samples\n(n={len(results["correctly_classified_entropies"])})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Overlapped entropy histograms
    axes[1, 0].hist(results['misclassified_entropies'], bins=50, alpha=0.6, 
                    color='red', label='Misclassified', density=True)
    axes[1, 0].hist(results['correctly_classified_entropies'], bins=50, alpha=0.6, 
                    color='green', label='Correctly Classified', density=True)
    axes[1, 0].set_xlabel('Entropy')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Entropy Distribution Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Box plot comparison
    data_to_plot = [results['correctly_classified_entropies'], results['misclassified_entropies']]
    labels = ['Correctly Classified', 'Misclassified']
    box_plot = axes[1, 1].boxplot(data_to_plot, labels=labels, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('green')
    box_plot['boxes'][1].set_facecolor('red')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Entropy Distribution - Box Plot Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, 'entropy_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Entropy analysis plot saved to: {save_path}")
    
    # Display statistics
    print("\nEntropy Statistics:")
    print(f"Misclassified samples - Mean: {np.mean(results['misclassified_entropies']):.4f}, "
          f"Std: {np.std(results['misclassified_entropies']):.4f}")
    print(f"Correctly classified samples - Mean: {np.mean(results['correctly_classified_entropies']):.4f}, "
          f"Std: {np.std(results['correctly_classified_entropies']):.4f}")
    
    plt.show()

def plot_confidence_analysis(results, save_dir='./'):
    """Plot additional confidence analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Separate confidence for misclassified vs correctly classified
    correct_mask = (results['all_predictions'] == results['all_labels'])
    misclassified_confidences = results['all_confidences'][~correct_mask]
    correct_confidences = results['all_confidences'][correct_mask]
    
    # Plot 1: Confidence histogram
    axes[0].hist(misclassified_confidences, bins=50, alpha=0.6, 
                color='red', label='Misclassified', density=True)
    axes[0].hist(correct_confidences, bins=50, alpha=0.6, 
                color='green', label='Correctly Classified', density=True)
    axes[0].set_xlabel('Confidence (Max Probability)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Confidence Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Entropy vs Confidence scatter plot
    axes[1].scatter(results['all_confidences'][correct_mask], 
                   results['all_entropies'][correct_mask], 
                   alpha=0.5, s=1, color='green', label='Correctly Classified')
    axes[1].scatter(results['all_confidences'][~correct_mask], 
                   results['all_entropies'][~correct_mask], 
                   alpha=0.7, s=1, color='red', label='Misclassified')
    axes[1].set_xlabel('Confidence (Max Probability)')
    axes[1].set_ylabel('Entropy')
    axes[1].set_title('Entropy vs Confidence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, 'confidence_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confidence analysis plot saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze misclassified samples and plot entropy histograms')
    parser.add_argument('--model_path', type=str, 
                       default='/mnt/backups/andycw/UDA-AI/ckps/source/uda/M58/C',
                       help='Path to pretrained model directory')
    parser.add_argument('--data_path', type=str,
                       default='/mnt/backups/andycw/UDA-AI/data/M58/Real_all_nobg_augmented_79_list.txt',
                       help='Path to target dataset list file')
    parser.add_argument('--net', type=str, default='resnet101',
                       help='Network architecture (resnet101, resnet50, etc.)')
    parser.add_argument('--class_num', type=int, default=79,
                       help='Number of classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--save_dir', type=str, default='./',
                       help='Directory to save plots and results')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Loading model...")
    # Load pretrained model
    netF, netB, netC, device = load_model(args.model_path, args.net, args.class_num)
    
    print("Loading data...")
    # Load data
    dataloader = load_data(args.data_path, args.batch_size, args.num_workers)
    
    print("Starting evaluation...")
    # Evaluate and collect misclassified samples
    results = evaluate_and_get_misclassified(netF, netB, netC, dataloader, device)
    
    print("Generating plots...")
    # Plot entropy histograms
    plot_entropy_histograms(results, args.save_dir)
    
    # Plot confidence analysis
    plot_confidence_analysis(results, args.save_dir)
    
    # Save detailed results
    np.savez(os.path.join(args.save_dir, 'analysis_results.npz'),
             all_entropies=results['all_entropies'],
             all_predictions=results['all_predictions'],
             all_labels=results['all_labels'],
             all_confidences=results['all_confidences'],
             misclassified_entropies=results['misclassified_entropies'],
             correctly_classified_entropies=results['correctly_classified_entropies'],
             accuracy=results['accuracy'])
    
    print(f"Analysis complete! Results saved in: {args.save_dir}")
    print(f"Overall accuracy: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()