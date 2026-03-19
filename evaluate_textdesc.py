"""
Compare source and target domain descriptions using cosine similarity
"""
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import sklearn.metrics as sm
import clip
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==================== Text Embedding Evaluator ====================
class TextEmbeddingEvaluator:
    def __init__(self, source_csv_path, target_csv_path, device='cuda'):
        """
        Initialize evaluator with original CLIP model (no adapter)
        
        Args:
            source_csv_path: Path to source domain description CSV
            target_csv_path: Path to target domain description CSV
            device: Device to run evaluation on
        """
        self.device = device
        
        # Load CLIP model (convert to float32, consistent with training)
        print(f"Loading CLIP model (RN101)...")
        self.clip_model, _ = clip.load('RN101', device=device)
        self.clip_model.float()
        self.clip_model.eval()
        
        # Load source and target CSV files
        self.source_df = pd.read_csv(source_csv_path)
        self.target_df = pd.read_csv(target_csv_path)
        
        # Build class names from source CSV
        self.classnames = sorted(list(self.source_df['class_name'].unique()))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classnames)}
        
        print(f"Number of classes: {len(self.classnames)}")

    def process_response(self, response_text):
        """
        Args:
            response_text: Input description text
        Returns:
            normalized embedding [1, 512]
        """
        # Split into sentence list
        sentences = str(response_text)
        sentences = response_text.strip().split('\n')
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = sentences[:10]
        
        # Tokenize all sentences
        tokens = torch.cat([clip.tokenize(s) for s in sentences]).to(self.device)
        tokens = tokens.to('cuda')
        
        with torch.no_grad():
            # CLIP encode: [num_sentences, 512]
            text_features = self.clip_model.encode_text(tokens)
            
            text_features = text_features.mean(dim=0, keepdim=True)  # [1, 512]
            
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def evaluate_source_target(self):
        """
        Compare source and target descriptions using CLIP text encoder 
        
        Returns:
            class_stats: Statistics for each class
            detailed_results: Detailed prediction results
        """

        # Calculate source embedding for each class (average)
        source_embeddings = {}  # class_name -> embedding [1, 512]
        
        for class_name in tqdm(self.classnames, desc="Processing Source"):
            class_data = self.source_df[self.source_df['class_name'] == class_name]
            if len(class_data) == 0:
                print(f"Warning: Class '{class_name}' not found in source CSV")
                continue
            
            # Collect all embeddings for this class
            class_embeddings = []
            for _, row in class_data.iterrows():
                response = row['response']
                emb = self.process_response(response)
                class_embeddings.append(emb)
            
            # Average all source images' embeddings
            class_emb = torch.cat(class_embeddings, dim=0).mean(dim=0, keepdim=True)  # [1, 512]
            class_emb = class_emb / class_emb.norm(dim=-1, keepdim=True)
            source_embeddings[class_name] = class_emb
        
        # Organize source embeddings into matrix [79, 512]
        source_features = torch.cat([source_embeddings[name] for name in self.classnames], dim=0)
        # print(f"Source embeddings shape: {source_features.shape}")
        
        # Evaluate target domain
        class_stats = {i: {'total': 0, 'correct': 0} for i in range(len(self.classnames))}
        detailed_results = []
        
        for idx, row in tqdm(self.target_df.iterrows(), total=len(self.target_df), desc="Evaluating Target"):
            class_name = row['class_name']
            response = row['response']
            img_name = row['image_file_name']
            
            # Get ground truth label
            true_label = self.class_to_idx[class_name]
            
            text_emb = self.process_response(response)  # [1, 512]
            
            # Calculate cosine similarity with source embeddings
            temperature = 1.0  # Keep consistent temperature coefficient
            logits = temperature * (text_emb @ source_features.t())  # [1, 79]
            
            # Get top-1 prediction
            pred_idx = logits.argmax(dim=1).item()
            prob = torch.nn.functional.softmax(logits, dim=1)[0, pred_idx].item()
            
            # Statistics
            class_stats[true_label]['total'] += 1
            if pred_idx == true_label:
                class_stats[true_label]['correct'] += 1
            
            # Record detailed results
            detailed_results.append({
                'class_id': true_label,
                'class_name': class_name,
                'img_name': img_name,
                'pred_class_id': pred_idx,
                'pred_class_name': self.classnames[pred_idx],
                'prob': f"{prob:.4f}",
                'correct': int(pred_idx == true_label)
            })
        
        return class_stats, detailed_results


def save_class_statistics(class_stats, class_names, output_dir, output_prefix):
    """
    Calculate and save class statistics
    
    Args:
        class_stats: Dictionary of class statistics
        class_names: List of class names
        output_dir: Output directory
        output_prefix: Prefix for output files
        
    Returns:
        avg_acc: Average accuracy
        total_images_all: Total number of images
        total_correct_all: Total number of correct predictions
    """
    result_data = []
    total_images_all = 0
    total_correct_all = 0
    
    for class_id in range(len(class_names)):
        class_name = class_names[class_id]
        stats = class_stats[class_id]
        total_img = stats['total']
        top1_correct = stats['correct']
        top1_acc = (top1_correct / total_img * 100) if total_img > 0 else 0.0
        
        result_data.append({
            'class_id': class_id,
            'class_name': class_name,
            'total_imgNum': total_img,
            'top1_correct': top1_correct,
            'top1_acc': f"{top1_acc:.2f}%"
        })
        
        total_images_all += total_img
        total_correct_all += top1_correct
    
    # Add average accuracy row
    avg_acc = (total_correct_all / total_images_all * 100) if total_images_all > 0 else 0.0
    result_data.append({
        'class_id': '',
        'class_name': 'Average',
        'total_imgNum': total_images_all,
        'top1_correct': total_correct_all,
        'top1_acc': f"{avg_acc:.2f}%"
    })
    
    # Save class statistics results
    result_df = pd.DataFrame(result_data)
    result_filename = os.path.join(output_dir, f"{output_prefix}_result.csv")
    result_df.to_csv(result_filename, index=False, encoding='utf-8-sig')
    print(f"✓ Class statistics results saved to: {result_filename}")
    
    return avg_acc, total_images_all, total_correct_all


def generate_confusion_matrix(detailed_results, class_names, output_dir, output_prefix):
    """
    Generate confusion matrix
    
    Args:
        detailed_results: List of detailed prediction results
        class_names: List of class names
        output_dir: Output directory
        output_prefix: Prefix for output files
    """
    print(f"\nGenerating confusion matrix...")
    
    # Extract true and predicted labels
    y_true = [result['class_id'] for result in detailed_results]
    y_pred = [result['pred_class_id'] for result in detailed_results]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, 
                         index=class_names, 
                         columns=class_names)
    cm_csv_filename = os.path.join(output_dir, f"{output_prefix}_confusion_matrix.csv")
    cm_df.to_csv(cm_csv_filename, encoding='utf-8-sig')
    print(f"Confusion matrix saved to: {cm_csv_filename}")
    
    # Plot confusion matrix with original values (with annotations)
    plt.figure(figsize=(24, 20))
    
    # Only annotate numbers on diagonal and high-frequency errors
    annot_array = np.array([[str(int(val)) if val > 0 else '' for val in row] for row in cm])
    
    sns.heatmap(cm, 
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues',
                annot=annot_array,
                fmt='',
                cbar_kws={'label': 'Count'},
                linewidths=0.5,
                linecolor='gray',
                annot_kws={'size': 6})
    
    plt.title(f'Confusion Matrix - {output_prefix}\n(Original Scale)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # Save raw version image
    cm_img_raw_filename = os.path.join(output_dir, f"{output_prefix}_confusion_matrix.png")
    plt.savefig(cm_img_raw_filename, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix image saved to: {cm_img_raw_filename}")
    plt.close()


def main():
    # Path configuration
    target_csv_path = '/mnt/backups/andycw/M58/M58_79classes_targetImage_descriptions.csv'
    source_csv_path = '/mnt/backups/andycw/M58/M58_79classes_sourceImage_descriptions.csv'
    output_dir = '/mnt/backups/andycw/UDA-AI/results'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator (original CLIP model)
    evaluator = TextEmbeddingEvaluator(
        source_csv_path=source_csv_path,
        target_csv_path=target_csv_path,
        device='cuda'
    )
    
    # Evaluate source-target comparison without adapter
    class_stats, detailed_results = evaluator.evaluate_source_target()
    output_prefix = "M58_SourceTarget_TextDescriptionsCompare"
    
    # ==================== Save detailed results ====================
    output_df = pd.DataFrame(detailed_results)
    output_df = output_df.sort_values('class_id').reset_index(drop=True)
    output_filename = os.path.join(output_dir, f"{output_prefix}_output.csv")
    output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n Detailed prediction results saved to: {output_filename}")
    
    # ==================== Calculate and save class statistics ====================
    avg_acc, total_images_all, total_correct_all = save_class_statistics(
        class_stats, evaluator.classnames, output_dir, output_prefix
    )
    
    # ==================== Generate confusion matrix ====================
    generate_confusion_matrix(detailed_results, evaluator.classnames, output_dir, output_prefix)
    
    # ==================== Display results preview ====================
    print(f"\nOverall accuracy: {avg_acc:.2f}%")
    print(f"  - Total samples: {total_images_all}")
    print(f"  - Correct predictions: {total_correct_all}")


if __name__ == '__main__':
    main()