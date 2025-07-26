import os
import copy
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

def get_image_paths(data_dir: str) -> pd.DataFrame:
    """Gathers image paths and their categories from the data directory."""
    records = []
    for category_dir in Path(data_dir).iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for image_path in category_dir.iterdir():
            if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                records.append({'path': str(image_path), 'category': category})
    return pd.DataFrame(records)

class ImageClassificationDataset(Dataset):
    def __init__(self, df, processor, label_encoder, transform=None):
        self.df = df
        self.processor = processor
        self.label_encoder = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['path']
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        label = self.label_encoder.transform([row['category']])[0]
        
        return pixel_values, label

def get_augmentations(strategy: str = 'none'):
    """Returns a composition of transforms for data augmentation."""
    if strategy == 'basic':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(10),
        ])
    # 'none' or any other value will result in no augmentations
    return None

def get_classifier_head(name: str, in_features: int, num_labels: int):
    """Returns a classifier head based on the specified name."""
    if name == 'simple':
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features, num_labels)
        )
    elif name == 'Alalibo et all':
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, num_labels)
        )
    else:
        raise ValueError(f"Unknown classifier head: {name}")

def get_loss_function(name: str, class_weights=None):
    """Returns a loss function based on the specified name."""
    if name == 'cross_entropy_weighted':
        return torch.nn.CrossEntropyLoss(weight=class_weights)
    if name == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")


def run_finetuning(train_df: pd.DataFrame, test_df: pd.DataFrame, le: LabelEncoder,
                   head_name: str = 'complex', loss_fn_name: str = 'cross_entropy',
                   augmentation_strategy: str = 'none'):
    print("Loading ResNet-50 model and replacing classification head...")
    num_labels = len(le.classes_)
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/resnet-50",
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(le.classes_)},
        label2id={label: i for i, label in enumerate(le.classes_)},
        ignore_mismatched_sizes=True # This allows replacing the head
    )
    
    # Get classifier head
    in_features = model.classifier[1].in_features
    model.classifier = get_classifier_head(head_name, in_features, num_labels)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}.")
    
    # Get augmentations
    train_transforms = get_augmentations(augmentation_strategy)
    # Note: No augmentations on test set, which is standard practice.
    
    train_dataset = ImageClassificationDataset(train_df, processor, le, transform=train_transforms)
    test_dataset = ImageClassificationDataset(test_df, processor, le, transform=None)
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
    
    for param in model.base_model.parameters():
        param.requires_grad = False

    print("Calculating class weights to handle imbalance...")
    class_counts = train_df['category'].value_counts()
    num_samples = len(train_df)
    num_classes = len(le.classes_)
    weights = [num_samples / (num_classes * class_counts[cls]) for cls in le.classes_]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"Using weights for loss function: {class_weights_tensor.cpu().numpy().round(2)}")

    lr = 5e-4
    optimizer = AdamW(model.classifier.parameters(), lr=lr)
    loss_fn = get_loss_function(loss_fn_name, class_weights=class_weights_tensor)
    num_epochs = 12

    # Early stopping parameters
    patience = 2
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = None

    # Instantiate TensorBoard writer
    writer = SummaryWriter()

    # Log configuration details to TensorBoard
    writer.add_text('Configuration/Model Architecture', '```\n' + str(model) + '\n```')
    writer.add_text('Configuration/Preprocessing', '```\n' + str(processor) + '\n```')
    writer.add_text('Configuration/Loss Function', '```\n' + str(loss_fn) + '\n```')

    print("\n--- Fine-tuning the classification head ---")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")
        for batch in progress_bar:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=inputs)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)")
            for batch in val_progress_bar:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(pixel_values=inputs)
                loss = loss_fn(outputs.logits, labels)
                total_val_loss += loss.item()
                val_progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        avg_val_loss = total_val_loss / len(test_loader)
        print(f"Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            print("Validation loss improved. Saving model.")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # Load best model weights before evaluation
    if best_model_weights:
        print("Loading best model weights for evaluation.")
        model.load_state_dict(best_model_weights)

    # 5. Evaluate the fine-tuned model
    print("\n--- Evaluating the fine-tuned model ---")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = batch
            inputs = inputs.to(device)

            outputs = model(pixel_values=inputs)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_true_resnet_tuned = all_labels
    y_pred_resnet_tuned = all_preds

    accuracy = accuracy_score(y_true_resnet_tuned, y_pred_resnet_tuned)
    f1_weighted = f1_score(y_true_resnet_tuned, y_pred_resnet_tuned, average='weighted')
    print(f"Fine-tuned ResNet-50 Accuracy: {accuracy:.4f}")
    print(f"Fine-tuned ResNet-50 F1-Score (Weighted): {f1_weighted:.4f}")

    print("\nClassification Report:")
    report = classification_report(y_true_resnet_tuned, y_pred_resnet_tuned, target_names=le.classes_)
    print(report)

    # Log classification report to TensorBoard as text
    writer.add_text('Evaluation/Classification Report', '```\n' + report + '\n```')

    # Save the report to a file
    report_path = 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    cm = confusion_matrix(y_true_resnet_tuned, y_pred_resnet_tuned)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='cividis')
    plt.title('Fine-tuned ResNet-50 Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Log confusion matrix figure to TensorBoard
    writer.add_figure('Evaluation/Confusion Matrix', fig)
    
    # Log hyperparameters and metrics to TensorBoard
    hparams = {
        'lr': lr,
        'optimizer': 'AdamW',
        'num_epochs': num_epochs,
        'patience': patience,
        'batch_size': batch_size,
        'head': head_name,
        'loss_function': loss_fn_name,
        'augmentation': augmentation_strategy
    }
    metrics = {
        'hparam/accuracy': accuracy,
        'hparam/f1_score_weighted': f1_weighted,
        'hparam/best_val_loss': best_val_loss
    }
    writer.add_hparams(hparams, metrics)
    writer.close()
    
    # Save the confusion matrix plot
    cm_path = 'confusion_matrix.png'
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")
    
    plt.show()

if __name__ == '__main__':
    # This allows running the fine-tuning script directly.
    data_dir = './tp1/data/1/dataset-resized'
    
    print(f"Loading data from {data_dir}...")
    df = get_image_paths(data_dir)
    
    if df.empty:
        print("No images found. Exiting.")
    else:
        print(f"Found {len(df)} images.")
        
        # Create labels and split data
        le = LabelEncoder()
        df['category_encoded'] = le.fit_transform(df['category'])
        
        train_df, test_df = train_test_split(
            df,
            test_size=0.3,
            random_state=42,
            stratify=df['category_encoded']
        )
        
        print(f"Train set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")
        
        # Run the fine-tuning process
        run_finetuning(
            train_df, 
            test_df, 
            le,
            head_name='complex',
            loss_fn_name='cross_entropy',
            augmentation_strategy='basic'
        )
