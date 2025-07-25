import os
import copy
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class ImageClassificationDataset(Dataset):
    def __init__(self, df, processor, label_encoder):
        self.df = df
        self.processor = processor
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['path']
        image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        label = self.label_encoder.transform([row['category']])[0]
        
        return pixel_values, label

def run_finetuning(train_df: pd.DataFrame, test_df: pd.DataFrame, le: LabelEncoder):
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}.")
    
    train_dataset = ImageClassificationDataset(train_df, processor, le)
    test_dataset = ImageClassificationDataset(test_df, processor, le)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=os.cpu_count())
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=os.cpu_count())
    
    for param in model.base_model.parameters():
        param.requires_grad = False
        
    optimizer = AdamW(model.classifier.parameters(), lr=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    num_epochs = 12

    # Early stopping parameters
    patience = 2
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = None

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
    print(f"Fine-tuned ResNet-50 Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true_resnet_tuned, y_pred_resnet_tuned, target_names=le.classes_))

    cm = confusion_matrix(y_true_resnet_tuned, y_pred_resnet_tuned)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='cividis')
    plt.title('Fine-tuned ResNet-50 Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
