import os
import copy
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.trainer_utils import EvalPrediction
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
            pixel_values = self.transform(image)
        else:
            # Default processing if no transforms are provided
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
        
        label = self.label_encoder.transform([row['category']])[0]
        
        return {'pixel_values': pixel_values, 'labels': label}

# A wrapper to make Albumentations transforms compatible with the unified interface
class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)
        return self.transform(image=img)['image']

def get_augmentations(strategy: str = 'none'):
    """Returns a unified transform callable that takes a PIL image and returns a tensor."""
    if strategy == 'basic':
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif strategy ==  'Alalibo et all':
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            # Using RandomAffine to combine rotation, translation, and scaling
            T.RandomAffine(degrees=45, translate=(10/224, 10/224), scale=(1.0, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif strategy == 'albumentations_advanced':
        alb_transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-20, 20), scale=(0.8, 1.2), translate_percent=(0.1, 0.1), p=0.7),
            A.RandomResizedCrop(size=(224, 224),height=224, width=224, scale=(0.5, 1.0), p=0.5),
            A.ColorJitter(hue=0.05, saturation=0.1, brightness=0.1, contrast=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        return AlbumentationsTransform(alb_transform)
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

def compute_metrics(p: EvalPrediction):
    """Computes accuracy and F1 score from predictions."""
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1}

class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def run_finetuning(train_df: pd.DataFrame, test_df: pd.DataFrame, le: LabelEncoder,
                   head_name: str = 'complex', loss_fn_name: str = 'cross_entropy',
                   augmentation_strategy: str = 'none', class_balancing_strategy: str = 'none',
                   balancing_target_samples: int = None, lr = 5e-4):
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
    
    # Oversample the training data to balance classes if an augmentation strategy is provided
    if class_balancing_strategy == 'oversampling':
        print("Balancing training data by resampling...")
        class_counts = train_df['category'].value_counts()

        if balancing_target_samples:
            target_size = balancing_target_samples
        else:
            # Default behavior: oversample to the size of the largest class
            target_size = class_counts.max()
        
        print(f"Target samples per class: {target_size}")
        
        balanced_dfs = []
        for class_name in class_counts.index:
            class_df = train_df[train_df['category'] == class_name]
            # Use resampling to either oversample or undersample to the target size
            resampled_df = class_df.sample(
                n=target_size, 
                replace=(len(class_df) < target_size), 
                random_state=42
            )
            balanced_dfs.append(resampled_df)
        
        # Concatenate and shuffle the new balanced DataFrame
        train_df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"New balanced training set size: {len(train_df)}")
    else:
        print("Skipping class balancing.")
    
    train_dataset = ImageClassificationDataset(train_df, processor, le, transform=train_transforms)
    test_dataset = ImageClassificationDataset(test_df, processor, le, transform=None)
    
    batch_size = 128
    
    for param in model.base_model.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=24,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=os.cpu_count(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./runs',
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        save_total_limit=1,
    )

    class_weights_tensor = None
    TrainerClass = Trainer
    trainer_kwargs = {}

    if 'weighted' in loss_fn_name:
        print("Calculating class weights for the loss function...")
        class_counts = train_df['category'].value_counts()
        num_samples = len(train_df)
        num_classes = len(le.classes_)
        weights = [num_samples / (num_classes * class_counts[cls]) for cls in le.classes_]
        class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        print(f"Calculated weights: {class_weights_tensor.cpu().numpy().round(2)}")
        
        TrainerClass = WeightedLossTrainer
        trainer_kwargs['class_weights'] = class_weights_tensor
    
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
        **trainer_kwargs,
    )
    
    print("\n--- Fine-tuning the classification head ---")
    trainer.train()

    # 5. Evaluate the fine-tuned model
    print("\n--- Evaluating the fine-tuned model ---")
    prediction_output = trainer.predict(test_dataset)
    y_pred_resnet_tuned = np.argmax(prediction_output.predictions, axis=1)
    y_true_resnet_tuned = prediction_output.label_ids

    accuracy = accuracy_score(y_true_resnet_tuned, y_pred_resnet_tuned)
    f1_weighted = f1_score(y_true_resnet_tuned, y_pred_resnet_tuned, average='weighted')
    print(f"Fine-tuned ResNet-50 Accuracy: {accuracy:.4f}")
    print(f"Fine-tuned ResNet-50 F1-Score (Weighted): {f1_weighted:.4f}")

    print("\nClassification Report:")
    report = classification_report(y_true_resnet_tuned, y_pred_resnet_tuned, target_names=le.classes_)
    print(report)

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

    # The Trainer creates a specific log directory, so we grab its path from the state
    writer = SummaryWriter(log_dir=trainer.state.log_dir)
    writer.add_text('Evaluation/Classification Report', '```\n' + report + '\n```')
    writer.add_figure('Evaluation/Confusion Matrix', fig)

    hparams = {
        'lr': lr,
        'batch_size': batch_size,
        'head': head_name,
        'loss_function': loss_fn_name,
        'augmentation': augmentation_strategy,
        'class_balancing': class_balancing_strategy,
        'balancing_target_samples': balancing_target_samples
    }
    metrics = {
        'hparam/accuracy': accuracy,
        'hparam/f1_score_weighted': f1_weighted,
        'hparam/best_val_loss': trainer.state.best_metric
    }
    writer.add_hparams(hparams, metrics)
    writer.close()
    
    # Save the confusion matrix plot
    cm_path = 'confusion_matrix.png'
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")
    
    # plt.show()

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

        run_finetuning(
            train_df, 
            test_df, 
            le,
            head_name='Alalibo et all',
            loss_fn_name='cross_entropy_weighted',
            augmentation_strategy='albumentation_advanced',
            class_balancing_strategy = 'oversampling',
            balancing_target_samples=600,
            lr = 5e-4
        )


