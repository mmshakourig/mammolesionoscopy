import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasethelpers import BreastCancerDataset, get_transforms
from Final.extra.model import BreastCancerClassifier

def train(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")

    # Dataset arguments
    BASE_DIR = args.base_dir
    train_csv = os.path.join(BASE_DIR, 'csv', 'mass_case_description_train_set.csv')
    test_csv = os.path.join(BASE_DIR, 'csv', 'mass_case_description_test_set.csv')

    # Load datasets
    train_dataset = BreastCancerDataset(train_csv, BASE_DIR, transform=get_transforms(is_train=True))
    test_dataset = BreastCancerDataset(test_csv, BASE_DIR, transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize model
    model = BreastCancerClassifier(num_classes=2, use_pretrained=True).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training for {args.epochs} epochs.")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(test_dataset)
        epoch_val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} Val Loss: {epoch_val_loss:.4f} Val Acc: {epoch_val_acc:.4f}")

    # Save final model
    os.makedirs('weights', exist_ok=True)
    torch.save(model.state_dict(), 'weights/final_mammography_cnn.pth')
    print("Training Complete. Model saved at weights/final_mammography_cnn.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/kaggle/input/cbis-ddsm-breast-cancer-image-dataset', help='Path to Kaggle dataset root')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for data loader')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    
    args = parser.parse_args()
    train(args)
