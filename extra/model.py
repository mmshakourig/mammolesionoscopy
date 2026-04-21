import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# Keras / TensorFlow imports
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class ShallowKerasCNN:
    def __init__(self, input_shape=(640, 640, 1), num_classes=1, device=torch.device('cpu')):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            
            Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])
        
    def forward(self, x):
        # Convert PyTorch tensor to NumPy array for Keras
        x_np = x.cpu().numpy()
        # Add channel dimension if missing
        if x_np.ndim == 3:
            x_np = np.expand_dims(x_np, axis=-1)
        # Run through Keras model
        output = self.model(x_np)
        # Convert back to PyTorch tensor
        output_tensor = torch.from_numpy(output).to(x.device)
        return output_tensor

    def train(self, x, y):
        

class ShallowCNN(nn.Module):
    """
    The PyTorch equivalent of the shallow Keras CNN.
    """
    def __init__(self, in_channels=1, num_classes=1):
        super(ShallowCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 54 * 54, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Identity()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResNet50CNN(nn.Module):
    def __init__(self, num_classes=2, use_pretrained=True):
        super(ResNet50CNN, self).__init__()
        
        # We classify into Benign (0) vs Malignant (1)
        if use_pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.model = resnet50(weights=weights)
        else:
            self.model = resnet50()

        # Replace the final fully connected layer to output binary labels
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
    def train(imgs_tensor, labels):
        # Implement training loop here
        def train_loop(model, dataloader, criterion, optimizer, device):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(dataloader, desc="Training"):
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

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = correct / total
            print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

