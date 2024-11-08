import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from torch.utils.data import Dataset, DataLoader
import random

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TwoDigitDataset(Dataset):
    def __init__(self, mnist_dataset, transform=None):
        self.mnist_dataset = mnist_dataset
        self.transform = transform
        self.length = len(mnist_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get first digit
        img1, label1 = self.mnist_dataset[idx]
        
        # Randomly select second digit
        idx2 = random.randint(0, self.length - 1)
        img2, label2 = self.mnist_dataset[idx2]
        
        # Combine images side by side
        combined_img = torch.zeros((1, 28, 56))  # New width is 56 (28*2)
        combined_img[:, :, :28] = img1
        combined_img[:, :, 28:] = img2
        
        # Calculate combined number (label1 * 10 + label2)
        combined_label = label1 * 10 + label2
        
        return combined_img, (label1, label2, combined_label)

class ImprovedTwoDigitRecognizer(nn.Module):
    def __init__(self):
        super(ImprovedTwoDigitRecognizer, self).__init__()
        # Modified conv layers for wider input (56 pixels instead of 28)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        
        # Modified FC layers with three outputs (first digit, second digit, combined number)
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 14, 512),  # Note: 14 instead of 7 due to wider input
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # Separate output layers for each digit and combined number
        self.digit1_out = nn.Linear(256, 10)
        self.digit2_out = nn.Linear(256, 10)
        self.combined_out = nn.Linear(256, 100)  # 0-99 range

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * 7 * 14)
        x = self.fc_layers(x)
        
        digit1 = self.digit1_out(x)
        digit2 = self.digit2_out(x)
        combined = self.combined_out(x)
        
        return digit1, digit2, combined

def train(model, train_loader, optimizer, epoch, scheduler):
    model.train()
    total_loss = 0
    correct_digits = 0
    correct_combined = 0
    total = 0
    start_time = time.time()

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        label1, label2, combined_label = targets
        label1, label2, combined_label = label1.to(device), label2.to(device), combined_label.to(device)
        
        optimizer.zero_grad()
        output1, output2, output_combined = model(data)
        
        # Calculate losses
        loss1 = nn.CrossEntropyLoss()(output1, label1)
        loss2 = nn.CrossEntropyLoss()(output2, label2)
        loss_combined = nn.CrossEntropyLoss()(output_combined, combined_label)
        total_batch_loss = loss1 + loss2 + loss_combined
        
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate accuracy
        pred1 = output1.argmax(dim=1)
        pred2 = output2.argmax(dim=1)
        pred_combined = output_combined.argmax(dim=1)
        
        correct_digits += (pred1.eq(label1) & pred2.eq(label2)).sum().item()
        correct_combined += pred_combined.eq(combined_label).sum().item()
        total += label1.size(0)
        
        total_loss += total_batch_loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {total_batch_loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    digit_accuracy = 100. * correct_digits / total
    combined_accuracy = 100. * correct_combined / total
    time_taken = time.time() - start_time
    
    print(f'Training Epoch {epoch}: Average loss: {avg_loss:.4f}, '
          f'Digit Accuracy: {correct_digits}/{total} ({digit_accuracy:.2f}%), '
          f'Combined Accuracy: {correct_combined}/{total} ({combined_accuracy:.2f}%), '
          f'Time: {time_taken:.2f}s')
    
    return avg_loss

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct_digits = 0
    correct_combined = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            label1, label2, combined_label = targets
            label1, label2, combined_label = label1.to(device), label2.to(device), combined_label.to(device)
            
            output1, output2, output_combined = model(data)
            
            loss1 = nn.CrossEntropyLoss()(output1, label1)
            loss2 = nn.CrossEntropyLoss()(output2, label2)
            loss_combined = nn.CrossEntropyLoss()(output_combined, combined_label)
            test_loss += (loss1 + loss2 + loss_combined).item()
            
            pred1 = output1.argmax(dim=1)
            pred2 = output2.argmax(dim=1)
            pred_combined = output_combined.argmax(dim=1)
            
            correct_digits += (pred1.eq(label1) & pred2.eq(label2)).sum().item()
            correct_combined += pred_combined.eq(combined_label).sum().item()
            total += label1.size(0)

    test_loss /= len(test_loader)
    digit_accuracy = 100. * correct_digits / total
    combined_accuracy = 100. * correct_combined / total
    
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Digit Accuracy: {correct_digits}/{total} ({digit_accuracy:.2f}%), '
          f'Combined Accuracy: {correct_combined}/{total} ({combined_accuracy:.2f}%)\n')
    
    return test_loss, digit_accuracy, combined_accuracy

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
def main():
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    mnist_test = datasets.MNIST('data', train=False, transform=test_transform)
    
    # Create two-digit datasets
    train_dataset = TwoDigitDataset(mnist_train)
    test_dataset = TwoDigitDataset(mnist_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    model = ImprovedTwoDigitRecognizer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    early_stopping = EarlyStopping(patience=7)

    best_accuracy = 0
    num_epochs = 50

    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch, scheduler)
        test_loss, digit_accuracy, combined_accuracy = test(model, test_loader)
        
        scheduler.step(test_loss)
        
        # Save best model based on combined accuracy
        if combined_accuracy > best_accuracy:
            best_accuracy = combined_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': combined_accuracy,
            }, 'best_two_digit_recognizer.pth')
            print(f"New best model saved with combined accuracy: {combined_accuracy:.2f}%")
        
        early_stopping(test_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    print(f"Best combined accuracy achieved: {best_accuracy:.2f}%")

if __name__ == '__main__':
    main()