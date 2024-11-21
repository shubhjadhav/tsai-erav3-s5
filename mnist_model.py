import ssl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Fix SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context

class LightMNIST(nn.Module):
    def __init__(self):
        super(LightMNIST, self).__init__()
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)  # 8x26x26
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)  # 16x24x24
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3)  # 16x22x22
        self.bn3 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 9 * 9, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 8x13x13
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 1)  # 16x11x11
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 1)  # 16x9x9
        
        x = x.view(-1, 16 * 9 * 9)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

def train_model():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Model initialization
    model = LightMNIST().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    
    # Training loop
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        running_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {running_loss/100:.4f}, '
                  f'Accuracy: {100. * correct / total:.2f}%')
            running_loss = 0.0
    
    final_loss = loss.item()  # Get the final batch loss
    print(f'Final Loss: {final_loss:.4f}, Final Accuracy: {100. * correct / total:.2f}%')

if __name__ == '__main__':
    train_model() 