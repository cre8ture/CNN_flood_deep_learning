import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(10, 31, kernel_size=2, padding=0)  # Adjust input channels to 10
        self.bn1 = nn.BatchNorm1d(31)
        self.conv2 = nn.Conv1d(31, 32, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        # Adjust the input size after flattening
        self.fc1 = nn.Linear(32, 512)

        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 16836)

        self.dropout = nn.Dropout(p=0.2)  # Set dropout rate to 0.2

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)  # Apply dropout with a rate of 0.2
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.dropout(x)  # Apply dropout with a rate of 0.2
        x = F.relu(self.bn5(self.fc3(x)))
        x = self.fc4(x)
        return x

# Create an instance of the ConvNet model
model = ConvNet()

criterion = nn.MSELoss()
    
# Set the optimizer
optimizer = optim.Adam(model.parameters())

