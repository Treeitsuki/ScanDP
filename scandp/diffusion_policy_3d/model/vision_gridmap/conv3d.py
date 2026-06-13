import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DEncoder(nn.Module):
    def __init__(self):
        super(Conv3DEncoder, self).__init__()
        
        self.device = 'cuda'
        self.input_channels=1
        self.output_dim=256
        
        # First 3D convolutional layer
        self.conv1 = nn.Conv3d(in_channels=self.input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        
        # Second 3D convolutional layer
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        
        # Third 3D convolutional layer
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        
        # Fully connected layer to output the final 256-dimensional vector
        self.fc = nn.Linear(128, self.output_dim)  # Use the output dimension 128 after Global Average Pooling as input

        # Move the model to the specified device
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global Average Pooling (average spatial dimensions while keeping the batch dimension)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))  # Convert to (B, C, 1, 1, 1)
        x = torch.flatten(x, start_dim=1)       # Convert to (B, C)
        
        # Apply fully connected layer
        x = self.fc(x)  # (256,)
        return x

# Example usage
if __name__ == "__main__":
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv3DEncoder()
    
    # Generate dummy input (batch of 1, 1 channel, depth=200, height=200, width=200)
    dummy_input = torch.randn(128, 1, 25, 25, 25).to(device)
    
    # Pass the dummy input through the model
    t = time.time()
    output = model(dummy_input)

    # Print the output shape
    print("Output shape:", output.shape)
    print("Time: {}s".format(time.time() - t))
