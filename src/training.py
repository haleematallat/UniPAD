import torch
import torch.nn as nn
import torch.optim as optim
from src.data_processing import load_dataset
from src.model.unipad import UniPAD

def train(data_dir, num_epochs=10, batch_size=4, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = load_dataset(data_dir, batch_size)

    model = UniPAD(image_backbone=nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for lidar, images in data_loader:
            lidar, images = lidar.to(device), images.to(device)
            optimizer.zero_grad()
            output = model(lidar, images)
            loss = criterion(output, images)  # Example loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}")

    torch.save(model.state_dict(), "unipad_model.pth")
