import torch
from src.model.unipad import UniPAD
from src.data_processing import load_dataset

def evaluate(data_dir, checkpoint="unipad_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = load_dataset(data_dir, batch_size=1)

    model = UniPAD(image_backbone=nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)).to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    for lidar, images in data_loader:
        lidar, images = lidar.to(device), images.to(device)
        with torch.no_grad():
            output = model(lidar, images)
        print("Output:", output)
