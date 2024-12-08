from src.training import train

if __name__ == "__main__":
    train(data_dir="datasets/nuscenes", num_epochs=5, batch_size=4)
