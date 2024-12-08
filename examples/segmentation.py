from src.evaluation import evaluate

if __name__ == "__main__":
    evaluate(data_dir="datasets/nuscenes", checkpoint="unipad_model.pth")
