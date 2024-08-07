import cv2
import torch
from ultralytics import YOLO


def train_model(model_path, data_path, save_path, stage, epochs=100, imgsz=640, batch_size=16, time = 44, run_dir='runs'):
    """
    Train or fine-tune a YOLO model.

    Parameters:
    - model_path (str): Path to the model file.
    - data_path (str): Path to the data configuration file.
    - save_path (str): Path to save the trained model.
    - stage (str): Training stage, either 'pretrain' or 'finetune'.
    - epochs (int): Number of training epochs. Default is 100.
    - imgsz (int): Image size for training. Default is 640.
    - batch_size (int): Batch size for training. Default is 16.
    """

    # Load the YOLO model
    model = YOLO(model_path)

    # Define training parameters
    train_params = {
        'data': data_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
        'plots': True,
        'time': time,
        'patience': 100,
        'project': run_dir  # Specify the directory for storing runs
    }

    if stage == 'pretrain':
        # Start pre-training
        print("Starting pre-training...")
        results = model.train(**train_params)
        # Save the pre-trained model
        model.save(save_path)
    elif stage == 'finetune':
        # Start fine-tuning
        print("Starting fine-tuning...")
        results = model.train(**train_params)
        # Save the fine-tuned model
        model.save(save_path)

    # Print training results
    print(results)

# Pre-training stage:
# Uncomment the block below for the pre-training stage
# if __name__ == "__main__":
#     stage = 'pretrain'  # Specify 'pretrain' for the pre-training stage
#     train_model(
#         model_path='Models/yolov8n.pt',  # Path to the initial YOLO model
#         data_path='data.yaml',  # Path to the data configuration file for pre-training
#         save_path='Models/pre_trained_yolov8s.pt',  # Path to save the pre-trained model
#         stage=stage,  # Specify the stage: 'pretrain' or 'finetune'
#         epochs=100  # Number of training epochs for pre-training
#     )

# Fine-tuning stage:
if __name__ == "__main__":
    stage = 'finetune'  # Specify 'finetune' for the fine-tuning stage
    train_model(
        model_path='Models/pre_trained_yolov8s.pt',  # Path to the pre-trained model
        data_path='data_finetune.yaml',  # Path to the data configuration file for fine-tuning
        save_path='Models/fine_tuned_yolov8s.pt',  # Path to save the fine-tuned model
        stage=stage,  # Specify the stage: 'pretrain' or 'finetune'
        epochs=300,  # Number of training epochs for fine-tuning
        run_dir='runs/fine_tuning'  # Specify the directory for this fine-tuning run
    )

