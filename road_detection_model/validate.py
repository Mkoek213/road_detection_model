from ultralytics import YOLO
import torch

def validate_model(model_path, data_path, save_dir):
    """
    Validates a YOLO model on a specified dataset and prints various evaluation metrics.

    Parameters:
    model_path (str): Path to the YOLO model file. This can be a pre-trained or fine-tuned model.
    data_path (str): Path to the data configuration file (e.g., data_finetune.yaml) which specifies
                     the dataset and its annotations for validation.
    save_dir (str): Directory where the validation results and logs will be saved.
    """
    # Load the model from the specified model path
    model = YOLO(model_path)

    # Run the evaluation (validation) using the specified data configuration file
    # The results will be saved in the specified directory
    results = model.val(data=data_path, plots=True)

    # Print specific metrics from the validation results:
    print("Class indices with average precision:", results.ap_class_index)  # Indices of classes with average precision
    print("Average precision for all classes:", results.box.all_ap)         # Average precision for all classes
    print("Average precision:", results.box.ap)                             # Average precision across all classes
    print("Average precision at IoU=0.50:", results.box.ap50)               # Average precision at IoU=0.50 threshold
    print("Class indices for average precision:", results.box.ap_class_index) # Class indices for average precision
    print("Class-specific results:", results.box.class_result)              # Class-specific evaluation results
    print("F1 score:", results.box.f1)                                      # F1 score for each class
    print("F1 score curve:", results.box.f1_curve)                          # F1 score curve across IoU thresholds
    print("F1 score (mean across all classes):", sum(results.box.f1) / len(results.box.f1))  # Mean F1 score
    print("Overall fitness score:", results.box.fitness)                    # Overall fitness score (combination of metrics)
    print("Mean average precision:", results.box.map)                       # Mean average precision across IoU thresholds
    print("Mean average precision at IoU=0.50:", results.box.map50)         # Mean average precision at IoU=0.50
    print("Mean average precision at IoU=0.75:", results.box.map75)         # Mean average precision at IoU=0.75
    print("Mean average precision for different IoU thresholds:", results.box.maps)  # mAP across different IoU thresholds
    print("Mean results for different metrics:", results.box.mean_results)  # Mean results for various metrics
    print("Mean precision:", results.box.mp)                                # Mean precision across all classes
    print("Mean recall:", results.box.mr)                                   # Mean recall across all classes
    print("Precision:", results.box.p)                                      # Precision for each class
    print("Precision curve:", results.box.p_curve)                          # Precision curve across IoU thresholds
    print("Precision values:", results.box.prec_values)                     # Precision values
    print("Specific precision metrics:", results.box.px)                    # Specific precision metrics
    print("Recall:", results.box.r)                                         # Recall for each class
    print("Recall curve:", results.box.r_curve)                             # Recall curve across IoU thresholds

if __name__ == "__main__":
    # Validate the model with specified paths and save the results
    validate_model(model_path='Models/fine_tuned_yolov8s.pt', data_path='data_finetune.yaml', save_dir='runs/fine_tuning')
