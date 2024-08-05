from ultralytics import YOLO
import torch

def validate_model(model_path, data_path):
    # Load the model
    model = YOLO(model_path)

    # Run the evaluation
    results = model.val(data=data_path)

    # Print specific metrics
    print("Class indices with average precision:", results.ap_class_index)
    print("Average precision for all classes:", results.box.all_ap)
    print("Average precision:", results.box.ap)
    print("Average precision at IoU=0.50:", results.box.ap50)
    print("Class indices for average precision:", results.box.ap_class_index)
    print("Class-specific results:", results.box.class_result)
    print("F1 score:", results.box.f1)
    print("F1 score curve:", results.box.f1_curve)
    print("F1 score (mean across all classes):", sum(results.box.f1) / len(results.box.f1))
    print("Overall fitness score:", results.box.fitness)
    print("Mean average precision:", results.box.map)
    print("Mean average precision at IoU=0.50:", results.box.map50)
    print("Mean average precision at IoU=0.75:", results.box.map75)
    print("Mean average precision for different IoU thresholds:", results.box.maps)
    print("Mean results for different metrics:", results.box.mean_results)
    print("Mean precision:", results.box.mp)
    print("Mean recall:", results.box.mr)
    print("Precision:", results.box.p)
    print("Precision curve:", results.box.p_curve)
    print("Precision values:", results.box.prec_values)
    print("Specific precision metrics:", results.box.px)
    print("Recall:", results.box.r)
    print("Recall curve:", results.box.r_curve)

if __name__ == "__main__":
    validate_model(model_path='Models/pre_trained_yolov8s.pt', data_path='data.yaml')
