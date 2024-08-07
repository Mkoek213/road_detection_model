# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
#
# def live_predict(model_path, setting, wait_key, classNames, video_path=None):
#     """
#     Perform live object detection using YOLO model.
#
#     Parameters:
#     - model_path (str): Path to the YOLO model weights file.
#     - setting (str): Mode of operation, either 'live' for webcam or 'static' for video file.
#     - wait_key (int): Time in milliseconds to wait between frames. A value of 0 means wait indefinitely.
#     - classNames (list of str): List of class names that the model has been trained to recognize.
#     - video_path (str, optional): Path to the video file for 'static' setting. Required if setting is 'static'.
#
#     Raises:
#     - ValueError: If 'setting' is not 'live' or 'static', or if 'video_path' is not provided for 'static' setting.
#     """
#
#     # Initialize video capture based on the setting
#     if setting == 'live':
#         # For live webcam feed
#         cap = cv2.VideoCapture(0)  # Open default webcam
#         cap.set(3, 640)  # Set the width of the frame to 640 pixels
#         cap.set(4, 480)  # Set the height of the frame to 480 pixels
#     elif setting == 'static':
#         # For video file
#         if video_path is None:
#             raise ValueError("In 'static' setting you must pass video_path")
#         cap = cv2.VideoCapture(video_path)  # Load video file
#     else:
#         # Raise an error if setting is invalid
#         raise ValueError(f"Invalid setting '{setting}'. Expected 'live' or 'static'.")
#
#     # Load the YOLO model from the specified path
#     model = YOLO(model_path)
#
#     # Define specific colors for selected classes
#     classColors = {
#         "different traffic sign": (255, 100, 50),  # Blue
#         "pedestrian": (128, 0, 128),  # Purple
#         "car": (0, 255, 0),  # Green
#         "truck": (255, 165, 0),  # Orange
#         "warning sign": (0, 255, 255),  # Yellow
#         "prohibition sign": (0, 0, 255),  # Red
#         "pedestrian crossing": (173, 216, 230),  # Light Blue
#         "speed limit sign": (255, 192, 203)  # Pink
#     }
#
#     # Define colors for remaining classes
#     remaining_colors = {
#         "dark green": (0, 100, 0),  # Dark Green
#         "dark yellow": (255, 255, 0)  # Dark Yellow
#     }
#
#     # Assign colors to the remaining classes
#     remaining_color_list = list(remaining_colors.values())
#     for i, cls in enumerate(classNames):
#         if cls not in classColors:
#             classColors[cls] = remaining_color_list[i % len(remaining_color_list)]
#
#     while True:
#         # Read a frame from the video capture
#         success, img = cap.read()
#         if not success:
#             break  # End of video or cannot read frame
#
#         # Perform object detection on the current frame
#         results = model(img, stream=True)
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 # Extract bounding box coordinates and convert to integers
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#                 # Get the color for the bounding box based on the detected class
#                 cls = classNames[int(box.cls[0])]
#                 color = classColors.get(cls, (255, 255, 255))  # Default to white if class not found in color map
#
#                 # Draw a thin rectangle around the detected object
#                 cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Thickness set to 2 for thin rectangles
#
#                 # Calculate the confidence score and format it
#                 conf = math.floor(box.conf[0] * 100) / 100
#
#                 # Display class name and confidence score
#                 cvzone.putTextRect(img, f"{cls}", (max(0, x1), max(35, y1)), scale=1.5, thickness=2, offset=3, colorR=color, colorT=(0, 0, 0))
#
#         # Display the resulting frame in a window
#         cv2.imshow("Image", img)
#
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(wait_key) & 0xFF == ord('q'):
#             break
#
#     # Release the video capture and close all OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     # Define class names for different settings
#     class_names_pretrained = [
#         "person", "rider", "car", "bus", "truck", "bike", "motor", "tl_green", "tl_red", "tl_yellow",
#         "tl_none", "traffic sign", "train", "tl_green"
#     ]
#
#     class_names_finetuned = [
#         "car", "different traffic sign", "green traffic light", "motorcycle", "pedestrian", "pedestrian crossing",
#         "prohibition sign", "red traffic light", "speed limit sign", "truck", "warning sign"
#     ]
#
#     # Run the live_predict function with the fine-tuned model and specified settings
#     live_predict(
#         model_path='Models/fine_tuned_yolov8s.pt',
#         setting='static',
#         wait_key=5,
#         classNames=class_names_finetuned,
#         video_path='test_images/video1.webm'
#     )





from ultralytics import YOLO
import cv2
import cvzone
import math

def live_predict(model_path, setting, wait_key, classNames, video_path=None, output_path='output_video.avi'):
    """
    Perform live object detection using YOLO model and save the video with annotations.

    Parameters:
    - model_path (str): Path to the YOLO model weights file.
    - setting (str): Mode of operation, either 'live' for webcam or 'static' for video file.
    - wait_key (int): Time in milliseconds to wait between frames. A value of 0 means wait indefinitely.
    - classNames (list of str): List of class names that the model has been trained to recognize.
    - video_path (str, optional): Path to the video file for 'static' setting. Required if setting is 'static'.
    - output_path (str): Path where the output video with annotations will be saved.
    """

    # Initialize video capture based on the setting
    if setting == 'live':
        # For live webcam feed
        cap = cv2.VideoCapture(0)  # Open default webcam
        cap.set(3, 640)  # Set the width of the frame to 640 pixels
        cap.set(4, 480)  # Set the height of the frame to 480 pixels
    elif setting == 'static':
        # For video file
        if video_path is None:
            raise ValueError("In 'static' setting you must pass video_path")
        cap = cv2.VideoCapture(video_path)  # Load video file
    else:
        # Raise an error if setting is invalid
        raise ValueError(f"Invalid setting '{setting}'. Expected 'live' or 'static'.")

    # Load the YOLO model from the specified path
    model = YOLO(model_path)

    # Define specific colors for selected classes
    classColors = {
        "different traffic sign": (255, 100, 50),  # Blue
        "pedestrian": (128, 0, 128),  # Purple
        "car": (0, 255, 0),  # Green
        "truck": (255, 165, 0),  # Orange
        "warning sign": (0, 255, 255),  # Yellow
        "prohibition sign": (0, 0, 255),  # Red
        "pedestrian crossing": (173, 216, 230),  # Light Blue
        "speed limit sign": (255, 192, 203)  # Pink
    }

    # Define colors for remaining classes
    remaining_colors = {
        "dark green": (0, 100, 0),  # Dark Green
        "dark yellow": (255, 255, 0)  # Dark Yellow
    }

    # Assign colors to the remaining classes
    remaining_color_list = list(remaining_colors.values())
    for i, cls in enumerate(classNames):
        if cls not in classColors:
            classColors[cls] = remaining_color_list[i % len(remaining_color_list)]

    # Get video properties for output file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize VideoWriter to save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        # Read a frame from the video capture
        success, img = cap.read()
        if not success:
            break  # End of video or cannot read frame

        # Perform object detection on the current frame
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract bounding box coordinates and convert to integers
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get the color for the bounding box based on the detected class
                cls = classNames[int(box.cls[0])]
                color = classColors.get(cls, (255, 255, 255))  # Default to white if class not found in color map

                # Draw a thin rectangle around the detected object
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Thickness set to 2 for thin rectangles

                # Calculate the confidence score and format it
                conf = math.floor(box.conf[0] * 100) / 100

                # Display class name and confidence score
                cvzone.putTextRect(img, f"{cls}", (max(0, x1), max(35, y1)), scale=1.5, thickness=2, offset=3, colorR=color, colorT=(0, 0, 0))

        # Write the frame with annotations to the output video
        out.write(img)

        # Display the resulting frame in a window
        cv2.imshow("Image", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(wait_key) & 0xFF == ord('q'):
            break

    # Release the video capture and output files, and close all OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Define class names for different settings
    class_names_pretrained = [
        "person", "rider", "car", "bus", "truck", "bike", "motor", "tl_green", "tl_red", "tl_yellow",
        "tl_none", "traffic sign", "train", "tl_green"
    ]

    class_names_finetuned = [
        "car", "different traffic sign", "green traffic light", "motorcycle", "pedestrian", "pedestrian crossing",
        "prohibition sign", "red traffic light", "speed limit sign", "truck", "warning sign"
    ]

    # Run the live_predict function with the fine-tuned model and specified settings
    live_predict(
        model_path='Models/fine_tuned_yolov8s.pt',
        setting='static',
        wait_key=5,
        classNames=class_names_finetuned,
        video_path='test_images/Film59 ‐ Wykonano za pomocą Clipchamp.mp4',
        output_path='output_video.avi'
    )
