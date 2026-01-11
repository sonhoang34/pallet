from ultralytics import YOLO
import cv2
import numpy as np
import time
import get_image

path_model_detection = r'C:\Lap trinh\realsense\final_project\model\yolo_model\best_2.pt'
path_tracking_yaml = r'C:\Lap trinh\realsense\project\models\botsort.yaml'

def load_model(path_model_detection):
    model = YOLO(path_model_detection, task="detect")
    return model

def detect_pallet(model, img_rgb):
    # Step 2: Prediction với GPU
    results = model.predict(img_rgb, verbose=False, device='cuda:0')

    # Step 3: Extract first detection only
    result = results[0]

    # Check if any detection
    if result.boxes is None or len(result.boxes) == 0:
        return 0, 0, 0, 0, 0.0

    # Get first detection only
    first_box = result.boxes[0]
    x1, y1, x2, y2 = first_box.xyxy[0].cpu().numpy()
    conf = first_box.conf[0].cpu().numpy()

    x_min, y_min, x_max, y_max = map(round, [x1, y1, x2, y2])

    # bbox = [x_min, y_min-5, x_max, y_max]
    
    return x_min, y_min, x_max, y_max, float(conf)


if __name__ == "__main__":
    pipeline, align = get_image.pre()
    list_time = []
    model = load_model(path_model_detection)
    while True:
        rgb, depth_raw, depth_color = get_image.get_frame(pipeline, align)
        start = time.time()
        x_min, y_min, x_max, y_max, conf = detect_pallet(model, rgb)
        end = time.time()
        print(end - start)
        list_time.append(end - start)
        if conf > 0.5:
            cv2.rectangle(rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        cv2.imshow('result', rgb)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            print(np.mean(list_time))
            fps = round(1/np.mean(list_time))
            print("FPS:", fps)
            cv2.destroyAllWindows()
            break
    