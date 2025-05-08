from ultralytics import YOLO
TARGET_CLASSES = {3, 4}

model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)

# Lấy kích thước video
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Dictionary lưu kết quả theo từng frame
tracking_results = {}
res_info_car = {}
frame_id = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break 
    print("frame: ", frame_id)
    
    results = model.track(frame, persist=True, device="cpu")

    for result in results:
        if result.boxes is not None:
            xywhn = result.boxes.xywhn.tolist() if result.boxes.xywhn is not None else []
            track_ids = result.boxes.id.tolist() if result.boxes.id is not None else []
            cls_ids = result.boxes.cls.tolist() if result.boxes.cls is not None else []
            for box, track_id, cls_id in zip(xywhn, track_ids, cls_ids):
                cls_id = int(cls_id) 

                if cls_id not in TARGET_CLASSES:
                    continue

                x_center, y_center, width, height = box[:4]
                x_center, y_center = int(x_center * frame_w), int(y_center * frame_h)
                width, height = int(width * frame_w), int(height * frame_h)
                x_left, y_top = x_center - width // 2, y_center - height // 2
                x_right, y_bottom = x_left + width, y_top + height
                object_crop = frame[y_top:y_bottom, x_left:x_right].copy()

                if cls_id == 3:
                    if res_info_car == {}:
                        res_info_car = {
                            "x_center": x_center, "y_center": y_center,
                            "left": round(x_left / frame_w, 4), "top": round(y_top / frame_h, 4),
                            "frame_w": frame_w, "frame_h": frame_h,
                            "width": width, "height": height,
                            "x-left": x_left, "y-left": y_top
                        }
                if track_id not in tracking_results:
                    tracking_results[track_id] = {}
                
                tracking_results[track_id][frame_id]={
                    "class_id": cls_id,
                    "bbox": {
                        "x_center": x_center, "y_center": y_center,
                        "left": round(x_left / frame_w, 4), "top": round(y_top / frame_h, 4),
                        "frame_w": frame_w, "frame_h": frame_h,
                        "width": width, "height": height,
                        "x-left": x_left, "y-left": y_top
                    },
                    "object_crop": object_crop
                }

    frame_id += 1
cap.release()
