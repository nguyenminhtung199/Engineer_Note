import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import glob
import time 
video_dir = ""
video_paths = glob.glob(video_dir)
save_folder = os.path.dirname(video_dir).split('/')[-1]
print(save_folder)
# exit()
print(len(video_paths))
num_frames_desired = 2000 
# lst_video_processed = [os.path.basename(path) for path in glob.glob('crop_object/*')]
# print(lst_video_processed)
for video_path in video_paths: 
    # print(os.path.basename(video_path).split('.')[0])
    if os.path.basename(video_path).split('.')[0] in lst_video_processed:
        print("Processed")
        continue
    # continue
    try:
        model = YOLO("yolov8s.pt")
        names = model.names
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error reading video file"
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # skipframe = max(1, total_frames // num_frames_desired)
        skipframe = 10 
        print("Skip frame = ", skipframe)
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        os.makedirs(save_folder, exist_ok=True)
        crop_dir_name = os.path.splitext(os.path.basename(video_path))[0]
        crop_dir_name = save_folder + "/" + crop_dir_name
        if not os.path.exists(crop_dir_name):
            os.mkdir(crop_dir_name)

        # Video writer
        # video_writer = cv2.VideoWriter("object_cropping_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        idx = 0
        people_class = [0]
        frame_id = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            # im0 = cv2.resize(im0, (w, h)) 

            if frame_id % skipframe == 0:
                results = model.track(im0, persist=True, device = "cpu")
                for result in results: 
                    if result.boxes.id is not None:
                        track_ids = result.boxes.id.int().cpu().tolist()
                        bboxes = result.boxes.xyxy.cpu().numpy()  # lấy bounding boxes từ YOLO
                        clss = result.boxes.cls.cpu().tolist()

                        for track_id, bbox, cls in zip(track_ids, bboxes, clss):
                            # Vẽ bounding box và mask
                            # color = colors(int(track_id), True)
                            # txt_color = annotator.get_txt_color(color)
                            # annotator.seg_bbox(mask=mask, mask_color=color, label=str(track_id), txt_color=txt_color)
                            if int(cls) in people_class: 
                                class_dir = os.path.join(crop_dir_name, 'people')
                                os.makedirs(class_dir, exist_ok=True)
                            # elif int(cls) in vehical_class:
                            #     class_dir = os.path.join(crop_dir_name, 'vehical')
                            #     os.makedirs(class_dir, exist_ok=True)
                            else:
                                continue
                            obj_dir = class_dir + f'/{track_id}/'
                            os.makedirs(obj_dir, exist_ok=True)
                            x1, y1, x2, y2 = map(int, bbox)
                            
                            cropped_obj = im0[y1: y2, x1: x2]
                            save_path = os.path.join(obj_dir, f"{frame_id}.jpg")
                        
                            cv2.imwrite(save_path, cropped_obj)

            # Tăng frame_id cho khung hình tiếp theo
            print("Frame: ", frame_id)

            frame_id += 1

        cap.release()
    except Exception as e: 
        os.makedirs('video_error', exist_ok=True)
        # raise e
        with open(f"video_error/{os.path.basename(video_path)}.txt", 'w') as file:
            print()
        print(e)
        continue
    # video_writer.release()
