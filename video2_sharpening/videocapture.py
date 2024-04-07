import cv2
import os
from datetime import datetime, timedelta

def capture_and_save_frames(video_path, times, output_folder):
    with open('starttime.txt', 'r') as f:
        start_time_str = f.readline().strip() 
        start_time = datetime.strptime(start_time_str, "%m-%d-%H-%M") 
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    for idx, time in enumerate(times):
        frame_number = int(time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            interval = timedelta(minutes=30*idx) 
            current_time = start_time + interval
            output_filename = current_time.strftime("%m-%d %H-%M") + ".jpg"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, frame)
            print(f"Saved {output_path}")
        else:
            print(f"Failed to capture frame at {time} seconds.")

    cap.release()
