import torch
from torchvision import models, transforms
import cv2
import time
import concurrent.futures
from queue import Queue
from threading import Lock
import numpy as np
import subprocess
import os
from datetime import datetime
from utils import format_time
import configparser
import re

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Load pre-trained Faster R-CNN model
model_name = config['Model']['model_name']
pretrained = config['Model'].getboolean('pretrained')
model = models.detection.__dict__[model_name](pretrained=pretrained)
model.eval()

# Load COCO category names from config file
try:
    COCO_INSTANCE_CATEGORY_NAMES = config['COCO_Categories']['categories'].split(',')
    COCO_INSTANCE_CATEGORY_NAMES = [name.strip() for name in COCO_INSTANCE_CATEGORY_NAMES] #remove extra whitespace
except KeyError:
    print("Error: 'categories' key not found in [COCO_Categories] section of config.ini")
    exit(1)


# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


threshold = config['Model'].getfloat('threshold')
target_label_id = config['Model'].getint('target_label_id')

def process_frame(frame, model, transform, target_label_id, threshold=threshold):
    frame_tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(frame_tensor)
    people_boxes = []
    for pred in range(len(predictions[0]['labels'])):
        label_id = predictions[0]['labels'][pred].item()
        if label_id == target_label_id and predictions[0]['scores'][pred] > threshold:
            people_boxes.append(predictions[0]['boxes'][pred].cpu().numpy())
    return people_boxes

try:
    interval_str = config['VideoProcessing']['interval']
    interval_cleaned = re.sub(r'[^\d.]', '', interval_str)
    interval = float(interval_cleaned)
except (KeyError, ValueError) as e:
    print(f"Error reading or parsing 'interval' from config.ini: {e}")
    print("Using default interval of 1 second.")
    interval = 1.0

def frame_generator(video_path, target_width=int(config['VideoProcessing']['target_width']), target_height=int(config['VideoProcessing']['target_height']), interval=interval):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_interval = int(fps * interval)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_width, target_height))
        yield int(cap.get(cv2.CAP_PROP_POS_FRAMES)), frame
        if interval > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frames_per_interval - 1)

    cap.release()


def estimate_remaining_time(start_time, processed_frames, total_frames, fps):
    elapsed_time = time.time() - start_time
    avg_time_per_frame = elapsed_time / processed_frames if processed_frames > 0 else 0
    remaining_frames = total_frames - processed_frames
    remaining_time = remaining_frames * avg_time_per_frame
    return remaining_time

def calculate_motion(previous_boxes, current_boxes, movement_threshold=int(config['VideoProcessing']['movement_threshold'])):
    def box_center(box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    if not previous_boxes or not current_boxes:
        return False

    for prev_box in previous_boxes:
        prev_center = box_center(prev_box)
        for curr_box in current_boxes:
            curr_center = box_center(curr_box)
            distance = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
            if distance > movement_threshold:
                return True
    return False

def convert_to_mp4(video_path):
    base, ext = os.path.splitext(video_path)
    output_path = base + ".mp4"
    ffmpeg_path = config['FFmpeg']['ffmpeg_path']

    if os.path.exists(output_path):
        print(f"MP4 file {output_path} already exists. Skipping conversion.")
        return output_path

    try:
        subprocess.run([ffmpeg_path, '-i', video_path, '-c:v', 'h264_nvenc', '-preset', 'medium', '-crf', '23', output_path, '-y'], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e}")
        return None

def detect_people_in_video(video_path, model, transform, target_label_id, interval=float(config['VideoProcessing']['interval']), num_threads=int(config['VideoProcessing']['num_threads']), threshold=threshold):

    if video_path.lower().endswith(".mts"):
        print("MTS file detected. Converting to MP4...")
        converted_path = convert_to_mp4(video_path)
        if converted_path:
            video_path = converted_path
        else:
            print("Failed to convert MTS file. Exiting.")
            return -1, -1, -1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.basename(video_path).split('.')[0]
    output_file_prefix = config['Paths']['output_file_prefix']
    output_dir = "output"  # Specify the output directory
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    output_file = os.path.join(output_dir, f"{video_name}_{timestamp}_{output_file_prefix}txt") #Join the directory and file name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    detected_times = 0
    active_times = 0
    lock = Lock()
    processed_frames = 0
    previous_boxes = None
    progress_lock = Lock()
    start_time = time.time()
    last_person_detected_time = 0
    person_detected_intervals = []

    def write_results_to_file(f, detected_times, active_times, person_detected_intervals, video_duration):
        detected_percentage = (detected_times / video_duration) * 100 if video_duration > 0 else 0
        active_percentage = (active_times / video_duration) * 100 if video_duration > 0 else 0
        f.write(f"\nTotal detected time: {detected_times} seconds ({format_time(detected_times)}), ({detected_percentage:.2f}% of video duration).\n")
        f.write(f"Total active time: {active_times} seconds ({format_time(active_times)}), ({active_percentage:.2f}% of video duration).\n")
        for start, end in person_detected_intervals:
            start_time_str = format_time(start / fps)
            end_time_str = format_time(end / fps)
            f.write(f"Person detected from {start_time_str} to {end_time_str}\n")


    with open(output_file, "w") as f:
        def worker(frame_queue):
            nonlocal processed_frames, detected_times, active_times, previous_boxes, last_person_detected_time, person_detected_intervals
            while True:
                try:
                    frame_num, frame = frame_queue.get(timeout=1)
                    current_boxes = process_frame(frame, model, transform, target_label_id, threshold)
                    if current_boxes:
                        with lock:
                            if last_person_detected_time == 0:
                                last_person_detected_time = frame_num
                            detected_times += interval
                            timestamp = format_time(frame_num / fps)
                            f.write(f"Person detected at {timestamp}, duration: {interval} seconds\n")
                        if previous_boxes and calculate_motion(previous_boxes, current_boxes):
                            with lock:
                                active_times += interval
                                f.write(f"Active person detected at {timestamp}, duration: {interval} seconds\n")
                        previous_boxes = current_boxes
                    else:
                        if last_person_detected_time != 0:
                            with lock:
                                person_detected_intervals.append((last_person_detected_time, frame_num))
                                last_person_detected_time = 0
                    with progress_lock:
                        processed_frames += 1
                    frame_queue.task_done()
                except Queue.Empty:
                    break

        frame_queue = Queue(maxsize=num_threads * 10)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(num_threads):
                executor.submit(worker, frame_queue)

            try:
                for frame_num, frame in frame_generator(video_path, interval=interval):
                    frame_queue.put((frame_num, frame))
                    with progress_lock:
                        current_progress = (processed_frames / total_frames) * 100
                        remaining_time = estimate_remaining_time(start_time, processed_frames, total_frames, fps)
                        remaining_time_formatted = format_time(remaining_time)
                        print(f"Processing frames: {processed_frames}/{total_frames} ({current_progress:.2f}%), Estimated time remaining: {remaining_time_formatted}", end='\r')
            except Exception as e:
                print(f"An error occurred during processing: {e}")
                return -1, -1, -1

        frame_queue.join()
        write_results_to_file(f, detected_times, active_times, person_detected_intervals, video_duration)

    print(f"\nProcessing frames: {processed_frames}/{total_frames} (100.00%)")
    print(f"Video analysis completed. Output written to {output_file}")

    return detected_times, active_times, fps