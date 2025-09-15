import cv2
import math
from ultralytics import YOLO
import threading
import pygame
import numpy as np

# --- Configuration ---
# Set the video source (0 for webcam)
VIDEO_SOURCE = 0
# VIDEO_SOURCE = "path/to/your/video.mp4"

# Set the maximum time (in frames) a bag can be stationary before being flagged
# 20 seconds * 30 frames/second = 600 frames
ABANDONED_FRAME_THRESHOLD = 600

# Set the minimum distance (in pixels) a person must be from their bag
OWNER_DISTANCE_THRESHOLD = 150

# Set the path to your alert sound file
ALERT_SOUND_PATH = "alert.mp3"

# --- Load the pre-trained YOLOv8 model ---
model = YOLO('yolov8n.pt')

# --- Initialization ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source '{VIDEO_SOURCE}'.")
    exit()

# Store state for tracking and abandonment logic
tracked_objects = {}
bag_assignments = {}
alert_playing = False
abandoned_frames_count = 0

# Initialize pygame mixer for sound playback
pygame.mixer.init()
try:
    pygame.mixer.music.load(ALERT_SOUND_PATH)
except pygame.error as e:
    print(f"Pygame error loading sound file: {e}")
    print(f"Ensure '{ALERT_SOUND_PATH}' is a valid audio file in the correct directory.")
    exit()

# Function for playing the alert sound in a separate thread
def play_alert_sound_threaded():
    global alert_playing
    if not alert_playing:
        alert_playing = True
        print("ALERT: Abandoned luggage detected! Playing sound...")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        alert_playing = False

# Function to draw a text box with a background
def put_text_with_bg(img, text, position, font, scale, color, bg_color, thickness):
    x, y = position
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x, y - 5), font, scale, color, thickness)

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection and tracking
    results = model.track(frame, persist=True, classes=[0, 24, 26, 28])
    abandoned_detected = False

    if results and results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().tolist()
        class_ids = results[0].boxes.cls.int().tolist()
        boxes = results[0].boxes.xyxy.int().tolist()
        
        current_persons = []
        current_bags = []
        
        for i, track_id in enumerate(track_ids):
            class_id = class_ids[i]
            bbox = boxes[i]
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            class_name = model.names[class_id]

            if track_id in tracked_objects:
                tracked_objects[track_id]['position'] = center
                if class_name in ['backpack', 'handbag', 'suitcase'] and math.hypot(center[0] - tracked_objects[track_id]['position'][0], center[1] - tracked_objects[track_id]['position'][1]) < 5:
                    tracked_objects[track_id]['abandoned_timer'] += 1
                else:
                    tracked_objects[track_id]['abandoned_timer'] = 0
            else:
                tracked_objects[track_id] = {'class': class_name, 'position': center, 'abandoned_timer': 0, 'is_abandoned': False}

            if class_name == 'person':
                current_persons.append({'id': track_id, 'center': center})
            elif class_name in ['backpack', 'handbag', 'suitcase']:
                current_bags.append({'id': track_id, 'center': center})
        
        for bag in current_bags:
            bag_id = bag['id']
            bag_center = bag['center']
            
            closest_person_id = None
            min_distance = float('inf')
            
            for person in current_persons:
                distance = math.hypot(person['center'][0] - bag_center[0], person['center'][1] - bag_center[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_person_id = person['id']

            if min_distance < OWNER_DISTANCE_THRESHOLD:
                bag_assignments[bag_id] = closest_person_id
            elif bag_id in bag_assignments:
                if closest_person_id not in [p['id'] for p in current_persons]:
                    tracked_objects[bag_id]['is_abandoned'] = True
            
        for i, track_id in enumerate(track_ids):
            class_name = model.names[class_ids[i]]
            bbox = boxes[i]
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            color = (0, 255, 0)
            label = f"{class_name} ({track_id})"
            
            if class_name in ['backpack', 'handbag', 'suitcase']:
                if tracked_objects.get(track_id, {}).get('abandoned_timer', 0) > ABANDONED_FRAME_THRESHOLD:
                    tracked_objects[track_id]['is_abandoned'] = True
                
                if tracked_objects.get(track_id, {}).get('is_abandoned'):
                    color = (0, 0, 255)
                    label = f"ABANDONED {class_name} ({track_id})"
                    abandoned_detected = True
                    
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            put_text_with_bg(frame, label, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), (0, 0, 0), 1)

    if abandoned_detected:
        abandoned_frames_count += 1
    else:
        abandoned_frames_count = 0

    if abandoned_frames_count > 0:
        alert_text = "!!! ABANDONED LUGGAGE DETECTED !!!"
        put_text_with_bg(frame, alert_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), (0, 0, 255), 2)
        if not alert_playing:
            alert_thread = threading.Thread(target=play_alert_sound_threaded)
            alert_thread.start()

    cv2.imshow('Abandoned Luggage Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
