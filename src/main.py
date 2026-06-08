import os
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from scenedetect import detect, ContentDetector

input_path = "data/raw/seahawks_rams_game.mp4"
diagnostic_output_path = "data/output/diagnostics/ai_annotated_seahawks_rams_game.mp4"
highlight_output_path = "data/output/seahawks_rams_highlights.mp4"  

print("Analyzing video for camera cuts...")
scene_list = detect(input_path, ContentDetector())
scenes = [(scene[0].frame_num, scene[1].frame_num) for scene in scene_list]

print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(diagnostic_output_path, fourcc, fps, (frame_width, frame_height))

# State Management
track_memory = {}
frame_count = 0
current_scene_idx = 0

# Temporal Buffer (6 seconds of pre-roll)
PRE_ROLL_SECONDS = 6
BUFFER_MAX_SIZE = PRE_ROLL_SECONDS * fps
frame_buffer = deque(maxlen=BUFFER_MAX_SIZE)

all_game_highlights = []  
current_play_frames = []   

highlight_triggered = False
trigger_frame = 0
POST_ROLL_SECONDS = 3
post_roll_frames_needed = POST_ROLL_SECONDS * fps  

print(f"\n--- Running ApexClip Compilation Engine ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Cache a pristine copy before we draw tracking boxes on it
    frame_buffer.append(frame.copy())
    
    # Track current camera scene boundaries
    scene_start_frame = 1
    if current_scene_idx < len(scenes):
        scene_start_frame, end_frame = scenes[current_scene_idx]
        if frame_count > end_frame:
            current_scene_idx += 1
            print(f"🎬 [Frame {frame_count}] Camera Cut! Wiping tracking vectors.")
            track_memory.clear()
            if hasattr(model, 'predictor') and model.predictor.trackers:
                model.predictor.trackers[0].reset()
            if current_scene_idx < len(scenes):
                scene_start_frame, end_frame = scenes[current_scene_idx]

    # Run AI tracking pipeline
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.15, verbose=False)[0]
    current_frame_velocities = []

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = box
            class_name = model.names[class_id]

            if class_name in ["person", "sports ball"]:
                current_center_x = int((x1 + x2) / 2)
                current_center_y = int((y1 + y2) / 2)
                
                velocity = 0.0
                if track_id in track_memory:
                    prev_center_x, prev_center_y = track_memory[track_id]
                    velocity = np.sqrt((current_center_x - prev_center_x)**2 + (current_center_y - prev_center_y)**2)
                    current_frame_velocities.append(velocity)
                
                track_memory[track_id] = (current_center_x, current_center_y)

                # Overlay box decorations on the master debug stream
                color_index = track_id % 5
                box_color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)][color_index]
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"ID:{track_id} {class_name} | Vel: {velocity:.1f}px"
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Compilation Logic
    if len(current_frame_velocities) > 0:
        frame_avg_velocity = np.mean(current_frame_velocities)
        
        # Condition A: Motion explodes, initiate clip accumulation
        if frame_avg_velocity > 45.0 and not highlight_triggered:
            print(f"🔥 HIGHLIGHT DETECTED at frame {frame_count}! (Motion: {frame_avg_velocity:.1f}px)")
            highlight_triggered = True
            trigger_frame = frame_count
            
            # Safety Wall equation: We can't look back further than the scene start frame
            max_allowed_lookback = frame_count - scene_start_frame
            target_lookback = min(len(frame_buffer), max_allowed_lookback)
            
            buffer_list = list(frame_buffer)
            current_play_frames = buffer_list[-target_lookback:]
            print(f"   ↳ Software Time Machine buffered {len(current_play_frames)} pre-snap frames.")

        # Condition B: Actively gathering post-play footage
        elif highlight_triggered:
            current_play_frames.append(frame.copy())
            
            # Append this play to our master reel array
            if frame_count >= trigger_frame + post_roll_frames_needed:
                print(f"➕ Appending play to master memory array. Total frames in this sequence: {len(current_play_frames)}")
                
                # Merge current play frames list into our master game array list
                all_game_highlights.extend(current_play_frames)
                
                # Reset temporary variables to search for next play
                highlight_triggered = False
                current_play_frames = []

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

#Export
if len(all_game_highlights) > 0:
    print(f"\n🎬 Exporting Final Compiled Master Reel...")
    print(f"   ↳ Total Compiled Frame Count: {len(all_game_highlights)}")
    print(f"   ↳ Saving to: {highlight_output_path}")
    
    highlight_writer = cv2.VideoWriter(highlight_output_path, fourcc, fps, (frame_width, frame_height))
    for h_frame in all_game_highlights:
        highlight_writer.write(h_frame)
    highlight_writer.release()
    
    print("🏆 Game Highlights File Successfully Generated!")
else:
    print("\n⚠️ Processing finished, but no highlights matched the motion thresholds.")

print(f"Master processing thread completed to {diagnostic_output_path}")