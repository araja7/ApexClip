import cv2
from ultralytics import YOLO

input_path = "data/raw/sample_video.mp4"
output_path = "data/output/tracking_output.mp4"

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
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
print(f"Running persistent tracking on: {input_path}")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    
    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, confidence, class_id in zip(boxes, track_ids, confidences, class_ids):
            x1, y1, x2, y2 = box
            class_name = model.names[class_id]

            if class_name in ["person", "sports ball"]:
                color_index = track_id % 5
                colors = [
                    (0, 255, 0),    #Green
                    (255, 0, 0),    #Blue
                    (0, 0, 255),    #Red
                    (0, 255, 255),  #Yellow
                    (255, 0, 255)   #Magenta
                ]
                box_color = colors[color_index]

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"ID: {track_id} {class_name} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        out.write(frame)
            



cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Finished processing! Tracking output saved to {output_path}")

