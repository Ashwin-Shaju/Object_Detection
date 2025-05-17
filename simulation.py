import cv2
import numpy as np
import datetime
import os
import time
import asyncio
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import FSInputFile
import tkinter as tk
from tkinter import filedialog
import threading
import winsound  

last_notification_time = 0 

TELEGRAM_BOT_TOKEN = "<YOUR_TOKEN>"
TELEGRAM_CHAT_IDS = ["<YOUR_CHAT_ID>"] 

# Initialize Aiogram Bot with correct parameters
bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create directories
detected_objects_dir = os.path.join(script_dir, "detected_objects")
os.makedirs(detected_objects_dir, exist_ok=True)

output_video_dir = os.path.join(script_dir, "output_videos")
os.makedirs(output_video_dir, exist_ok=True)

log_file = os.path.join(script_dir, "detection_log.txt")

# Event loop for the thread
thread_loop = asyncio.new_event_loop()

def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

# Start the event loop in a separate thread
threading.Thread(target=start_loop, args=(thread_loop,), daemon=True).start()

async def send_telegram_notification(message, image_path=None):
    """Sends an alert message and image to multiple Telegram chat IDs"""
    try:
        for chat_id in TELEGRAM_CHAT_IDS:
            if image_path:
                photo = FSInputFile(image_path)
                await bot.send_photo(chat_id=chat_id, photo=photo, caption=message)
            else:
                await bot.send_message(chat_id=chat_id, text=message)
        print("Telegram notifications sent successfully!")
    except Exception as e:
        print(f"Failed to send Telegram notifications: {str(e)}")
import random
def play_beep():
    """Plays a Windows beep sound"""
    try:
        winsound.Beep(1000, 500)  # Frequency: 1000Hz, Duration: 500ms
    except Exception as e:
        print(f"Could not play beep sound: {str(e)}")

def log_detection(detected_objects, confidences, frame, boxes, class_ids, classes):
    """Logs detected objects and sends notifications for all detections"""
    global last_notification_time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_entries = []

    for i, obj in enumerate(detected_objects):
        x, y, w, h = boxes[i]
        confidence = confidences[i]
        
        # Log all detections
        log_entry = f"[{timestamp}] {obj} detected, Confidence: {confidence:.2f}, Location: (x: {x}, y: {y}, w: {w}, h: {h})\n"
        log_entries.append(log_entry)
        
        # Save the full frame image for all detections
        image_path = os.path.join(detected_objects_dir, f"{obj}_detected_{timestamp}_{i}.jpg")
        cv2.imwrite(image_path, frame)

        # Check if 5 seconds have passed since the last notification
        current_time = time.time()
        if current_time - last_notification_time >= 5:
            last_notification_time = current_time
            random_distance = random.randint(850, 900)  # Random distance between 850-900m
            alert_message = f"🚨 OBJECT DETECTED 🚨\n{obj} detected on tracks!\nDistance: {random_distance}m\nConfidence: {confidence:.2f}%"
            asyncio.run_coroutine_threadsafe(send_telegram_notification(alert_message, image_path), thread_loop)
            play_beep()  # Play beep sound when object is detected

    if log_entries:
        with open(log_file, "a") as f:
            f.writelines(log_entries)
def load_yolo():
    """Loads YOLOv4-tiny model"""
    net = cv2.dnn.readNet(r"D:\\Python\\yolov4-tiny.weights", r"D:\\Python\\yolov4-tiny.cfg")

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    with open(r"D:\\Python\\coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, classes, output_layers

def detect_objects(img, net, output_layers):
    """Runs object detection using YOLOv4-tiny"""
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (224, 224), (0, 0, 0), True, crop=False) #A "blob" is a 4D array (batch, channels, height, width) 
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs, height, width

def get_box_dimensions(outs, height, width):
    """Extracts bounding box dimensions and class IDs"""
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2) #topleft corner equation
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return class_ids, confidences, boxes

def draw_labels(boxes, confidences, class_ids, classes, img):
    """Draws bounding boxes and labels with yellow borders for all objects"""
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    detected_objects = []

    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            x, y, w, h = boxes[i]
            
            # Draw yellow border for all objects
            color = (0, 255, 255)  # Yellow for all objects
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
            detected_objects.append(label)

    if detected_objects:
        log_detection(detected_objects, confidences, img, boxes, class_ids, classes)

    return img

def simulate_video(video_path):
    """Runs object detection on video at the original speed"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    global net, classes, output_layers, paused
    net, classes, output_layers = load_yolo()

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    paused = False
    video_ended = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Video has ended. Stopping notifications...")
                video_ended = True
                break

            outs, height, width = detect_objects(frame, net, output_layers)
            class_ids, confidences, boxes = get_box_dimensions(outs, height, width)
            
            if not video_ended:
                frame = draw_labels(boxes, confidences, class_ids, classes, frame)

            cv2.imshow("Railway Track Object Detection", frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            print("Video stopped manually.")
            video_ended = True
            break
        elif key == ord(' '):
            paused = not paused
            print("Video Paused" if paused else "Video Resumed")

    print("Press 'Q' to close the window.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

async def main():
    """Main function to select video and start detection"""
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )

    if not video_path:
        print("No video file selected. Exiting...")
        return

    simulate_video(video_path)

if __name__ == "__main__":
    asyncio.run(main())
