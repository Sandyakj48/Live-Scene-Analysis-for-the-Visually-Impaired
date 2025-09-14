"""
nav_assistant.py
Robot-like voice navigation + safety monitor using Google Maps Directions + YOLOv5 + BLIP (optional)

Usage:
    python nav_assistant.py

Speak commands (examples):
    - "navigate to Bangalore City Railway Station"
    - "start navigation to <destination>"
    - "stop navigation"  (to cancel)
    - "describe" (while navigating — returns BLIP caption of current frame)
"""

import os
import time
import json
import requests
import threading
import re

from PIL import Image
import cv2
import numpy as np
import torch

import speech_recognition as sr
import pyttsx3

# Optional BLIP captioning
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------------------------
# Configuration
# -------------------------
# Get Google Maps API key
GOOGLE_MAPS_API_KEY = os.getenv("AIzaSyDRWj8cHH1_1TZ2GVR7z9N8hp_Vk0jj-hA")
if not GOOGLE_MAPS_API_KEY:
    GOOGLE_MAPS_API_KEY = "AIzaSyDRWj8cHH1_1TZ2GVR7z9N8hp_Vk0jj-hA"  # put your valid key here

# Choose device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Device:", DEVICE)

# Load YOLOv5
print("[INFO] Loading YOLOv5 (this may take a while)...")
yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, trust_repo=True)
yolo_model.to(DEVICE)
yolo_model.eval()
print("[INFO] YOLOv5 loaded.")

# Load BLIP (optional)
print("[INFO] Loading BLIP captioning model (optional, may take time)...")
try:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model.to(DEVICE)
    blip_model.eval()
    BLIP_AVAILABLE = True
    print("[INFO] BLIP loaded.")
except Exception as e:
    print("[WARN] BLIP failed to load (describe command disabled):", e)
    BLIP_AVAILABLE = False

# Text-to-Speech (server-side)
tts = pyttsx3.init()
tts.setProperty("rate", 150)

def speak(text: str):
    """Speak text using pyttsx3."""
    if not text:
        return
    print("[TTS]", text)
    try:
        tts.say(text)
        tts.runAndWait()
    except Exception as e:
        print("[WARN] TTS error:", e)

# -------------------------
# Speech recognition helper
# -------------------------
recognizer = sr.Recognizer()
try:
    mic = sr.Microphone()
    print("[INFO] Microphone ready.")
except Exception as e:
    print("[WARN] Microphone not available:", e)
    mic = None

def listen_once(timeout=8, phrase_time_limit=5):
    """Listen once and return recognized lowercased string (or empty on failure)."""
    if mic is None:
        return ""
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("[SR] Listening for command...")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            cmd = recognizer.recognize_google(audio)
            cmd = cmd.lower().strip()
            print("[SR] Recognized:", cmd)
            return cmd
        except sr.WaitTimeoutError:
            print("[SR] Listen timeout")
            return ""
        except sr.UnknownValueError:
            print("[SR] Could not understand")
            return ""
        except sr.RequestError as e:
            print("[SR] Request error:", e)
            return ""

# -------------------------
# Google Maps Directions helper
# -------------------------
def get_directions(origin=None, destination=None, mode="walking"):
    """Query Google Directions API."""
    if not destination:
        return None, "No destination provided"

    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "key": GOOGLE_MAPS_API_KEY,
        "destination": destination,
        "mode": mode,
    }
    if origin:
        params["origin"] = origin

    print("[INFO] Requesting directions for:", destination)
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    data = r.json()
    if data.get("status") != "OK":
        return None, data.get("status", "UNKNOWN")
    try:
        steps = data["routes"][0]["legs"][0]["steps"]
        return steps, None
    except Exception as e:
        return None, str(e)

# -------------------------
# Camera + YOLO safety monitor
# -------------------------
cap = None
def start_camera():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap is not None and cap.isOpened()

def release_camera():
    global cap
    if cap:
        cap.release()
        cap = None

def capture_frame():
    """Return an RGB PIL image from webcam or None."""
    if cap is None:
        started = start_camera()
        if not started:
            return None
    ret, frame = cap.read()
    if not ret:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return pil

def yolo_counts_from_pil(pil_image):
    """Run YOLO on PIL image, return counts left/center/right + object positions."""
    np_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    results = yolo_model(np_img)
    df = results.pandas().xyxy[0]
    width, _ = pil_image.size
    left_count = right_count = center_count = 0
    objects_with_dir = []
    for _, row in df.iterrows():
        name = str(row["name"])
        xmin = float(row["xmin"])
        xmax = float(row["xmax"])
        xcenter = (xmin + xmax) / 2.0
        if xcenter < width / 3.0:
            dirn = "left"; left_count += 1
        elif xcenter > 2 * width / 3.0:
            dirn = "right"; right_count += 1
        else:
            dirn = "center"; center_count += 1
        objects_with_dir.append(f"{name} on the {dirn}")
    return left_count, right_count, center_count, objects_with_dir

# -------------------------
# BLIP describe helper
# -------------------------
def blip_describe(pil_image):
    if not BLIP_AVAILABLE:
        return "Description not available."
    try:
        inputs = blip_processor(pil_image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out_ids = blip_model.generate(**inputs, max_new_tokens=30)
        caption = blip_processor.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        return caption
    except Exception as e:
        print("[WARN] BLIP describe failed:", e)
        return "Unable to describe the scene."

# -------------------------
# Navigation loop
# -------------------------
stop_navigation_flag = threading.Event()

def navigation_run(steps, safety_check_interval=0.6):
    """Iterate through navigation steps with YOLO safety checks."""
    for i, step in enumerate(steps):
        if stop_navigation_flag.is_set():
            speak("Navigation stopped.")
            return
        instr_html = step.get("html_instructions", "")
        instr_text = re.sub("<.*?>", "", instr_html)
        dist_text = step.get("distance", {}).get("text", "")
        speak(f"Step {i+1}: {instr_text}. {dist_text}.")
        print(f"[NAV] Step {i+1}: {instr_text} ({dist_text})")

        step_monitor_seconds = max(8, int(step.get("duration", {}).get("value", 8)))
        start_time = time.time()
        clear_count_needed = 3
        clear_counter = 0

        while time.time() - start_time < step_monitor_seconds:
            if stop_navigation_flag.is_set():
                speak("Navigation stopped.")
                return
            pil = capture_frame()
            if pil is None:
                print("[WARN] No camera frame")
                time.sleep(safety_check_interval)
                continue
            left_c, right_c, center_c, objs_with_dir = yolo_counts_from_pil(pil)
            print(f"[DETECT] L{left_c} C{center_c} R{right_c} -> {objs_with_dir[:3]}")

            if center_c > 0:
                speak("Obstacle ahead. Please stop.")
                while True:
                    if stop_navigation_flag.is_set():
                        speak("Navigation stopped.")
                        return
                    pil2 = capture_frame()
                    if pil2 is None:
                        time.sleep(safety_check_interval)
                        continue
                    _, _, c2, _ = yolo_counts_from_pil(pil2)
                    if c2 == 0:
                        clear_counter += 1
                    else:
                        clear_counter = 0
                    if clear_counter >= clear_count_needed:
                        speak("Path is clear. Continue.")
                        break
                    time.sleep(safety_check_interval)
            time.sleep(safety_check_interval)

    speak("Navigation complete. You have arrived at your destination.")

# -------------------------
# Main loop
# -------------------------
def main_loop():
    print("Voice controls: 'navigate to <place>', 'stop navigation', 'describe', 'exit'")
    speak("System ready. Say navigate to followed by destination to begin.")
    while True:
        cmd = listen_once(timeout=10, phrase_time_limit=6)
        if not cmd:
            continue
        if "exit" in cmd or "quit" in cmd:
            speak("Exiting system. Goodbye.")
            break
        if "stop navigation" in cmd or cmd.strip() == "stop":
            stop_navigation_flag.set()
            speak("Stopping navigation.")
            continue
        if cmd.startswith(("navigate to", "start navigation to", "navigate")):
            dest = cmd
            for prefix in ["navigate to", "start navigation to", "navigate"]:
                if dest.startswith(prefix):
                    dest = dest[len(prefix):].strip()
                    break
            if not dest:
                speak("Please tell the destination after 'navigate to'.")
                continue
            speak(f"Getting directions to {dest}.")
            steps, err = get_directions(destination=dest, mode="walking")
            if err:
                speak(f"Could not get directions: {err}")
                continue
            stop_navigation_flag.clear()
            nav_thread = threading.Thread(target=navigation_run, args=(steps,), daemon=True)
            nav_thread.start()
            continue
        if "describe" in cmd and BLIP_AVAILABLE:
            pil = capture_frame()
            if pil is None:
                speak("Camera not available.")
            else:
                desc = blip_describe(pil)
                speak(desc)
            continue
        speak("Command not recognized. Say navigate to destination, describe, or stop navigation.")

if __name__ == "__main__":
    try:
        started = start_camera()
        if not started:
            print("[WARN] Could not access camera at startup. Will retry when needed.")
        main_loop()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        release_camera()
