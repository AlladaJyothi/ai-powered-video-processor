import os
import cv2
import ffmpeg
import numpy as np
from fastapi import FastAPI, UploadFile, File, Query
from tempfile import NamedTemporaryFile
from fastapi.responses import JSONResponse, FileResponse
from deepface import DeepFace  # DeepFace for emotion detection
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
import google.generativeai as genai

# Initialize FastAPI app
app = FastAPI()

# Set API key for Google Gemini Model
os.environ["GOOGLE_API_KEY"] = "************************************************"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash-001")

# Load AI Models
caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

# Output directory for saving clips
OUTPUT_DIR = "generated_clips"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🎬 Detect Scenes Using SceneDetect
def detect_scenes(video_path):
    video = cv2.VideoCapture(video_path)
    total_duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
    return total_duration

# ✂️ Split Video Using FFmpeg with mood-based emotion detection
def split_video_by_mood(video_path, duration=90):
    output_clips = []
    cap = cv2.VideoCapture(video_path)
    total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    
    # Variables for video processing
    start_time = 0
    clip_index = 1
    moods = ["happy", "sad", "angry","Surprise","Fear","Neutral"]  # Example moods
    current_mood = "happy"  # Default mood
    
    while start_time < total_duration:
        end_time = min(start_time + duration, total_duration)
        clip_filename = os.path.join(OUTPUT_DIR, f"clip_{clip_index}.mp4")

        # Extract frame to detect mood from
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)  # Convert to milliseconds
        success, frame = cap.read()
        
        if not success:
            break
        
        # Use DeepFace to predict mood (emotion detection)
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            # Assuming the first detected emotion as the "mood"
            current_mood = result[0]['dominant_emotion']
        except Exception as e:
            print(f"❌ Emotion detection failed: {e}")
            current_mood = "happy"  # Default to happy if detection fails

        # Update filename to include mood
        clip_filename = os.path.join(OUTPUT_DIR, f"clip_{clip_index}_{current_mood}.mp4")

        # Use FFmpeg to split the video based on detected mood and resize to 9:16 aspect ratio
        ffmpeg.input(video_path, ss=start_time, to=end_time).output(clip_filename, vf="scale=720:1280", vcodec='libx264', acodec='aac').run()

        output_clips.append(clip_filename)
        start_time += duration
        clip_index += 1

    return output_clips

# 🖼️ Generate Thumbnail with Text Overlay
def generate_thumbnail(video_path, text="Short Clip"):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)  # Capture at 1 second
    success, frame = cap.read()
    
    if not success:
        print(f"❌ Failed to generate thumbnail for {video_path}")
        return None

    thumbnail_path = video_path.replace(".mp4", ".jpg")
    cv2.imwrite(thumbnail_path, frame)
    
    # Overlay text on thumbnail
    image = Image.open(thumbnail_path)
    draw = ImageDraw.Draw(image)
    
    font_path = "arial.ttf"  # Ensure this font is available or provide an absolute path
    try:
        font = ImageFont.truetype(font_path, 40)
    except IOError:
        font = ImageFont.load_default()

    text_position = (20, 20)
    draw.text(text_position, text, fill="white", font=font)
    
    image.save(thumbnail_path)
    
    return thumbnail_path

# 🤖 AI-Based Video Summary
def generate_ai_summary(thumbnail_path):
    if not thumbnail_path or not os.path.exists(thumbnail_path):
        print(f"❌ Thumbnail file not found: {thumbnail_path}")
        return "Thumbnail generation failed, AI summary not available."
    
    image = Image.open(thumbnail_path)
    summary = caption_pipeline(image)
    return summary[0]['generated_text']

# 📝 Generate Title & Description using Gemini AI
def generate_title_description(clip_summary):
    prompt = f"Generate an engaging title and description for this video clip: {clip_summary}"
    response = model.generate_content(prompt)
    return response.text

# 📌 API Endpoint to Process Video with Custom Duration and Mood-based Clips
@app.post("/process_video/")
async def process_video(file: UploadFile = File(...), duration: int = Query(20, description="Duration of each short video in seconds")):
    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(file.file.read())
        temp_video_path = temp_video.name  # The path of the saved file

    # Split video into short clips based on mood and duration, with 9:16 aspect ratio
    short_clips = split_video_by_mood(temp_video_path, duration)

    # Generate AI Thumbnails, Titles, and Descriptions for each clip
    response_data = []
    for i, clip in enumerate(short_clips, 1):
        # Generate Thumbnail with Clip Number
        thumbnail = generate_thumbnail(clip, f"Clip {i}")

        if thumbnail is None:
            ai_summary = "Thumbnail generation failed, AI summary not available."
        else:
            ai_summary = generate_ai_summary(thumbnail)

        title_description = generate_title_description(ai_summary)

        response_data.append({
            "clip": clip,
            "thumbnail": thumbnail,
            "title": title_description.split(".")[0],
            "description": title_description
        })

    return JSONResponse(content=response_data)

# 📌 API Endpoint to Download Processed Videos
@app.get("/download/{clip_name}")
async def download_clip(clip_name: str):
    clip_path = os.path.join(OUTPUT_DIR, clip_name)
    if os.path.exists(clip_path):
        return FileResponse(clip_path, media_type='video/mp4', filename=clip_name)
    return JSONResponse(content={"error": "File not found"}, status_code=404)
