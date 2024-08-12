import cv2
import librosa
import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip


# Load the audio file
y, sr = librosa.load("audio.wav")

# Compute the Short-Time Fourier Transform (STFT) to get the intensity
stft = np.abs(librosa.stft(y))**2

# Convert amplitude to decibels
energy_db = librosa.amplitude_to_db(np.sum(stft, axis=0))

# Define a threshold to detect talking
threshold = np.mean(energy_db) + 8  # Adjust the threshold as needed

# Detect talking (energy above threshold)
talking = energy_db > threshold

# Ensure you have the correct number of frames
fps = 24  # Adjust FPS as needed
num_frames = int(len(y) / sr * fps)


# Ensure talking array has the correct number of frames
if len(talking) < num_frames:
    # Extend the array if needed
    talking = np.concatenate([talking, np.zeros(num_frames - len(talking))])
    
    
# Define mouth images
mouth_images = {
    "closed": cv2.imread("avatar_mouth_closed.png"),
    "semi-open": cv2.imread("avatar_mouth_semi_open.png"),
    "open": cv2.imread("avatar_mouth_open.png")
}

def get_mouth_image(index):
    # Loop through the three mouth images
    return mouth_images[list(mouth_images.keys())[index % 3]]

# Generate frames
frames = []
for i in range(num_frames):
    if talking[i]:
        # Loop through 3 mouth shapes
        mouth_image = get_mouth_image((i // (num_frames // 3)) % 3)  # Adjust speed of looping
        avatar = cv2.imread("avatar_face.png")  # Base avatar image
        x_offset, y_offset = 300, 310  # Example offsets, modify as needed
        avatar[y_offset:y_offset+mouth_image.shape[0], x_offset:x_offset+mouth_image.shape[1]] = mouth_image
    else:
        # Use a neutral frame when not talking
        avatar = cv2.imread("avatar_face.png")

    frames.append(avatar)

# Ensure frames are of consistent size and format
frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]


# Create a video clip from the generated frames
video_clip = ImageSequenceClip(frames, fps=fps)

# Load the original audio file
audio = AudioFileClip("audio.wav")

# Set the audio to the video clip
video_clip = video_clip.set_audio(audio)

# Write the final video file
video_clip.write_videofile("output_video.mp4", codec="libx264")