import cv2
import librosa
import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip

def load_audio(audio_path):
    """Load the audio file and return the samples and sample rate."""
    y, sr = librosa.load(audio_path)
    return y, sr

def calculate_talking_frames(y, sr, fps, threshold=8):
    """Determine which frames contain talking based on the audio energy."""
    duration = librosa.get_duration(y=y, sr=sr)
    num_frames = int(duration * fps)
    samples_per_frame = int(sr / fps)
    talking = np.zeros(num_frames, dtype=bool)
    
    for i in range(num_frames):
        start_sample = i * samples_per_frame
        end_sample = start_sample + samples_per_frame
        frame_samples = y[start_sample:end_sample]
        frame_energy = np.sum(frame_samples**2)
        frame_energy_db = librosa.amplitude_to_db([frame_energy])[0]
        
        if frame_energy_db > threshold:
            talking[i] = True

    return talking, num_frames

def load_mouth_images():
    """Load the mouth images for different mouth positions."""
    mouth_images = {
        "closed": cv2.imread("avatar_mouth_closed.png"),
        "semi-open": cv2.imread("avatar_mouth_semi_open.png"),
        "open": cv2.imread("avatar_mouth_open.png")
    }
    return mouth_images

def generate_frames(talking, num_frames, mouth_images, mouth_y_offset=310, mouth_x_offset=300):
    """Generate avatar frames based on talking frames and mouth images."""
    frames = []
    
    for i in range(num_frames):
        if talking[i]:
            if i == 0 or not talking[i-1]:
                mouth_image = mouth_images["closed"]
            elif i > 1 and talking[i-1] and talking[i-2]:
                previous_frame_open = (
                    frames[-1][mouth_y_offset:mouth_y_offset+mouth_images["open"].shape[0], 
                               mouth_x_offset:mouth_x_offset+mouth_images["open"].shape[1]] == mouth_images["open"]).all()
                previous_frame_open_2 = (
                    frames[-2][mouth_y_offset:mouth_y_offset+mouth_images["open"].shape[0], 
                               mouth_x_offset:mouth_x_offset+mouth_images["open"].shape[1]] == mouth_images["open"]).all()
                if previous_frame_open and previous_frame_open_2:
                    mouth_image = mouth_images["semi-open"]
                else:
                    mouth_image = mouth_images["open"]
            else:
                mouth_image = mouth_images["semi-open"]
            
            avatar = cv2.imread("avatar_face.png")
            avatar[mouth_y_offset:mouth_y_offset+mouth_image.shape[0], mouth_x_offset:mouth_x_offset+mouth_image.shape[1]] = mouth_image
        else:
            avatar = cv2.imread("avatar_face.png")

        frames.append(avatar)
    
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    return frames

def create_video(frames, fps, audio_path, output_path="output_video.mp4"):
    """Create the final video with the generated frames and audio."""
    video_clip = ImageSequenceClip(frames, fps=fps)
    audio = AudioFileClip(audio_path)
    video_clip = video_clip.set_audio(audio)
    video_clip.write_videofile(output_path, codec="libx264")

def create_talking_avatar_video(audio_path, output_path="output_video.mp4", fps=24, threshold=8):
    """Main function to generate the talking avatar video from an audio file."""
    y, sr = load_audio(audio_path)
    talking, num_frames = calculate_talking_frames(y, sr, fps, threshold)
    mouth_images = load_mouth_images()
    frames = generate_frames(talking, num_frames, mouth_images)
    create_video(frames, fps, audio_path, output_path)