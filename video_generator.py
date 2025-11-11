import os
import cv2
import numpy as np
from moviepy import VideoFileClip, AudioFileClip
from typing import Dict, List, Tuple, Optional


class VideoGenerator:
    
    def __init__(
        self,
        timeline: Dict,
        audio_path: str,
        video_fps: int = 30,
        width: int = 1920,
        height: int = 1080,
        base_radius: int = 80,
        max_scale: float = 1.5
    ):
        self.timeline = timeline
        self.audio_path = audio_path
        self.fps = video_fps
        self.width = width
        self.height = height
        self.base_radius = base_radius
        self.max_scale = max_scale
        
        self.num_speakers = len(timeline['speakers'])
        self.positions = self._calculate_positions()
        self.colors = self._generate_colors()
        
        self.bg_color = (26, 26, 26)
        
    def _calculate_positions(self) -> List[Tuple[int, int]]:
        positions = []
        center_x = self.width // 2
        center_y = self.height // 2
        
        radius = min(self.width, self.height) // 4
        
        if self.num_speakers == 1:
            positions.append((center_x, center_y))
        else:
            angles = np.linspace(np.pi, 0, self.num_speakers)
            
            for angle in angles:
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                positions.append((x, y))
        
        return positions
    
    def _generate_colors(self) -> List[Tuple[int, int, int]]:
        colors = []
        
        for i in range(self.num_speakers):
            hue = int(180 * i / max(self.num_speakers, 1))
            
            hsv_color = np.uint8([[[hue, 200, 255]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            
            colors.append(tuple(map(int, bgr_color)))
        
        return colors
    
    def generate_fluid_outline(
        self,
        center_x: int,
        center_y: int,
        base_radius: float,
        energy: float,
        frame_idx: int,
        num_points: int = 80
    ) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        
        phase = frame_idx * 0.1
        
        wobble = (
            np.sin(angles * 3 + phase) * 0.1 +
            np.sin(angles * 5 + phase * 1.3) * 0.05 +
            np.sin(angles * 7 + phase * 0.7) * 0.03
        )
        
        energy_scale = 1.0 + (energy - 0.1) * (self.max_scale - 1.0) / 0.9
        
        radius = base_radius * energy_scale + wobble * base_radius * energy
        
        x = center_x + radius * np.cos(angles)
        y = center_y + radius * np.sin(angles)
        
        points = np.column_stack([x, y]).astype(np.int32)
        return points
    
    def draw_bob(
        self,
        frame: np.ndarray,
        position: Tuple[int, int],
        energy: float,
        color: Tuple[int, int, int],
        frame_idx: int
    ) -> None:
        center_x, center_y = position
        
        outline_points = self.generate_fluid_outline(
            center_x, center_y, self.base_radius, energy, frame_idx
        )
        
        cv2.fillPoly(frame, [outline_points], color)
        
        inner_color = tuple(max(0, c - 30) for c in color)
        inner_radius = int(self.base_radius * 0.7 * (1.0 + (energy - 0.1) * 0.5))
        cv2.circle(frame, (center_x, center_y), inner_radius, inner_color, -1)
        
        if energy > 0.3:
            glow_radius = int(self.base_radius * 1.2 * (1.0 + (energy - 0.1) * 0.5))
            glow_color = tuple(min(255, c + 20) for c in color)
            cv2.circle(frame, (center_x, center_y), glow_radius, glow_color, 2)
    
    def render(self, output_path: str) -> None:
        total_frames = self.timeline['total_frames']
        abs_output_path = os.path.join(os.getcwd(), output_path) if not os.path.isabs(output_path) else output_path
        temp_video_path = abs_output_path.replace('.mp4', '_no_audio.mp4')
        
        print(f"Rendering {total_frames} frames at {self.fps} fps...")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            temp_video_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {temp_video_path}")
        
        for frame_idx in range(total_frames):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:] = self.bg_color
            
            for speaker_id in range(self.num_speakers):
                energy = self.timeline['speakers'][speaker_id][frame_idx]
                position = self.positions[speaker_id]
                color = self.colors[speaker_id]
                
                self.draw_bob(frame, position, energy, color, frame_idx)
            
            video_writer.write(frame)
            
            if (frame_idx + 1) % (total_frames // 10) == 0:
                progress = (frame_idx + 1) / total_frames * 100
                print(f"Progress: {progress:.1f}% ({frame_idx + 1}/{total_frames} frames)")
        
        video_writer.release()
        print(f"Video rendered (no audio): {temp_video_path}")
        
        print("Combining video with audio...")
        try:
            audio_path = os.path.join(os.getcwd(), self.audio_path) if not os.path.isabs(self.audio_path) else self.audio_path
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise ValueError(f"Audio file is empty: {audio_path}")
            
            print(f"Loading video: {temp_video_path}")
            video_clip = VideoFileClip(temp_video_path)
            print(f"Video duration: {video_clip.duration:.2f}s")
            
            print(f"Loading audio: {audio_path} ({file_size} bytes)")
            audio_clip = AudioFileClip(audio_path)
            print(f"Audio duration: {audio_clip.duration:.2f}s, fps: {audio_clip.fps}")
            
            duration_diff = abs(audio_clip.duration - video_clip.duration)
            print(f"Duration difference: {duration_diff:.2f}s")
            
            if duration_diff > 0.1:
                if audio_clip.duration > video_clip.duration:
                    print(f"Trimming audio from {audio_clip.duration:.2f}s to {video_clip.duration:.2f}s")
                    audio_clip = audio_clip.subclip(0, video_clip.duration)
                elif audio_clip.duration < video_clip.duration:
                    from moviepy.audio.AudioClip import CompositeAudioClip
                    silence_duration = video_clip.duration - audio_clip.duration
                    print(f"Extending audio with {silence_duration:.2f}s of silence")
                    silence = audio_clip.subclip(0, 0.01).volumex(0).set_duration(silence_duration)
                    audio_clip = CompositeAudioClip([audio_clip, silence.set_start(audio_clip.duration)])
            
            print("Combining video and audio...")
            final_clip = video_clip.with_audio(audio_clip)
            
            print(f"Final clip has audio: {final_clip.audio is not None}")
            if final_clip.audio:
                print(f"Final clip audio duration: {final_clip.audio.duration:.2f}s")
            
            print(f"Writing final video to: {abs_output_path}")
            final_clip.write_videofile(
                abs_output_path,
                codec='libx264',
                audio_codec='aac',
                fps=self.fps,
                bitrate='5000k',
                audio_bitrate='192k',
                logger=None
            )
            
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
            print(f"Final video saved: {abs_output_path}")
            
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                print("Cleaned up temporary video file")
                
        except Exception as e:
            import traceback
            print(f"Error combining audio: {e}")
            traceback.print_exc()
            print(f"Video without audio saved at: {temp_video_path}")
            raise

