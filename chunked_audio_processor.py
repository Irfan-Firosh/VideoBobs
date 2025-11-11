import librosa
import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional


class ChunkedAudioProcessor:
    
    def __init__(self, sample_rate: int = 44100, video_fps: int = 30, smoothing_alpha: float = 0.2):
        self.chunks: List[Dict] = []
        self.current_time: float = 0.0
        self.sample_rate: int = sample_rate
        self.video_fps: int = video_fps
        self.smoothing_alpha: float = smoothing_alpha
        
    def add_chunk(self, speaker_id: int, audio_array: np.ndarray) -> Dict:
        if len(audio_array) == 0:
            print(f"Warning: Empty audio chunk for speaker {speaker_id}")
            return None
            
        rms = librosa.feature.rms(
            y=audio_array,
            frame_length=2048,
            hop_length=512
        )[0]
        
        rms_times = librosa.frames_to_time(
            frames=range(len(rms)),
            sr=self.sample_rate,
            hop_length=512
        )
        
        duration = len(audio_array) / self.sample_rate
        
        chunk = {
            'speaker_id': speaker_id,
            'start_time': self.current_time,
            'end_time': self.current_time + duration,
            'duration': duration,
            'rms': rms,
            'rms_times': rms_times + self.current_time
        }
        
        self.chunks.append(chunk)
        self.current_time += duration
        
        print(f"Added chunk: speaker={speaker_id}, duration={duration:.2f}s, "
              f"energy_range=[{rms.min():.3f}, {rms.max():.3f}]")
        
        return chunk
    
    def build_timeline(self, num_speakers: int = 5) -> Dict:
        if not self.chunks:
            raise ValueError("No chunks added. Add chunks before building timeline.")
        
        total_duration = self.current_time
        
        frame_interval = 1.0 / self.video_fps
        total_frames = int(np.ceil(total_duration * self.video_fps))
        frame_times = np.arange(total_frames) * frame_interval
        
        speaker_energies = {i: np.zeros(total_frames) for i in range(num_speakers)}
        
        for chunk in self.chunks:
            speaker_id = chunk['speaker_id']
            rms = chunk['rms']
            rms_times = chunk['rms_times']
            
            chunk_start_frame = int(np.floor(chunk['start_time'] * self.video_fps))
            chunk_end_frame = int(np.ceil(chunk['end_time'] * self.video_fps))
            chunk_end_frame = min(chunk_end_frame, total_frames)
            
            if chunk_start_frame >= total_frames:
                continue
                
            if len(rms) > 1:
                interpolator = interp1d(
                    rms_times,
                    rms,
                    kind='linear',
                    fill_value=0.0,
                    bounds_error=False
                )
                
                chunk_frame_times = frame_times[chunk_start_frame:chunk_end_frame]
                interpolated_energy = interpolator(chunk_frame_times)
                
                speaker_energies[speaker_id][chunk_start_frame:chunk_end_frame] = interpolated_energy
            else:
                if chunk_start_frame < total_frames:
                    speaker_energies[speaker_id][chunk_start_frame] = rms[0] if len(rms) > 0 else 0.0
        
        for speaker_id in range(num_speakers):
            energy = speaker_energies[speaker_id]
            
            smoothed = np.zeros_like(energy)
            smoothed[0] = energy[0]
            for i in range(1, len(energy)):
                smoothed[i] = self.smoothing_alpha * energy[i] + (1 - self.smoothing_alpha) * smoothed[i-1]
            
            if smoothed.max() > 0:
                normalized = smoothed / smoothed.max()
            else:
                normalized = smoothed
            
            normalized = np.maximum(normalized, 0.1)
            
            speaker_energies[speaker_id] = normalized.tolist()
        
        timeline = {
            'frame_times': frame_times.tolist(),
            'total_frames': total_frames,
            'total_duration': total_duration,
            'speakers': speaker_energies
        }
        
        print(f"Built timeline: {total_frames} frames, {total_duration:.2f}s duration")
        
        return timeline

