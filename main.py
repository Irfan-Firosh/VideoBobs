import os
import sys
import random
import numpy as np
from typing import List, Tuple
from pydub import AudioSegment

from generator.video.normalvid import AudioProcessor
from .chunked_audio_processor import ChunkedAudioProcessor
from .video_generator import VideoGenerator


class TalkingBobsPipeline:
    
    def __init__(
        self,
        sample_rate: int = 44100,
        video_fps: int = 30,
        output_file: str = "talking_bobs.mp4"
    ):
        self.sample_rate = sample_rate
        self.video_fps = video_fps
        self.output_file = os.path.join(os.getcwd(), output_file)
        
    def process_conversation(
        self,
        conversation: List[Tuple[int, str]],
        temp_audio_dir: str = "public/temp_audio"
    ) -> str:
        temp_audio_dir = os.path.join(os.getcwd(), temp_audio_dir)
        os.makedirs(temp_audio_dir, exist_ok=True)
        
        num_speakers = max(speaker_id for speaker_id, _ in conversation) + 1
        
        print(f"Processing conversation with {num_speakers} speakers, {len(conversation)} turns")
        
        print("\n=== Step 1: Generating audio ===")
        audio_processors = {}
        audio_chunks_data = []
        
        for speaker_id in range(num_speakers):
            script = {'script': []}
            audio_processor = AudioProcessor(script, os.path.join(temp_audio_dir, f"speaker_{speaker_id}.wav"))
            
            if random.random() < 0.5:
                audio_processor.setRandomMaleVoice()
            else:
                audio_processor.setRandomFemaleVoice()
            
            audio_processors[speaker_id] = audio_processor
            print(f"Speaker {speaker_id}: Voice ID = {audio_processor.voice.id}")
        
        audio_chunk_files = []
        
        for turn_idx, (speaker_id, text) in enumerate(conversation):
            print(f"Turn {turn_idx + 1}/{len(conversation)}: Speaker {speaker_id} - \"{text[:50]}...\"")
            
            audio_processor = audio_processors[speaker_id]
            
            chunk_file_path = os.path.join(temp_audio_dir, f"chunk_{turn_idx:04d}.wav")
            
            chunk_iter = audio_processor.generateAudioChunk(text)
            with open(chunk_file_path, 'wb') as f:
                for chunk in chunk_iter:
                    f.write(chunk)
            
            print(f"  Saved chunk to: {chunk_file_path}")
            
            numpy_array = audio_processor.wav_bytes_to_numpy_from_file(chunk_file_path)
            
            audio_chunk_files.append({
                'speaker_id': speaker_id,
                'file_path': chunk_file_path,
                'numpy_array': numpy_array
            })
        
        print("\n=== Step 2: Merging audio ===")
        merged_audio_path = os.path.join(temp_audio_dir, "merged_audio.wav")
        self._merge_audio_files(audio_chunk_files, merged_audio_path)
        print(f"Merged audio saved: {merged_audio_path}")
        
        print("\n=== Step 3: Extracting features ===")
        chunk_processor = ChunkedAudioProcessor(
            sample_rate=self.sample_rate,
            video_fps=self.video_fps
        )
        
        for chunk_data in audio_chunk_files:
            chunk_processor.add_chunk(
                chunk_data['speaker_id'],
                chunk_data['numpy_array']
            )
        
        timeline = chunk_processor.build_timeline(num_speakers=num_speakers)
        print(f"Timeline built: {timeline['total_frames']} frames, {timeline['total_duration']:.2f}s")
        
        print("\n=== Step 4: Rendering video ===")
        video_generator = VideoGenerator(
            timeline=timeline,
            audio_path=merged_audio_path,
            video_fps=self.video_fps
        )
        
        video_generator.render(self.output_file)
        print(f"\n✅ Complete! Video saved: {self.output_file}")
        print(f"Note: Audio files kept in {temp_audio_dir} for debugging")
        
        return self.output_file
    
    def _merge_audio_files(
        self,
        audio_chunk_files: List[dict],
        output_path: str
    ) -> None:
        import subprocess
        import tempfile
        
        file_paths = [chunk_data['file_path'] for chunk_data in audio_chunk_files]
        
        valid_files = [fp for fp in file_paths if os.path.exists(fp)]
        
        if not valid_files:
            raise ValueError("No valid audio files to merge")
        
        print(f"  Merging {len(valid_files)} segments using ffmpeg...")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            concat_file = f.name
            for file_path in valid_files:
                abs_path = os.path.abspath(file_path)
                f.write(f"file '{abs_path}'\n")
        
        try:
            result = subprocess.run(
                [
                    'ffmpeg',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-c', 'copy',
                    '-y',
                    output_path
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"ffmpeg error: {result.stderr}")
                print("Falling back to pydub method...")
                self._merge_audio_files_pydub(audio_chunk_files, output_path)
            else:
                print(f"  Successfully merged using ffmpeg: {output_path}")
        finally:
            if os.path.exists(concat_file):
                os.remove(concat_file)
    
    def _merge_audio_files_pydub(
        self,
        audio_chunk_files: List[dict],
        output_path: str
    ) -> None:
        audio_segments = []
        
        for chunk_data in audio_chunk_files:
            file_path = chunk_data['file_path']
            
            if not os.path.exists(file_path):
                print(f"Warning: Audio file not found: {file_path}")
                continue
            
            try:
                audio_segment = AudioSegment.from_wav(file_path)
                print(f"  Loaded: {file_path} ({len(audio_segment)}ms, {audio_segment.frame_rate}Hz, {audio_segment.channels}ch, {audio_segment.sample_width}sw)")
                audio_segments.append(audio_segment)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not audio_segments:
            raise ValueError("No valid audio segments to merge")
        
        print(f"  Merging {len(audio_segments)} segments with pydub...")
        
        first_seg = audio_segments[0]
        print(f"  First segment: {first_seg.frame_rate}Hz, {first_seg.channels}ch, {first_seg.sample_width}sw")
        
        for i, seg in enumerate(audio_segments[1:], 1):
            if seg.frame_rate != first_seg.frame_rate:
                print(f"  Warning: Segment {i} has different sample rate: {seg.frame_rate}Hz")
            if seg.channels != first_seg.channels:
                print(f"  Warning: Segment {i} has different channels: {seg.channels}")
            if seg.sample_width != first_seg.sample_width:
                print(f"  Warning: Segment {i} has different sample width: {seg.sample_width}")
        
        merged = sum(audio_segments)
        
        print(f"  Merged audio: {len(merged)}ms, {merged.frame_rate}Hz, {merged.channels}ch, {merged.sample_width}sw")
        
        merged.export(
            output_path, 
            format="wav",
            parameters=["-ar", str(merged.frame_rate)]
        )
        print(f"  Exported merged audio to: {output_path}")


def main():
    conversation = [
        (0, "Hello everyone, welcome to the discussion."),
        (1, "Thanks for having me!"),
        (2, "I'd like to add something here."),
        (0, "Please go ahead."),
        (1, "That's a great point."),
        (2, "I completely agree with that."),
    ]
    
    pipeline = TalkingBobsPipeline(output_file="talking_bobs.mp4")
    
    output_path = pipeline.process_conversation(conversation)
    
    print(f"\n🎉 Video generation complete: {output_path}")


