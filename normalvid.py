from cartesia import Cartesia
from cartesia.voices.types.gender import Gender
import os
import random
import dotenv
import io
import numpy as np
import librosa

dotenv.load_dotenv()

SAMPLE_RATE = 44100

class AudioProcessor:

    def __init__(self, script: dict, output_file: str):
        self.script = script
        self.client = Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))
        self.output_file = output_file
        self.voice = None
    

    def setRandomMaleVoice(self):
        voices = list(self.client.voices.list(limit=20, gender="masculine"))
        self.voice = random.choice(voices)
    

    def setRandomFemaleVoice(self):
        voices = list(self.client.voices.list(limit=20, gender="feminine"))
        self.voice = random.choice(voices)

    def generateAudioChunk(self, text):
        if self.voice is None:
            raise ValueError("Voice not set!")
        
        chunk_iter = self.client.tts.bytes(
            model_id="sonic-3",
            transcript=text,
            voice={
                "mode": "id",
                "id": self.voice.id
            },
            output_format={
                "container": "wav",
                "sample_rate": SAMPLE_RATE,
                "encoding": "pcm_f32le",
            }
        )
        return chunk_iter

    
    @staticmethod
    def wav_bytes_to_numpy(wav_bytes):
        try:
            with io.BytesIO(wav_bytes) as wav_file:
                array, sr = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)
                return array
        except Exception as e:
            print(f"Error converting WAV bytes to numpy array: {e}")
            raise

    
    @staticmethod
    def wav_bytes_to_numpy_from_file(file_path):
        try:
            array, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            return array
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            raise

        
    
    def processChunks(self):
        for item in self.script['script']:
            chunk_iter = self.generateAudioChunk(item['dialogue'])
            chunks = []
            for chunk in chunk_iter:
                chunks.append(chunk)
            wav_bytes = b''.join(chunks)
            numpy_array = self.wav_bytes_to_numpy(wav_bytes)
            yield wav_bytes, numpy_array

class AudioGenerator:

    def __init__(self, script: dict, output_file: str):
        self.script = script
        self.client = Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))
        self.output_file = output_file
    

    def getRandomMaleVoice(self):
        voices = list(self.client.voices.list(limit=20, gender="masculine"))
        return random.choice(voices)
    

    def getRandomFemaleVoice(self):
        voices = list(self.client.voices.list(limit=20, gender="feminine"))
        return random.choice(voices)

    def generateAudioChunk(self, voice, text):
        chunk_iter = self.client.tts.bytes(
            model_id="sonic-3",
            transcript=text,
            voice={
                "mode": "id",
                "id": voice.id
            },
            output_format={
                "container": "wav",
                "sample_rate": 44100,
                "encoding": "pcm_f32le",
            }
        )
        return chunk_iter

    
    def generateAudio(self, chunk_iter):
        with open(self.output_file, "wb") as f:
            for chunk in chunk_iter:
                f.write(chunk)

if __name__ == "__main__":
    audioGenerator = AudioGenerator({"script": "Hello, how are you?"}, "output.wav")
    voice = audioGenerator.getRandomMaleVoice()
    chunk = audioGenerator.generateAudioChunk(voice, "Hello, how are you?")
    audioGenerator.generateAudio(chunk)

