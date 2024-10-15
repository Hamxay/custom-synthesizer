import os
from typing import Any, List, Tuple
import wave
import asyncio
from pydub import AudioSegment
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
)
from vocode.streaming.models.synthesizer import SynthesizerConfig
from pydantic.v1 import BaseModel as Pydantic1BaseModel


# Base model handling TypedModel
class BaseModel(Pydantic1BaseModel):
    def __init__(self, **data):
        for key, value in data.items():
            if isinstance(value, dict):
                if "type" in value and key != "action_trigger":
                    data[key] = TypedModel.parse_obj(value)
            if isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, dict):
                        if "type" in v:
                            value[i] = TypedModel.parse_obj(v)
        super().__init__(**data)


# Adapted from https://github.com/pydantic/pydantic/discussions/3091
class TypedModel(BaseModel):
    _subtypes_: List[Tuple[Any, Any]] = []

    def __init_subclass__(cls, type=None):  # type: ignore
        cls._subtypes_.append((type, cls))

    @classmethod
    def get_cls(_cls, type):
        for t, cls in _cls._subtypes_:
            if t == type:
                return cls
        raise ValueError(f"Unknown type {type}")

    @classmethod
    def get_type(_cls, cls_name):
        for t, cls in _cls._subtypes_:
            if cls.__name__ == cls_name:
                return t
        raise ValueError(f"Unknown class {cls_name}")

    @classmethod
    def parse_obj(cls, obj):
        data_type = obj.get("type")
        if data_type is None:
            raise ValueError(f"type is required for {cls.__name__}")

        sub = cls.get_cls(data_type)
        if sub is None:
            raise ValueError(f"Unknown type {data_type}")
        return sub(**obj)

    def _iter(self, **kwargs):
        yield "type", self.get_type(self.__class__.__name__)
        yield from super()._iter(**kwargs)

    @property
    def type(self):
        return self.get_type(self.__class__.__name__)


# Configuration model for FileSynthesizer
class FileSynthesizerConfig(BaseModel):
    file_path: str  # Path to the pre-downloaded audio file
    chunk_size: int = 1024  # Size of each audio chunk to stream
    audio_encoding: str = "LINEAR16"  # Define audio encoding format
    sampling_rate: int = 16000  # Default sampling rate


# AudioChunk class to wrap chunks of audio data
class AudioChunk:
    """Simple class to represent an audio chunk."""

    def __init__(self, chunk):
        self.chunk = chunk


# FileSynthesizer class for streaming audio from a file
class FileSynthesizer(BaseSynthesizer[FileSynthesizerConfig]):
    def __init__(self, synthesizer_config: FileSynthesizerConfig):
        super().__init__(synthesizer_config)
        self.file_path = synthesizer_config.file_path
        self.chunk_size = synthesizer_config.chunk_size
        self.sampling_rate = synthesizer_config.sampling_rate
        self.sample_width = 2  # For LINEAR16, sample width is 2 bytes
        self.is_streaming = False  # Track if streaming is active

        # Convert to WAV if necessary
        self.wav_file_path = self._convert_to_wav_if_needed(self.file_path)

    def _convert_to_wav_if_needed(self, file_path):
        """Convert non-WAV audio files to WAV format using pydub."""
        if file_path.endswith(".wav"):
            return file_path  # If already a WAV file, return as is

        # Otherwise, convert it to WAV
        audio = AudioSegment.from_file(file_path)
        wav_file_path = os.path.splitext(file_path)[0] + ".wav"
        audio.export(wav_file_path, format="wav")
        return wav_file_path

    async def stream_audio(self):
        """Stream the audio file in chunks."""
        if not os.path.exists(self.wav_file_path):
            raise FileNotFoundError(f"Audio file not found: {self.wav_file_path}")

        self.is_streaming = True  # Set streaming to active

        with wave.open(self.wav_file_path, "rb") as wf:
            frames_per_chunk = self.chunk_size // self.sample_width
            print("frames_per_chunk:", frames_per_chunk, flush=True)

            while True:
                if not self.is_streaming:
                    break  # Stop streaming if set to False (end stream)

                data = wf.readframes(frames_per_chunk)
                if not data:
                    break  # End of file

                yield AudioChunk(data)

                # Calculate the duration for the chunk and sleep to match real-time playback
                duration_per_chunk = frames_per_chunk / self.sampling_rate
                await asyncio.sleep(duration_per_chunk)

        print("Audio stream ended.")

    def end_stream(self):
        """End the current audio stream."""
        print("Ending current audio stream.")
        self.is_streaming = False  # Set to False to stop streaming

    async def create_new_stream(self, new_file_path=None):
        """Create a new audio stream (from the same or a new file)."""
        if new_file_path:
            self.file_path = new_file_path
            self.wav_file_path = self._convert_to_wav_if_needed(new_file_path)

        return SynthesisResult(
            self.stream_audio(), lambda seconds: "New pre-recorded audio"
        )

    async def create_speech_uncached(
        self, message, chunk_size, is_first_text_chunk=False, is_sole_text_chunk=False
    ):
        """This function is overridden to return the audio from the file."""
        return self.get_current_utterance_synthesis_result()

    def get_current_utterance_synthesis_result(self):
        """Return a generator that streams AudioChunk objects."""
        return SynthesisResult(
            self.stream_audio(), lambda seconds: "Pre-recorded audio"
        )

    async def send_token_to_synthesizer(self, message, chunk_size):
        """No token-to-speech streaming needed for file playback."""
        return None

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: FileSynthesizerConfig):
        return []
