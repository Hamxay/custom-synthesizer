import os
import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from vocode.streaming.models.synthesizer import (
    SynthesizerConfig,
    AzureSynthesizerConfig,
)
from vocode.streaming.synthesizer.azure_synthesizer import WordBoundaryEventPool
from vocode.streaming.synthesizer.base_synthesizer import (
    FillerAudio,
    SynthesisResult,
    BaseSynthesizer,
)
from vocode.streaming.models.message import BaseMessage


class FileSynthesizerConfig(SynthesizerConfig):
    file_path: str

    class Config:
        arbitrary_types_allowed = True


class FileSynthesizer(BaseSynthesizer[FileSynthesizerConfig]):
    OFFSET_MS = 100

    def __init__(self, synthesizer_config: FileSynthesizerConfig):
        super().__init__(synthesizer_config)
        self.file_path = synthesizer_config.file_path
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        """
        Generate audio for a given message by retrieving from the specified file path.

        Args:
            message (BaseMessage): Message containing text or SSML to be converted to speech.
            chunk_size (int): The size of each audio chunk to return.
            is_first_text_chunk (bool): Indicates if this is the first chunk of the message.
            is_sole_text_chunk (bool): Indicates if this is the only chunk of the message.

        Returns:
            SynthesisResult: An object containing a generator for streaming the audio.
        """
        # Use the message text as the filename to retrieve
        file_key = message.text.strip()
        audio_data_stream = await self.get_audio_stream(file_key, chunk_size)

        if not audio_data_stream:
            raise FileNotFoundError(f"Audio file for '{file_key}' not found.")

        word_boundary_event_pool = WordBoundaryEventPool()

        async def chunk_generator(audio_stream: io.BytesIO):
            """
            Generator that yields audio chunks from the audio stream.
            """
            while True:
                audio_chunk = audio_stream.read(chunk_size)
                if not audio_chunk:
                    break
                yield SynthesisResult.ChunkResult(
                    audio_chunk, len(audio_chunk) < chunk_size
                )

        return SynthesisResult(
            chunk_generator(audio_data_stream), lambda _: message.text
        )

    async def get_audio_stream(
        self, file_key: str, chunk_size: int
    ) -> Optional[io.BytesIO]:
        """
        Retrieve the audio file from the given file path and return it as a stream.

        Args:
            file_key (str): The file name or key to identify the audio file.
            chunk_size (int): Size of each chunk to be read from the file.

        Returns:
            io.BytesIO: A byte stream of the audio file, or None if not found.
        """
        audio_file_path = os.path.join(self.file_path, file_key)

        # Asynchronously read the file from the given path
        try:
            audio_stream = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool_executor, self.read_file, self.file_path
            )
            return io.BytesIO(audio_stream)
        except FileNotFoundError:
            return None

    @staticmethod
    def read_file(file_path: str) -> bytes:
        """
        Synchronously read the file and return its content.

        Args:
            file_path (str): Path to the file to read.

        Returns:
            bytes: Content of the file as bytes.
        """
        with open(file_path, "rb") as file:
            return file.read()

    def synthesize_ssml(self, ssml: str) -> io.BytesIO:
        """
        No-op for SSML synthesis. Instead, this synthesizer directly reads from files.

        Args:
            ssml (str): SSML string for synthesis.

        Returns:
            io.BytesIO: Placeholder stream.
        """
        raise NotImplementedError("SSML synthesis is not supported in FileSynthesizer.")

    async def get_phrase_filler_audios(self) -> List[FillerAudio]:
        """
        Retrieve filler audio files for predefined phrases.

        Returns:
            List[FillerAudio]: List of filler audio objects.
        """
        # Implement logic to get filler phrase audio if necessary
        return []

    @classmethod
    def from_config(cls, file_path: str):
        """
        Class method to create a FileSynthesizer from a file path.

        Args:
            file_path (str): The base file path where audio files are stored.

        Returns:
            FileSynthesizer: An instance of the FileSynthesizer.
        """
        config = FileSynthesizerConfig(file_path=file_path)
        return cls(config)

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: FileSynthesizerConfig) -> str:
        return ":".join(("file_path", synthesizer_config.file_path))
