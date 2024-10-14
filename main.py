import asyncio
import os
import signal

from pydantic import Field
import pygame

from pydantic_settings import BaseSettings, SettingsConfigDict
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from file_synthesizer import FileSynthesizer, FileSynthesizerConfig
from vocode.helpers import create_streaming_microphone_input_and_speaker_output

from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig

from vocode.streaming.models.message import BaseMessage

from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer

from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber

# Configure logging
configure_pretty_logging()


# Custom MP3 Synthesizer
# class MP3SynthesizerConfig(SynthesizerConfig):
#     file_path: str = Field(..., description="Path to the MP3 file")

#     @classmethod
#     def from_output_device(cls, speaker_output, file_path):
#         return cls(
#             file_path=file_path,
#             sampling_rate=speaker_output.sampling_rate,
#             audio_encoding=speaker_output.audio_encoding,
#         )


# class MP3Synthesizer(BaseSynthesizer):
#     def __init__(self, config: MP3SynthesizerConfig):
#         super().__init__(config)
#         self.file_path = config.file_path
#         pygame.mixer.init(frequency=config.sampling_rate)

#     async def create_speech(self, text: str):
#         pygame.mixer.music.load(self.file_path)
#         pygame.mixer.music.play()
#         while pygame.mixer.music.get_busy():
#             await asyncio.sleep(0.1)


# # Settings for the streaming conversation
# class Settings(BaseSettings):
#     deepgram_api_key: str = "cb1396052363179025b6fbee81c3ff9c54098086"

#     # This means a .env file can be used to overload these settings
#     # ex: "OPENAI_API_KEY=my_key" will set openai_api_key over the default above
#     model_config = SettingsConfigDict(
#         env_file=".env",
#         env_file_encoding="utf-8",
#         extra="allow",
#     )


# settings = Settings()


# # Main function to run the conversation
# async def main():
#     # Create microphone input and speaker output for the conversation
#     microphone_input, speaker_output = (
#         create_streaming_microphone_input_and_speaker_output(
#             use_default_devices=True,
#         )
#     )
#     print("speaker_output", speaker_output.__dict__)

#     # Create a streaming conversation
#     conversation = StreamingConversation(
#         output_device=speaker_output,
#         transcriber=DeepgramTranscriber(
#             DeepgramTranscriberConfig.from_input_device(
#                 microphone_input,
#                 endpointing_config=PunctuationEndpointingConfig(
#                     time_offset=1.0,  # Adjust as needed
#                     silence_duration=1,  # Adjust as needed
#                 ),
#                 api_key=settings.deepgram_api_key,
#             ),
#         ),
#         agent=ChatGPTAgent(
#             ChatGPTAgentConfig(
#                 openai_api_key=settings.openai_api_key,
#                 initial_message=BaseMessage(text="What up"),
#                 prompt_preamble="""The AI is having a pleasant conversation about life""",
#             )
#         ),
#         # Use custom MP3 synthesizer
#         synthesizer=MP3Synthesizer(
#             MP3SynthesizerConfig.from_output_device(
#                 speaker_output,
#                 file_path=os.path.dirname(__file__)
#                 + "/27171216-44100-2-e9d20b5bba90b.mp3",
#             )
#         ),
#     )

#     # Start the conversation
#     await conversation.start()
#     print("Conversation started, press Ctrl+C to end")

#     # Setup signal handler to terminate on Ctrl+C
#     signal.signal(
#         signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate())
#     )

#     # Continuously receive audio chunks from the microphone
#     while conversation.is_active():
#         chunk = await microphone_input.get_audio()
#         conversation.receive_audio(chunk)


# # Entry point for the script
# if __name__ == "__main__":
#     asyncio.run(main())


import pygame
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer
from vocode.streaming.models.synthesizer import SynthesizerConfig
from pydantic import Field


class MP3SynthesizerConfig(SynthesizerConfig):
    file_path: str = Field(..., description="Path to the MP3 file")

    @classmethod
    def from_output_device(cls, speaker_output, file_path):
        return cls(
            file_path=file_path,
            sampling_rate=speaker_output.sampling_rate,
            audio_encoding=speaker_output.audio_encoding,
        )


# # Settings for the streaming conversation
# class Settings(BaseSettings):

#     # This means a .env file can be used to overload these settings
#     # ex: "OPENAI_API_KEY=my_key" will set openai_api_key over the default above
#     model_config = SettingsConfigDict(
#         env_file=".env",
#         env_file_encoding="utf-8",
#         extra="allow",
#     )


# settings = Settings()


class MP3SynthesizerConfig(SynthesizerConfig):
    file_path: str = Field(..., description="Path to the MP3 file")

    @classmethod
    def from_output_device(cls, speaker_output, file_path):
        return cls(
            file_path=file_path,
            sampling_rate=speaker_output.sampling_rate,
            audio_encoding=speaker_output.audio_encoding,
        )


class MP3Synthesizer(BaseSynthesizer):
    def __init__(self, config: MP3SynthesizerConfig):
        super().__init__(config)
        self.file_path = config.file_path
        pygame.mixer.init(frequency=config.sampling_rate)

    async def create_speech(self, text: str):
        pygame.mixer.music.load(self.file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)


async def main():
    microphone_input, speaker_output = (
        create_streaming_microphone_input_and_speaker_output(use_default_devices=True)
    )

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=PunctuationEndpointingConfig(),
                api_key=deepgram_api_key,
            )
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                openai_api_key=openai_api_key,
                initial_message=BaseMessage(text="Hello! How can I help you today?"),
                prompt_preamble="You are a helpful AI assistant engaging in a friendly conversation.",
            )
        ),
        synthesizer=FileSynthesizer(
            FileSynthesizerConfig.from_output_device(
                speaker_output, file_path="27171216-44100-2-e9d20b5bba90b.mp3"
            ),
        ),
        # synthesizer=MP3Synthesizer(
        #     MP3SynthesizerConfig.from_output_device(
        #         speaker_output, file_path="27171216-44100-2-e9d20b5bba90b.mp3"
        #     )
        # ),
    )

    # Start the conversation
    await conversation.start()
    print("Conversation started, press Ctrl+C to end")

    # Setup signal handler to terminate on Ctrl+C
    signal.signal(
        signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate())
    )

    # Continuously receive audio chunks from the microphone
    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())
