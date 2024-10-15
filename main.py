import asyncio
import signal
from pydantic_settings import BaseSettings, SettingsConfigDict
from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from main_synthesizer import FileSynthesizer, FileSynthesizerConfig

configure_pretty_logging()


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


async def main():
    microphone_input, speaker_output = (
        create_streaming_microphone_input_and_speaker_output(
            use_default_devices=True,
        )
    )

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=PunctuationEndpointingConfig(),
                api_key=settings.deepgram_api_key,
            ),
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                openai_api_key=settings.openai_api_key,
                initial_message=BaseMessage(text="What up"),
                prompt_preamble="""The AI is having a pleasant conversation about life""",
            )
        ),
        synthesizer=FileSynthesizer(
            FileSynthesizerConfig(
                file_path="27171216-44100-2-e9d20b5bba90b.mp3",  # Path to the audio file
                chunk_size=1024,
                audio_encoding="mulaw",
                sampling_rate=8000,
            )
        ),
    )

    async def handle_manual_input():
        while conversation.is_active():
            command = (
                input(
                    "Enter 'pause' to pause playback or 'resume' to resume playback: "
                )
                .strip()
                .lower()
            )

            if command == "pause":
                print("Pausing file playback...")
                conversation.synthesizer.pause()
            elif command == "resume":
                print("Resuming file playback...")
                conversation.synthesizer.resume()
            else:
                print("Unknown command. Please enter 'pause' or 'resume'.")

            await asyncio.sleep(0.1)  # Small delay to prevent busy loop

    await conversation.start()
    print("Conversation started, press Ctrl+C to end")

    signal.signal(
        signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate())
    )

    await handle_manual_input()

    print("Conversation terminated.")


if __name__ == "__main__":
    asyncio.run(main())
