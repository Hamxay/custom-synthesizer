AZURE_SYNTHESIZER_DEFAULT_VOICE_NAME = "en-US-SteffanNeural"
AZURE_SYNTHESIZER_DEFAULT_PITCH = 0
AZURE_SYNTHESIZER_DEFAULT_RATE = 15
from pydantic.v1 import validator
import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import io
from os import getenv
import os
from typing import List, Optional
import azure.cognitiveservices.speech as speechsdk
from pydantic import BaseModel
from vocode.streaming.models.message import BaseMessage, SSMLMessage
from vocode.streaming.synthesizer.base_synthesizer import (
    FILLER_AUDIO_PATH,
    FILLER_PHRASES,
    BaseSynthesizer,
    FillerAudio,
    SynthesisResult,
    encode_as_wav,
)
from vocode.streaming.models.client_backend import OutputAudioConfig
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from loguru import logger

from xml.etree import ElementTree

import re


from typing import Any, List, Tuple

from pydantic.v1 import BaseModel as Pydantic1BaseModel


class BaseModel(Pydantic1BaseModel):
    def __init__(self, **data):
        for key, value in data.items():
            if isinstance(value, dict):
                if (
                    "type" in value and key != "action_trigger"
                ):  # TODO: this is a quick workaround until we get a vocode object version of action trigger (ajay has approved it)
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


class SynthesizerType(str, Enum):
    BASE = "synthesizer_base"
    AZURE = "synthesizer_azure"
    GOOGLE = "synthesizer_google"
    ELEVEN_LABS = "synthesizer_eleven_labs"
    RIME = "synthesizer_rime"
    PLAY_HT = "synthesizer_play_ht"
    GTTS = "synthesizer_gtts"
    STREAM_ELEMENTS = "synthesizer_stream_elements"
    COQUI_TTS = "synthesizer_coqui_tts"
    COQUI = "synthesizer_coqui"
    BARK = "synthesizer_bark"
    POLLY = "synthesizer_polly"
    CARTESIA = "synthesizer_cartesia"


class WordBoundaryEventPool:
    def __init__(self):
        self.events = []

    def add(self, event):
        self.events.append(
            {
                "text": event.text,
                "text_offset": event.text_offset,
                "audio_offset": (event.audio_offset + 5000) / (10000 * 1000),
                "boudary_type": event.boundary_type,
            }
        )

    def get_events_sorted(self):
        return sorted(self.events, key=lambda event: event["audio_offset"])


NAMESPACES = {
    "mstts": "https://www.w3.org/2001/mstts",
    "": "https://www.w3.org/2001/10/synthesis",
}

ElementTree.register_namespace("", NAMESPACES[""])
ElementTree.register_namespace("mstts", NAMESPACES["mstts"])

_AZURE_INSIDE_VOICE_REGEX = r"<voice[^>]*>(.*?)<\/voice>"


class SamplingRate(int, Enum):
    RATE_8000 = 8000
    RATE_16000 = 16000
    RATE_22050 = 22050
    RATE_24000 = 24000
    RATE_44100 = 44100
    RATE_48000 = 48000


class AudioEncoding(str, Enum):
    LINEAR16 = "linear16"
    MULAW = "mulaw"


class SentimentConfig(BaseModel):
    emotions: List[str] = ["angry", "friendly", "sad", "whispering"]

    @validator("emotions")
    def emotions_must_not_be_empty(cls, v):
        if len(v) == 0:
            raise ValueError("must have at least one emotion")
        return v


class SynthesizerConfig(TypedModel, type=SynthesizerType.BASE.value):  # type: ignore
    sampling_rate: int
    audio_encoding: AudioEncoding
    should_encode_as_wav: bool = False
    sentiment_config: Optional[SentimentConfig] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_output_device(cls, output_device: BaseOutputDevice, **kwargs):
        return cls(
            sampling_rate=output_device.sampling_rate,
            audio_encoding=output_device.audio_encoding,
            **kwargs,
        )

    # TODO(EPD-186): switch to from_twilio_output_device and from_vonage_output_device
    @classmethod
    def from_telephone_output_device(cls, **kwargs):
        return cls(
            sampling_rate=411000, audio_encoding=AudioEncoding.LINEAR16, **kwargs
        )

    @classmethod
    def from_output_audio_config(cls, output_audio_config: OutputAudioConfig, **kwargs):
        return cls(
            sampling_rate=output_audio_config.sampling_rate,
            audio_encoding=output_audio_config.audio_encoding,
            **kwargs,
        )


class FileSynthesizerConfig(SynthesizerConfig, type=SynthesizerType.AZURE.value):  # type: ignore
    voice_name: str = AZURE_SYNTHESIZER_DEFAULT_VOICE_NAME
    pitch: int = AZURE_SYNTHESIZER_DEFAULT_PITCH
    rate: int = AZURE_SYNTHESIZER_DEFAULT_RATE
    language_code: str = "en-US"


class FileSynthesizer(BaseSynthesizer[FileSynthesizerConfig]):
    OFFSET_MS = 100

    def __init__(
        self,
        synthesizer_config: FileSynthesizerConfig,
        azure_speech_key: Optional[str] = None,
        azure_speech_region: Optional[str] = None,
    ):
        super().__init__(synthesizer_config)
        # Instantiates a client
        azure_speech_key = azure_speech_key or getenv("AZURE_SPEECH_KEY")
        azure_speech_region = azure_speech_region or getenv("AZURE_SPEECH_REGION")
        if not azure_speech_key:
            raise ValueError(
                "Please set AZURE_SPEECH_KEY environment variable or pass it as a parameter"
            )
        if not azure_speech_region:
            raise ValueError(
                "Please set AZURE_SPEECH_REGION environment variable or pass it as a parameter"
            )
        speech_config = speechsdk.SpeechConfig(
            subscription=azure_speech_key, region=azure_speech_region
        )
        if self.synthesizer_config.audio_encoding == AudioEncoding.LINEAR16:
            if self.synthesizer_config.sampling_rate == SamplingRate.RATE_44100:
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw44100Hz16BitMonoPcm
                )
            if self.synthesizer_config.sampling_rate == SamplingRate.RATE_48000:
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm
                )
            if self.synthesizer_config.sampling_rate == SamplingRate.RATE_24000:
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
                )
            elif self.synthesizer_config.sampling_rate == SamplingRate.RATE_16000:
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
                )
            elif self.synthesizer_config.sampling_rate == SamplingRate.RATE_8000:
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw8Khz16BitMonoPcm
                )
        elif self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw
            )
        self.synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=None
        )

        self.voice_name = self.synthesizer_config.voice_name
        self.pitch = self.synthesizer_config.pitch
        self.rate = self.synthesizer_config.rate
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: FileSynthesizerConfig) -> str:
        return ":".join(
            (
                "azure",
                synthesizer_config.voice_name,
                str(synthesizer_config.pitch),
                str(synthesizer_config.rate),
                synthesizer_config.language_code,
                synthesizer_config.audio_encoding,
            )
        )

    async def get_phrase_filler_audios(self) -> List[FillerAudio]:
        filler_phrase_audios = []
        for filler_phrase in FILLER_PHRASES:
            cache_key = "-".join(
                (
                    str(filler_phrase.text),
                    str(self.synthesizer_config.type),
                    str(self.synthesizer_config.audio_encoding),
                    str(self.synthesizer_config.sampling_rate),
                    str(self.voice_name),
                    str(self.pitch),
                    str(self.rate),
                )
            )
            filler_audio_path = os.path.join(FILLER_AUDIO_PATH, f"{cache_key}.bytes")
            if os.path.exists(filler_audio_path):
                audio_data = open(filler_audio_path, "rb").read()
            else:
                logger.debug(f"Generating filler audio for {filler_phrase.text}")
                ssml = self.create_ssml(filler_phrase.text)
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool_executor, self.synthesizer.speak_ssml, ssml
                )
                offset = self.synthesizer_config.sampling_rate * self.OFFSET_MS // 1000
                audio_data = result.audio_data[offset:]
                with open(filler_audio_path, "wb") as f:
                    f.write(audio_data)
            filler_phrase_audios.append(
                FillerAudio(
                    filler_phrase,
                    audio_data,
                    self.synthesizer_config,
                )
            )
        return filler_phrase_audios

    def add_marks(self, message: str, index=0) -> str:
        search_result = re.search(r"([\.\,\:\;\-\—]+)", message)
        if search_result is None:
            return message
        start, end = search_result.span()
        with_mark = message[:start] + f'<mark name="{index}" />' + message[start:end]
        rest = message[end:]
        rest_stripped = re.sub(r"^(.+)([\.\,\:\;\-\—]+)$", r"\1", rest)
        if len(rest_stripped) == 0:
            return with_mark
        return with_mark + self.add_marks(rest_stripped, index + 1)

    def word_boundary_cb(self, evt, pool):
        pool.add(evt)

    def create_ssml(self, message: str) -> str:
        voice_language_code = self.synthesizer_config.voice_name[:5]
        ssml_root = ElementTree.fromstring(
            f'<speak version="1.0" xmlns="https://www.w3.org/2001/10/synthesis" xml:lang="{voice_language_code}"></speak>'
        )
        voice = ElementTree.SubElement(ssml_root, "voice")
        voice.set("name", self.voice_name)
        if self.synthesizer_config.language_code != "en-US":
            lang = ElementTree.SubElement(voice, "{%s}lang" % NAMESPACES.get(""))
            lang.set("xml:lang", self.synthesizer_config.language_code)
            voice_root = lang
        else:
            voice_root = voice
        # this ugly hack is necessary so we can limit the gap between sentences
        # for normal sentences, it seems like the gap is > 500ms, so we're able to reduce it to 500ms
        # for very tiny sentences, the API hangs - so we heuristically only update the silence gap
        # if there is more than one word in the sentence
        if " " in message:
            silence = ElementTree.SubElement(
                voice_root, "{%s}silence" % NAMESPACES.get("mstts")
            )
            silence.set("value", "500ms")
            silence.set("type", "Tailing-exact")
        prosody = ElementTree.SubElement(voice_root, "prosody")
        prosody.set("pitch", f"{self.pitch}%")
        prosody.set("rate", f"{self.rate}%")
        prosody.text = message.strip()
        ssml = ElementTree.tostring(ssml_root, encoding="unicode")
        regmatch = re.search(_AZURE_INSIDE_VOICE_REGEX, ssml, re.DOTALL)
        if regmatch:
            self.total_chars += len(regmatch.group(1))
        return ssml

    def synthesize_ssml(self, ssml: str) -> speechsdk.AudioDataStream:
        result = self.synthesizer.start_speaking_ssml_async(ssml).get()
        return speechsdk.AudioDataStream(result)

    def ready_synthesizer(self, chunk_size: int):
        # TODO: remove warming up the synthesizer for now
        # connection = speechsdk.Connection.from_speech_synthesizer(self.synthesizer)
        # connection.open(True)
        pass

    # given the number of seconds the message was allowed to go until, where did we get in the message?
    def get_message_up_to(
        self,
        message: str,
        ssml: str,
        seconds: float,
        word_boundary_event_pool: WordBoundaryEventPool,
    ) -> str:
        events = word_boundary_event_pool.get_events_sorted()
        for event in events:
            if event["audio_offset"] > seconds:
                ssml_fragment = ssml[: event["text_offset"]]
                # TODO: this is a little hacky, but it works for now
                return ssml_fragment.split(">")[-1]
        return message

    async def get_pre_downloaded_audio(
        self, message: BaseMessage
    ) -> Optional[io.BytesIO]:
        file_name = (
            f"{message.text}.wav"  # Assuming the file is named after the message text
        )
        file_path = "C:/Users/user/Documents/GitHub/ai-voice-agent-vocode-template/27171216-44100-2-e9d20b5bba90b.mp3"

        if os.path.exists(file_path):
            # If the file exists, load it into a BytesIO stream and return it
            audio_stream = io.BytesIO()
            with open(file_path, "rb") as audio_file:
                audio_stream.write(audio_file.read())
            audio_stream.seek(0)  # Reset the stream position
            return audio_stream
        else:
            return None

    # Update create_speech_uncached to use pre-downloaded audio if available
    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        # Check if the audio for this message has already been downloaded
        pre_downloaded_audio = await self.get_pre_downloaded_audio(message)
        if pre_downloaded_audio:
            return SynthesisResult(
                chunk_generator=None,  # No need for chunk generation
                message=lambda _: message.text,
                audio_stream=pre_downloaded_audio,  # Return the pre-downloaded audio stream
            )

        # If no pre-downloaded audio, fall back to synthesizing new audio
        return await super().create_speech_uncached(
            message, chunk_size, is_first_text_chunk, is_sole_text_chunk
        )
