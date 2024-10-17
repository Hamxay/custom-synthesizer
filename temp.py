from pydub.utils import mediainfo

info = mediainfo("voice.mp3")
print(info)