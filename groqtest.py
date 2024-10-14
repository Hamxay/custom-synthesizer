import os
from groq import Groq


# Initialize the Groq client
client = Groq(api_key="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")

# Specify the path to the audio file
filename = (
    os.path.dirname(__file__) + "/27171216-44100-2-e9d20b5bba90b.mp3"
)  # Replace with your audio file!

# Open the audio file
with open(filename, "rb") as file:
    # Create a transcription of the audio file
    transcription = client.audio.transcriptions.create(
        file=(filename, file.read()),  # Required audio file
        model="whisper-large-v3-turbo",  # Required model to use for transcription
        prompt="Specify context or spelling",  # Optional
        response_format="json",  # Optional
        temperature=0.0,  # Optional
    )
    # Print the transcription text
    print(transcription.text)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": transcription.text,
            }
        ],
        model="llama3-8b-8192",
        stream=True,
    )

    for chunk in chat_completion:
        print(chunk.choices[0].delta.content, end="")
