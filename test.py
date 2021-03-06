import time
from pathlib import Path

from nbswave import render_audio

song = "test.nbs"
output = "test.mp3"

custom_sound_path = Path.home() / "Minecraft Note Block Studio" / "Data" / "Sounds"

start = time.time()
render_audio(
    song,
    output,
    custom_sound_path=custom_sound_path,
    format="mp3",
    exclude_locked_layers=False,
)
end = time.time()

print(f"Done! Took {end-start:.2f} seconds")
