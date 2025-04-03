from services.stabilizer import video_stabilizer
from services.get_video import get_video
import json
from pathlib import Path

videoIndex = Path("./videoIndex.json")

if not videoIndex.is_file():
    print("No existe el archivo videoIndex.json")
    exit(1)
with open(videoIndex, "r") as f:
    videos = json.load(f)
if not videos:
    print("No hay videos en el archivo videoIndex.json")
    exit(1)


video_id = videos["r1"]["h2"]["t"]

salida = get_video(video_id, "./videos")

video_stabilizer(salida, "./videos/stable.mp4")