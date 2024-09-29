# Subtitle Creator

This scripts creates english subtitles for an input video.
It works completely locally using [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) for transcription
and translation.

## Requirements

- Audio extracting and adding the subtitles to the video requires [FFmpeg](https://ffmpeg.org/).
- Python dependencies are managed via [Poetry](https://python-poetry.org/): ``poetry install``
- It is recommended to install [CUDA](https://developer.nvidia.com/cuda-downloads) for faster transcription and translation.

## Usage

Start a shell in the virtual env via ``poetry shell``, then run the script with:
``python subtitleCreator.py <input video>``.
Full options: ``python subtitleCreator.py -h``:

````text
usage: subtitleCreator.py [-h] [-o OUTPUT] [--temp TEMP] [--overwrite] [--burn] [--keep] video_file

Add translated subtitles to a video.

positional arguments:
  video_file            Path to the input video file.

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to the output video file. If not specified the video will be saved as
                        '<input_file>_subtitles.mp4'.
  --temp TEMP           Path to the temporary directory where the intermediate files will be saved. If not specified
                        the working directory will be used.
  --overwrite           Overwrite the original video file.
  --burn                Burn the subtitles in the video instead of adding them as selectable.
  --keep                Keep temporary files instead of deleting them after processing.
````