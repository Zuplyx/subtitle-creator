# Subtitle Creator

This scripts creates english subtitles for an input video.
It works completely locally using [Vosk](https://github.com/alphacep/vosk-api) for transcription
and [Argos](https://github.com/argosopentech/argos-translate) for translation.

## Requirements

- Audio extracting and adding the subtitles requires [FFmpeg](https://ffmpeg.org/).
- Transcription of the original video requires the corresponding [Vosk Model](https://alphacephei.com/vosk/models) for
  the video's language.
- Python dependencies are managed via [Poetry](https://python-poetry.org/): ``poetry install``

## Usage

Start a shell in the virtual env via ``poetry shell``, then run the script with:
``python subtitleCreator.py <input video> <vosk model>``.
Full options: ``python subtitleCreator.py -h``:

````text
usage: subtitleCreator.py [-h] [-o OUTPUT] [--overwrite] video_file vosk_model_path

Add translated subtitles to a video.

positional arguments:
  video_file            Path to the input video file.
  vosk_model_path       Path to the Vosk model directory.

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to the output video file. If not specified the video will be saved as '<input_file>_subtitles.mp4'.
  --overwrite           Overwrite the original video file.
````