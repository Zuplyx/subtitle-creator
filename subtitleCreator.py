import argparse
import os
import shutil
import subprocess
import datetime

import srt
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def check_ffmpeg():
    """Check if FFmpeg is installed, otherwise raise an error."""
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("FFmpeg is not installed or not on the PATH. Please install it to use this script.")


def extract_audio(video_file, audio_file):
    """Extract the audio from a video file and save it to an audio file."""
    try:
        subprocess.run(['ffmpeg', '-i', video_file, '-q:a', '0', '-map', 'a', audio_file, '-y'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed with exit code {e.returncode}. Check the logs for more information.")
        raise e
    except Exception as e:
        print("An unexpected error occurred while trying to extract audio. The error was: ", str(e))
        raise e


def translate_audio(audio_file: str) -> list[dict]:
    """Translate the audio from a file using openai/whisper-large-v3 model.

    Args:
        audio_file (str): Path to the audio file

    Returns:
        list[dict]: List of translated chunks with timestamps
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    print(f"Creating processor for model {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        chunk_length_s=30,
        batch_size=16,  # batch size for inference - set based on your device
        device=device,
    )

    # get the result with timestamps
    print("Starting translation and transcription")
    result = pipe(audio_file, return_timestamps=True, generate_kwargs={"task": "translate"})
    print("Finished translation and transcription")
    return result["chunks"]


def create_subs(translations: list[dict]) -> list[srt.Subtitle]:
    """Create a list of Subtitle objects for each translation chunk.

    Preprocess translations: if consecutive entries have the same text, combine them.
    """

    # Initialize an empty list to store the subtitles
    subs = []

    # Iterate over the translations with their indices
    for i in range(len(translations)):
        translation = translations[i]
        timestamp = translation['timestamp']
        text = translation['text']

        # If this is not the first entry, check if the text is the same as the previous one
        if i > 0 and translations[i-1]['text'] == text:
            # If it's the same, append the current timestamp to the previous subtitle
            subs[-1].end = datetime.timedelta(seconds=max(subs[-1].end.total_seconds(), timestamp[1]))
        else:
            # Otherwise, create a new subtitle with the current text and timestamp
            s = srt.Subtitle(index=i, content=text, start=datetime.timedelta(seconds=timestamp[0]), end=datetime.timedelta(seconds=timestamp[1]))
            subs.append(s)

    return subs



def create_srt_files(subs: list[srt.Subtitle], srt_file: str):
    """
    Create SRT file from transcriptions and translations.

    :param subs: List of SRT subtitle objects to be written into the SRT file.
    :param srt_file: Path where the SRT file will be created. It should have a '.srt' extension.
    :return: None
    """
    if not srt_file.endswith('.srt'):
        raise ValueError("The provided path should point to an empty file with a '.srt' extension.")

    with open(srt_file, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subs))
        f.close()


def add_selectable_subtitles_to_video(video_file, subtitle_file, output_video):
    """Add subtitle file to video using ffmpeg."""
    try:
        subprocess.run([
            'ffmpeg',
            '-i', video_file,
            '-i', subtitle_file,
            '-c:v', 'copy',
            '-c:a', 'copy',
            '-c:s', 'mov_text',
            '-metadata:s:s:0', 'language=eng',
            '-map', '0:v',
            '-map', '0:a',
            '-map', '1:0',
            '-y',
            output_video
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg command failed: {e.returncode}. Check the logs for more information.")
        raise e
    except Exception as e:
        print(f"An unexcpected error occurred during adding subtitles to video: {str(e)}")
        raise e

def add_burned_subtitles_to_video(video_file, subtitle_file, output_video):
    """Add burned in subtitles to video using ffmpeg."""
    try:
        subprocess.run([
            'ffmpeg',
            '-i', video_file,
            '-vf', "subtitles={}".format(subtitle_file),
            '-y',
            output_video
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg command failed: {e.returncode}. Check the logs for more information.")
        raise e
    except Exception as e:
        print(f"An unexcpected error occurred during adding subtitles to video: {str(e)}")
        raise e


# Main Workflow
def main(video_file: str, output_file: str, overwrite: bool, burn_in: bool, keep:bool, temp_dir: str ):
    check_ffmpeg()

    # temp files
    audio_file = 'audio.wav'
    subtitle_file = 'subtitles.srt'
    if temp_dir:
        audio_file = temp_dir +"/" + audio_file
        subtitle_file = temp_dir +"/" +subtitle_file

    # extract audio
    extract_audio(video_file, audio_file)

    # transcribe and translate audio
    translations = translate_audio(audio_file)

   # create subtitles from translations
    subs = create_subs(translations)

    # Create Subtitle file
    create_srt_files(subs, subtitle_file)

    # Determine output file path
    if not output_file:
        output_video = os.path.splitext(video_file)[0] + "_subtitles.mp4"
    else:
        if os.path.exists(output_file) and not overwrite:
            raise FileExistsError(
                f"Output file '{output_file}' already exists. Use the --overwrite flag to overwrite it.")
        output_video = output_file

    if not burn_in:
        # Add selectable subtitles to video
        add_selectable_subtitles_to_video(video_file, subtitle_file, output_video)
        if overwrite:
            shutil.move(output_video, video_file)
            output_video = video_file
        print(f"Selectable subtitles added successfully to the video! Output saved to: {output_video}")
    else:
        # Add burned in subtitles
        add_burned_subtitles_to_video(video_file, subtitle_file, output_video)
        if overwrite:
            shutil.move(output_video, video_file)
            output_video = video_file
        print(f"Burned in subtitles added successfully! Output saved to: {output_video}")



    # Delete temp files
    if not keep:
        os.remove(audio_file)
        os.remove(subtitle_file)

# Command-line arguments setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add translated subtitles to a video.")
    parser.add_argument("video_file", help="Path to the input video file.")
    parser.add_argument("-o", "--output",
                        help="Path to the output video file. If not specified the video will be saved as "
                             "'<input_file>_subtitles.mp4'.")
    parser.add_argument("--temp",
                        help="Path to the temporary directory where the intermediate files will be saved. If not specified the working directory will be used.")
    parser.add_argument("--overwrite", help="Overwrite the original video file.", action="store_true")
    parser.add_argument("--burn", help="Burn the subtitles in the video instead of adding them as selectable.", action="store_true")
    parser.add_argument("--keep", help="Keep temporary files instead of deleting them after processing.",
                        action="store_true")
    args = parser.parse_args()

    main(args.video_file, args.output, args.overwrite, args.burn, args.keep, args.temp)
