import argparse
import json
import os
import shutil
import subprocess
import wave
from datetime import timedelta

from argostranslate import package, translate
from langdetect import detect, DetectorFactory
from vosk import Model, KaldiRecognizer

# Set seed to make langdetect results reproducible
DetectorFactory.seed = 0


def check_ffmpeg():
    """Check if FFmpeg is installed, otherwise raise an error."""
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("FFmpeg is not installed. Please install it to use this script.")


def install_argos_model(source_lang, target_lang='en'):
    """Install the Argos Translate model for the specified source and target languages."""
    available_packages = package.get_available_packages()
    for pkg in available_packages:
        if pkg.from_code == source_lang and pkg.to_code == target_lang:
            package.install_from_path(pkg.download())
            print(f"Installed {pkg} for translation from {source_lang} to {target_lang}.")
            return True
    raise ValueError(f"No Argos Translate model available for {source_lang} to {target_lang}.")


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


def transcribe_audio(audio_file, vosk_model_path):
    """
    Transcribe audio using Vosk.

    :param audio_file: Path to the audio file to be transcribed
    :param vosk_model_path: Path to the Vosk model file
    :return: A list of transcription results in JSON format
    """
    # Open the audio file as a wave object
    wf = wave.open(audio_file, 'rb')

    # Initialize the Vosk model and recognizer with the provided model path and audio file parameters
    model = Model(vosk_model_path)
    recognizer = KaldiRecognizer(model, wf.getframerate())
    recognizer.SetWords(True)

    results = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:  # If no more data to read, break the loop
            break
        if recognizer.AcceptWaveform(data):
            results.append(json.loads(recognizer.Result()))
    results.append(json.loads(recognizer.FinalResult()))

    # Add the final transcription result to the list
    results.append(json.loads(recognizer.FinalResult()))
    wf.close()  # Close the wave object
    return results


def detect_language(transcriptions):
    """Detect the source language of a video by looking at the first few transcription results."""
    sample_text = ' '.join(result['text'] for result in transcriptions if 'text' in result)
    detected_lang = detect(sample_text)
    return detected_lang


def translate_transcription(transcriptions, source_lang_code):
    """Translate transcriptions to English using Argos Translate."""
    installed_languages = translate.get_installed_languages()
    source_lang = next(lang for lang in installed_languages if lang.code == source_lang_code)
    target_lang = next(lang for lang in installed_languages if lang.code == 'en')
    translator = source_lang.get_translation(target_lang)

    translations = []
    for result in transcriptions:
        if 'text' in result:
            text = result['text']
            translated = translator.translate(text)
            translations.append(translated)
    return translations


def seconds_to_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    td = timedelta(seconds=seconds)
    return str(td)


def create_srt(transcriptions, translations, srt_file):
    """Create SRT file from transcriptions and translations."""
    with open(srt_file, 'w', encoding='utf-8') as f:
        # remove empty entries
        filtered_pairs = [
            (transcription, translation)
            for transcription, translation in zip(transcriptions, translations)
            if transcription['text'] and translation.strip()
        ]

        for i, (transcription, translation) in enumerate(filtered_pairs):
            if 'result' in transcription and transcription['result']:
                start_time = seconds_to_timestamp(transcription['result'][0]['start'])
                end_time = seconds_to_timestamp(transcription['result'][-1]['end'])
                f.write(f"{i + 1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{translation}\n\n")
        f.close()



def add_selectable_subtitles_to_video(video_file, subtitle_file, output_video):
    """Add subtitle file to video using ffmpeg."""
    try:
        subprocess.run([
            'ffmpeg', '-i', video_file, '-i', subtitle_file,
            '-map', '0:v:0', '-map', '0:a:0', '-map', '1:s:0',
            '-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text',
            '-metadata:s:s:0', 'language=eng',
            '-shortest',
            '-y', output_video
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg command failed: {e.returncode}. Check the logs for more information.")
        raise e
    except Exception as e:
        print(f"An unexcpected error occurred during adding subtitles to video: {str(e)}")
        raise e


# Main Workflow
def main(video_file, vosk_model_path, output_file, overwrite):
    check_ffmpeg()

    audio_file = 'audio.wav'
    subtitle_file = 'subtitles.srt'

    extract_audio(video_file, audio_file)
    transcriptions = transcribe_audio(audio_file, vosk_model_path)
    detected_lang_code = detect_language(transcriptions)

    print(f"Detected language code: {detected_lang_code}")

    # Install Argos Translate model based on detected language
    install_argos_model(detected_lang_code)

    translations = translate_transcription(transcriptions, detected_lang_code)
    create_srt(transcriptions, translations, subtitle_file)

    # Determine output file path
    if not output_file:
        output_file = os.path.splitext(video_file)[0] + "_subtitles.mp4"
    if overwrite:
        output_video = video_file
    else:
        if os.path.exists(output_file):
            raise FileExistsError(
                f"Output file '{output_file}' already exists. Use the --overwrite flag to overwrite it.")
        output_video = output_file

    add_selectable_subtitles_to_video(video_file, subtitle_file, output_video)
    print(f"Selectable subtitles added successfully to the video! Output saved to: {output_video}")



# Command-line arguments setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add translated subtitles to a video.")
    parser.add_argument("video_file", help="Path to the input video file.")
    parser.add_argument("vosk_model_path", help="Path to the Vosk model directory.")
    parser.add_argument("-o", "--output",
                        help="Path to the output video file. If not specified the video will be saved as "
                             "'<input_file>_subtitles.mp4'.")
    parser.add_argument("--overwrite", help="Overwrite the original video file.", action="store_true")
    args = parser.parse_args()

    main(args.video_file, args.vosk_model_path, args.output, args.overwrite)
