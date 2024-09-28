import argparse
import datetime
import json
import os
import shutil
import subprocess
import wave
import srt

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


def transcribe_audio(audio_file: str, vosk_model_path: str) -> list[srt.Subtitle]:
    """
    Transcribe audio using Vosk.

    :param audio_file: Path to the audio file to be transcribed
    :param vosk_model_path: Path to the Vosk model file
    :return: A list of transcription results in JSON format, each represented as a SRT subtitle object
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
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            results.append(recognizer.Result())
    results.append(recognizer.FinalResult())

    subs = []
    words_per_line = 7
    for res in results:
        jres = json.loads(res)
        if not "result" in jres:
            continue
        words = jres["result"]
        for j in range(0, len(words), words_per_line):
            line = words[j: j + words_per_line]
            s = srt.Subtitle(index=len(subs),
                             content=" ".join([l["word"] for l in line]),
                             start=datetime.timedelta(seconds=line[0]["start"]),
                             end=datetime.timedelta(seconds=line[-1]["end"]))
            subs.append(s)



    wf.close()  # Close the wave object
    return subs


def detect_language(transcriptions: list[srt.Subtitle]) -> str:
    """Detect the source language of a video by looking at the first few transcription results."""
    sample_text = ' '.join(sub.content for sub in transcriptions[:10])
    detected_lang = detect(sample_text)
    return detected_lang


def translate_transcription(transcriptions: list[srt.Subtitle], source_lang_code) -> list[srt.Subtitle]:
    """
    Translate transcriptions to English using Argos Translate.

    :param transcriptions: A list of SRT subtitle objects containing the transcription results
    :param source_lang_code: The language code for the source language (e.g. 'en', 'es', etc.)
    :return: A list of translated SRT subtitle objects in the target language ('en')
    """
    installed_languages = translate.get_installed_languages()

    # Check if a translation model is available for the specified source language
    if not any(lang.code == source_lang_code for lang in installed_languages):
        raise ValueError(f"No Argos Translate model available for {source_lang_code}.")

    # Get the translator object for the specified source and target languages
    source_lang = next(lang for lang in installed_languages if lang.code == source_lang_code)
    target_lang = next(lang for lang in installed_languages if lang.code == 'en')
    translator = source_lang.get_translation(target_lang)

    # Translate each transcription result using the Argos Translate model
    translations = []
    for result in transcriptions:
        text = result.content
        translated = translator.translate(text)
        subs_clone = srt.Subtitle(index=result.index, content=translated, start=result.start, end=result.end)
        translations.append(subs_clone)

    return translations


def create_srt(subs: list[srt.Subtitle], srt_file: str):
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
    create_srt(translations, subtitle_file)

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
