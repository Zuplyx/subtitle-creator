import argparse
import os
import shutil
import subprocess
import tempfile

import langcodes
import langid
import soundfile as sf
import srt
import torch
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from argostranslate import package, translate
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


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


def load_audio(file_path):
    """
    Load the audio file and return audio data and sample rate.
    """
    audio_input, sample_rate = sf.read(file_path)
    return audio_input, sample_rate


def detect_language(audio_input, sample_rate):
    """
    Detect the language of the spoken content in the audio using a portion of the input.
    Returns an ISO 639-1 language code.
    """
    # Convert audio to a string format for language detection
    # Here we take a portion of the audio input (first 30 seconds or so) for detection
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    inputs = processor(audio_input[:sample_rate * 30], sampling_rate=sample_rate, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    transcription_sample = processor.batch_decode(predicted_ids)[0]

    # Detect language from transcription sample
    detected_lang, confidence = langid.classify(transcription_sample)
    return detected_lang


def transcribe_audio(audio_input: torch.Tensor, sample_rate: int) -> tuple[str, torch.Tensor, Wav2Vec2Processor]:
    """
    Transcribe the audio using XLS-R and return the transcription.

    Args:
        audio_input (torch.Tensor): The input audio tensor.
        sample_rate (int): The sample rate of the audio.

    Returns:
        tuple[str, torch.Tensor, Wav2Vec2Processor]: A tuple containing the transcription string,
            the logits tensor from the model, and the processor used for preprocessing.
    """
    # Load XLS-R model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    # Preprocess the audio for the model
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    # Run the model and get the logits
    with torch.no_grad():
        logits = model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    # Convert predicted ids to text (basic transcription)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription, logits, processor


def convert_iso639_1_to_2(iso639_1_code):
    lang = langcodes.Language.get(iso639_1_code)
    return lang.to_tag().split('-')[0] if lang else None


def align_transcript_with_audio(audio_file_path: str, transcript_string: str, language: str = "eng") -> list[
    srt.Subtitle]:
    """
    Align the given transcript with the audio file and generate an SRT subtitle list.

    Args:
        audio_file_path (str): The path to the audio file.
        transcript_string (str): The transcription string to align with the audio.
        language (str, optional): The language of the audio. Defaults to "eng".

    Returns:
        list[srt.Subtitle]: A list of SRT subtitle objects representing the aligned transcript and audio.
    """
    # Create a temporary text file for the transcript
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_transcript_file:
        temp_transcript_file.write(transcript_string.encode('utf-8'))
        temp_transcript_file_path = temp_transcript_file.name

    # Create a Task object
    task = Task(config_string=f"task_language={language}|os_task_file_format=srt|")
    task.audio_file_path = audio_file_path
    task.text_file_path = temp_transcript_file_path

    # Execute the task
    execute_task_result = ExecuteTask(task).execute()

    # Get the SRT content as a string
    srt_content = task.output_sync_map_file()

    # Parse the SRT content into Subtitle objects
    subtitles = list(srt.parse(srt_content))

    return subtitles


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


# Main Workflow
def main(video_file, vosk_model_path, output_file, overwrite):
    check_ffmpeg()

    # temp files. should be configurable in the future and probably be deleted after processing is done.
    audio_file = 'audio.wav'
    subtitle_file = 'subtitles.srt'

    extract_audio(video_file, audio_file)

    # load audio
    audio_input, sample_rate = load_audio(audio_file)

    # detect language
    detected_lang_code = detect_language(audio_input, sample_rate)
    print(f"Detected language code: {detected_lang_code}")

    # transcribe audio
    transcription, logits, processor = transcribe_audio(audio_input, sample_rate)

    # Align transcription and cCreate Subtitles
    subtitles = align_transcript_with_audio(audio_file, transcription, convert_iso639_1_to_2(detected_lang_code))

    # Install Argos Translate model based on detected language
    install_argos_model(detected_lang_code)

    # translate transcription to target language
    translations = translate_transcription(subtitles, detected_lang_code)

    # Create Subtitle file
    create_srt_files(translations, subtitle_file)

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

    # Add selectable subtitles to video
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
