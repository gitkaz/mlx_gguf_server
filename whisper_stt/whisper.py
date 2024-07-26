import os
import mlx_whisper
from typing import Optional, Dict, Any

class AudioTranscriber:
    """
    A class for transcribing audio files using the mlx_whisper library.
    It converts audio to text using mlx-whisper.
    """

    def __init__(self, model_path: str = None, file_path: Optional[str] = None):
        """
        Constructor for the AudioTranscriber class.

        :param model_path: Hugging Face path or local directory path to the mlx-whisper model file
        :param file_path: Path to the audio file to be processed (optional)
        """
        self.model_path = model_path
        self.file_path = file_path

    def set_file_path(self, file_path: str):
        """
        Sets the path of the audio file to be processed.

        :param file_path: Path to the audio file
        """
        self.file_path = file_path

    def transcribe(self, language: str = None) -> Dict[str, Any]:
        """
        Performs transcription on the audio file.

        :param language: Language of the audio (optional). If not specified, automatic detection is attempted.
        :return: A dictionary containing the transcription result.
                 On success: {"text": transcription_text}
                 On failure: {"error": error_message}
        :raises ValueError: If the file path is not set
        """
        if not self.file_path:
            raise ValueError("File path is not set. Use set_file_path() or provide it during initialization.")
        try:
            if language:
                result = mlx_whisper.transcribe(self.file_path, path_or_hf_repo=self.model_path, language=language)
            else:
                result = mlx_whisper.transcribe(self.file_path, path_or_hf_repo=self.model_path)
            
            return {"text": result["text"]}
        except Exception as e:
            return {"error": f"Transcription failed: {str(e)}"}

    def delete_file(self) -> None:
        """
        Deletes the processed audio file.
        Does nothing if the file doesn't exist.
        """
        if self.file_path and os.path.exists(self.file_path):
            os.remove(self.file_path)
            self.file_path = None