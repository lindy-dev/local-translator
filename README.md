 ## Installations instructions

 1. Make sure you have uv, python installed on your system.
 2. Clone this repository and install dependencies: `uv pip install -r pyproject.toml`
 3. Activate the environment 
 4. Create a directory "tts-models"
 5. Download the kokoro model from "https://github.com/thewh1teagle/kokoro-onnx" and place it in the tts-models folder.
 6. Download the kokoro voices.json and place it in the tts-models folder.
 7. Run the following command to start the server: `python run main.py`
