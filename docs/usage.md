# Usage  
  
How to use this solution?  
  
1. **Clone this repository**  
2. **Install System Dependencies**
   * You must have **FFmpeg** installed and available in your system PATH.
   * *Ubuntu/Debian*: `sudo apt install ffmpeg`
   * *Mac*: `brew install ffmpeg`
   * *Windows*: Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add `bin` to PATH.
2. **Create a python environement**  
    python3.12 -m venv .venv
3. **Activate your venv**
    source .venv/bin/activate
4. **Install this python package and its dependencies**  
    pip install -e .
5. **Provide your OpenAI API key in the OPENAI_API_KEY environemnt variable.**  
6. **Start the streamlit application**  
    streamlit run src/video_analyst/main.py
  
