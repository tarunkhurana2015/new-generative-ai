    Environment setup:

    conda create -n env_llama_local python=3.13 numpy pandas
    conda activate env_llama_local
    python -m pip install --upgrade pip
    Install packages:
    pip install -r requirements.txt

Download llama locally

> ollama pull llama3.2

Compile App:

    python app1.py

Run App:

    streamlit run app1.py
