import os
import shutil
import subprocess
import argparse

REPO_URL                    = "https://github.com/multimodal-art-projection/YuE.git"
XCODEC_MINI_INFER_REPO_URL  = "https://huggingface.co/m-a-p/xcodec_mini_infer"
NGROK_INSTALLER_URL         = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"

INFER_GUI_CODE              = """
import os
import torch
import warnings
import subprocess
import streamlit as st
from datetime import datetime


warnings.filterwarnings(action="ignore")

PROMPT_PATH = os.path.join("prompts", str(datetime.now()).replace(":", "-"))
os.makedirs(PROMPT_PATH, exist_ok=True)

def write_text_to_disk(text: str, path: str):
    with open(path, "w") as f:
        f.write(text)
    f.close()


if __name__ == "__main__": 
    st.title("YuE Music Generation")

    options                                   = {}
    device_idxs                               = [i for i in range(0, torch.cuda.device_count())]
    options["cuda_idx"]                       = st.selectbox("CUDA device", options=device_idxs)
    options["genre_txt"]                      = st.text_area("Genre Text", value="")
    options["lyrics_txt"]                     = st.text_area("Lyrics Text", value="")
    options["seed"]                           = int(st.text_input("Seed", value="42"))
    options["stage1_model"]                   = st.text_input("Stage 1 model", value="m-a-p/YuE-s1-7B-anneal-en-cot")
    options["stage2_model"]                   = st.text_input("Stage 2 model", value="m-a-p/YuE-s2-1B-general")
    options["vocal_track_prompt_path"]        = st.text_input("Vocal track prompt path", value="")
    options["instrumental_track_prompt_path"] = st.text_input("instrumental_track_prompt_path", value="")
    options["audio_prompt_path"]              = st.text_input("audio_prompt_path", value="")
    options["stage2_batch_size"]              = int(st.text_input("Stage 2 batch size", value="4"))
    options["run_n_segments"]                 = int(st.text_input("Number of segments ", value="2"))
    options["output_dir"]                     = st.text_input("output directory", value="../output")
    options["max_new_tokens"]                 = int(st.text_input("max_new_tokens", value="3000"))
    options["repetition_penalty"]             = float(st.text_input("Repitition penalty", value="1.1"))
    options["prompt_start_time"]              = float(st.text_input("prompt start time", value="0"))
    options["prompt_end_time"]                = float(st.text_input("prompt end time", value="30"))

    options["keep_intermediate"]              = st.checkbox("Keep Intermediate", value=False)
    options["disable_offload_model"]          = st.checkbox("Disable Offload Model", value=False)
    options["use_audio_prompt"]               = st.checkbox("Use Audio Prompt", value=False)
    options["use_dual_tracks_prompt"]         = st.checkbox("Use Dual track prompts", value=False)
    options["rescale"]                        = st.checkbox("Rescale", value=False)
    
    if st.button("Create Music"):
        if len(options["genre_txt"]) > 0:
            path = os.path.join(PROMPT_PATH, "genre.txt")
            write_text_to_disk(options["genre_txt"], path)
            options["genre_text"] = path

        if len(options["lyrics_txt"]) > 0:
            path = os.path.join(PROMPT_PATH, "lyrics.txt")
            write_text_to_disk(options["lyrics_txt"], path)
            options["lyrics_txt"] = path

        args = [(f"--{k}" if isinstance(v, bool) else f"--{k} {v}") for k, v in options.items()]

        process = subprocess.Popen(
            ["python",  "infer.py"] + args,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE, 
            text=True
        )
        for line in process.stdout:
            st.write(line)

        stderr_output = process.stderr.read()
        if stderr_output:
            st.error(stderr_output, icon="🚨")

        process.wait()
        process.terminate()
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ngrok_auth_token", type=str, help="ngrok auth token")
    parser.add_argument("--ngrok_port", default=8501, type=int, help="ngrok auth token")
    args = parser.parse_args()

    subprocess.run(
        f"""
        conda create -n yue python &&
        conda activate yue &&
        conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia &&
        curl -sSL {REPO_URL.replace('.git', '')}/main/requirements.txt | pip install -r /dev/stdin &&
        pip install flash-attn --no-build-isolation &&
        pip install streamlit &&
        sudo apt update &&
        sudo apt install git-lfs &&
        git lfs install &&
        git clone {REPO_URL} &&
        git clone {XCODEC_MINI_INFER_REPO_URL}
        """, 
        shell=True
    )

    repo           = REPO_URL.split("/")[-1].split(".")[0]
    infer_repo     = XCODEC_MINI_INFER_REPO_URL.split("huggingface.co")[-1]
    infer_repo_dir = os.path.join(repo, infer_repo)
    inference_dir  = os.path.join(repo, "inference")

    if os.path.isdir(infer_repo_dir):
        shutil.move(infer_repo, inference_dir)

    with open(os.path.join(inference_dir, "infer_gui.py"), "w") as f:
        f.write(INFER_GUI_CODE)
    f.close()

    subprocess.run(
        f"""
        wget {NGROK_INSTALLER_URL} &&
        sudo tar xvzf ./{NGROK_INSTALLER_URL.split('/')[-1]} -C /usr/local/bin &&
        ngrok authtoken {args.ngrok_auth_token} &&
        ngrok http {args.ngrok_port} &&
        streamlit run infer_gui.py
        """, shell=True
    )