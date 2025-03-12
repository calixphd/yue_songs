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