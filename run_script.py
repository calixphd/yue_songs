import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ngrok_port", default=8501, type=int, help="ngrok auth token")
    args = parser.parse_args()
    subprocess.run(f"ngrok http {args.ngrok_port} && streamlit run infer_gui.py", shell=True)