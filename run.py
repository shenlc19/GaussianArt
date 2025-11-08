import os
import json
import subprocess
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, required=True)
parser.add_argument("--gpu", type=int, default='0')
parser.add_argument("--root_dir", type=str, default='./data')
parser.add_argument("--save_dir", type=str, default='./output')
args = parser.parse_args()

gpu_id = args.gpu

DATA_PROCESS = "depth_sem_init.py"
SCRIPT_NAME = "train.py"
ROOT_DIR = args.root_dir

dirpath = glob.glob(os.path.join(ROOT_DIR, f'*{args.model_id}*'))[0]
trans_path = os.path.join(dirpath, 'gt/trans.json')

try:
    with open(trans_path, 'r') as f:
        data = json.load(f)

    trans_info = data.get("trans_info", [])
    translate_indices = [i for i, item in enumerate(trans_info) if item.get("type") == "translate"]

    translate_indices.append(len(trans_info))

    relative_path = dirpath.replace("/gt", "")
    opt_path = os.path.relpath(relative_path, ROOT_DIR)
    model_path = os.path.join("output/MPArt90", opt_path)
    print(f"Model Path: {model_path}")

    cmd1 = [
        "python", DATA_PROCESS,
        "--source_path", relative_path,
        "--model_path", model_path,
        "-r", "1",
        "--eval"
    ]
    
    cmd2 = [
        "python", SCRIPT_NAME,
        "-s", relative_path,
        "-m", model_path,
        "-r", "1",
        "--eval",
        "--num_parts", str(len(trans_info) + 1),
        "--freeze_parts"
    ] + [str(i) for i in translate_indices]

    print("Running command:", " ".join(cmd1))
    os.system(" ".join(cmd1))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    print("Running command:", " ".join(cmd2))
    os.system(" ".join(cmd2))

except Exception as e:
    print(f"Error:{e}")
