import os
import sys
import subprocess
import argparse
import glob
import random
from tqdm import tqdm
import shutil

# --- Absolute Path Configuration (Robust Kaggle Version) ---
BASE_DIR = '/kaggle/working'
ROOP_DIR = os.path.join(BASE_DIR, 'roop')
CODEFORMER_DIR = os.path.join(BASE_DIR, 'CodeFormer')
ROOP_RUN_PY = os.path.join(ROOP_DIR, 'run.py')
CODEFORMER_INFERENCE_PY = os.path.join(CODEFORMER_DIR, 'inference_codeformer.py')

# --- Setup and Utility Functions ---
def run_command(command, description, cwd=None):
    """Runs a shell command and provides detailed error output on failure."""
    print(description)
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print("\n--- ERROR DURING SETUP ---")
        print(f"Failed to run command: {command}")
        print(f"\n--- Stderr ---\n{e.stderr.strip()}")
        print(f"\n--- Stdout ---\n{e.stdout.strip()}")
        print("--- END ERROR ---")
        sys.exit(1)

def setup_environment():
    """Clones repos, installs all dependencies, and downloads all models using fixed absolute paths."""
    print(f"--- Using Base Directory: {BASE_DIR}")
    
    # --- Roop Setup ---
    if not os.path.exists(ROOP_DIR):
        run_command(f'git clone https://github.com/s0md3v/roop.git "{ROOP_DIR}"', "--- Cloning Roop repository ---")
    
    requirements_path_roop = os.path.join(ROOP_DIR, 'requirements.txt')
    if not os.path.exists(requirements_path_roop):
        print(f"FATAL: Could not find Roop requirements file at {requirements_path_roop}"), sys.exit(1)

    print("--- Installing Essential Dependencies ---")
    run_command(f'pip install -r "{requirements_path_roop}" -q', "--- Installing Roop base requirements ---")
    run_command('pip install tqdm gfpgan insightface==0.7.3 onnxruntime-gpu==1.15.1 lpips -q', "--- Installing GPU, Enhancement, and CodeFormer libraries ---")
    run_command('pip install --upgrade numpy==1.26.4 jax==0.4.23 jaxlib==0.4.23 -q', "--- Applying NumPy Compatibility Fix ---")

    # --- Roop Model Downloads (Checks for both 128 and 512 models) ---
    inswapper_128_path = os.path.join(ROOP_DIR, 'inswapper_128.onnx')
    if not os.path.exists(inswapper_128_path):
        run_command(f'wget -q -N -P "{ROOP_DIR}" https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx', "--- Downloading 128px Roop Model ---")
    else:
        print("--- 128px Roop Model already exists. ---")
        
    inswapper_512_path = os.path.join(ROOP_DIR, 'inswapper_128_fp16.onnx')
    if not os.path.exists(inswapper_512_path):
        run_command(f'wget -q -N -P "{ROOP_DIR}" https://huggingface.co/mapooon/roop/resolve/main/inswapper_128_fp16.onnx', "--- Downloading 512px Roop Model ---")
    else:
        print("--- 512px Roop Model already exists. ---")
        
    gfpgan_path = os.path.join(ROOP_DIR, 'GFPGANv1.3.pth')
    if not os.path.exists(gfpgan_path):
        run_command(f'wget -q -N -P "{ROOP_DIR}" https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', "--- Downloading Roop GFPGAN Model ---")
    else:
        print("--- Roop GFPGAN Model already exists. ---")

    # --- CodeFormer Setup ---
    if not os.path.exists(CODEFORMER_DIR):
        run_command(f'git clone https://github.com/sczhou/CodeFormer.git "{CODEFORMER_DIR}"', "--- Cloning CodeFormer repository ---")

    requirements_path_cf = os.path.join(CODEFORMER_DIR, 'requirements.txt')
    run_command(f'pip install -r "{requirements_path_cf}" -q', "--- Installing CodeFormer requirements ---")
    run_command('python basicsr/setup.py develop -q', "--- Building CodeFormer basicsr ---", cwd=CODEFORMER_DIR)

    codeformer_model_path = os.path.join(CODEFORMER_DIR, 'weights', 'CodeFormer', 'codeformer.pth')
    if not os.path.exists(codeformer_model_path):
        os.makedirs(os.path.dirname(codeformer_model_path), exist_ok=True)
        run_command(f'wget -q -N -P "{os.path.dirname(codeformer_model_path)}" https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth', "--- Downloading CodeFormer model ---")
    else:
        print("--- CodeFormer model already exists. ---")

    print("\n\n✅✅✅ Full setup complete for all models. The environment is ready. ✅✅✅\n")


# --- Main Application Logic ---
def run_batch_processing(args):
    if args.enhance_with_codeformer and not os.path.exists(CODEFORMER_INFERENCE_PY):
        print(f"FATAL ERROR: CodeFormer not found. Please run --run-setup first."), sys.exit(1)
        
    if args.roop_model == 128:
        roop_model_path = os.path.join(ROOP_DIR, 'inswapper_128.onnx')
    elif args.roop_model == 512:
        roop_model_path = os.path.join(ROOP_DIR, 'inswapper_128_fp16.onnx')
    else:
        print(f"FATAL: Invalid Roop model size specified: {args.roop_model}"), sys.exit(1)
        
    if not os.path.exists(roop_model_path):
        print(f"FATAL: Roop model file not found at {roop_model_path}. Please run --run-setup.")
        sys.exit(1)

    source_dir, target_dir, output_dir = map(os.path.abspath, [args.source_dir, args.target_dir, args.output_dir])
    os.makedirs(output_dir, exist_ok=True)
    
    temp_output_dir = os.path.join(output_dir, 'temp_roop_swaps')
    if args.enhance_with_codeformer: os.makedirs(temp_output_dir, exist_ok=True)
    
    source_images, target_images = glob.glob(os.path.join(source_dir, "*")), glob.glob(os.path.join(target_dir, "*"))
    if not source_images or not target_images: print("Error: No images found."), exit()

    jobs = []
    if args.random_targets:
        if args.num_jobs <= 0: print("Error: --num-jobs must be > 0."), exit()
        for _ in range(args.num_jobs): jobs.append((random.choice(source_images), random.choice(target_images)))
    else:
        target_index = 0
        for source_path in source_images:
            for _ in range(2):
                if target_index >= len(target_images): break
                jobs.append((source_path, target_images[target_index])); target_index += 1
        if args.num_jobs > 0: jobs = jobs[:args.num_jobs]
    print(f"--- Created {len(jobs)} swap jobs to process using the {args.roop_model}px model. ---")

    for source_path, target_path in tqdm(jobs, desc="Total Progress"):
        try:
            current_output_dir = temp_output_dir if args.enhance_with_codeformer else output_dir
            roop_output_path = os.path.join(current_output_dir, f"{os.path.splitext(os.path.basename(source_path))[0]}_to_{os.path.splitext(os.path.basename(target_path))[0]}.png")
            
            roop_processors = args.frame_processor.replace(',', ' ')
            roop_command = (f"python '{ROOP_RUN_PY}' -s '{source_path}' -t '{target_path}' -o '{roop_output_path}' "
                            f"--frame-processor {roop_processors} --execution-provider {args.execution_provider} "
                            f"--face-swapper-model '{roop_model_path}'")
            subprocess.run(roop_command, shell=True, check=True, capture_output=True, text=True)

            if args.enhance_with_codeformer:
                codeformer_command = (f"python '{CODEFORMER_INFERENCE_PY}' --input_path '{roop_output_path}' "
                                      f"--output_path '{output_dir}' --fidelity {args.codeformer_fidelity} "
                                      f"--face_upsample --bg_upsampler realesrgan")
                subprocess.run(codeformer_command, shell=True, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"\n\n--- ERROR: A job failed! ---")
            print(f"The following command failed to execute:\n{e.args}")
            print(f"\n--- Error message ---\n{e.stderr.strip()}")
            print("--- Aborting batch process. ---")
            sys.exit(1)

    if args.enhance_with_codeformer and os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)

    print(f"\n✅ Processing complete.")
    if args.enhance_with_codeformer:
        print(f"Final enhanced results are in: '{output_dir}'")
    else:
        print(f"Results saved in '{output_dir}'")

def main():
    parser = argparse.ArgumentParser(description="High-Quality Face Processor with Roop and CodeFormer")
    parser.add_argument('--run-setup', action='store_true', help="Run one-time setup for all tools.")
    parser.add_argument('--source-dir', type=str, help="Directory with source face images.")
    parser.add_argument('--target-dir', type=str, help="Directory with target images.")
    parser.add_argument('--output-dir', type=str, default="final_output", help="Final directory for all results.")
    parser.add_argument('--num-jobs', type=int, default=0, help="Total jobs to run.")
    parser.add_argument('--random-targets', action='store_true', help="Use random source/target pairs.")
    parser.add_argument('--roop-model', type=int, choices=[128, 512], default=512, help="Roop model resolution (128 for speed, 512 for quality).")
    parser.add_argument('--frame-processor', type=str, default='face_swapper', help="Roop processors, e.g., 'face_swapper,face_enhancer'.")
    parser.add_argument('--execution-provider', type=str, default='cuda', help="Provider for Roop: 'cuda' or 'cpu'.")
    parser.add_argument('--enhance-with-codeformer', action='store_true', help="Enable CodeFormer enhancement.")
    parser.add_argument('--codeformer-fidelity', type=float, default=0.7, help="CodeFormer fidelity (0=quality, 1=identity).")
    
    args = parser.parse_args()

    if args.run_setup:
        setup_environment()
    elif args.source_dir and args.target_dir:
        run_batch_processing(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
