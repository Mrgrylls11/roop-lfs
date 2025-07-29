
import os
import sys
import subprocess
import argparse
import glob
import random
from tqdm import tqdm
import cv2

# --- Setup and Utility Functions ---
def run_command(command, description):
    print(description)
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to run command: '{command}'")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)

def setup_environment():
    base_dir = os.path.abspath(os.getcwd())
    repo_dir = os.path.join(base_dir, 'roop')
    if repo_dir not in sys.path: sys.path.insert(0, repo_dir)
    if not os.path.exists(repo_dir):
        run_command('git clone https://github.com/s0md3v/roop.git', "--- Cloning the roop repository ---")
    original_dir = os.getcwd()
    os.chdir(repo_dir)
    print("--- Installing dependencies (this may take a few minutes) ---")
    run_command('pip install -r requirements.txt -q', "--- Installing base requirements ---")
    run_command('pip install tqdm gfpgan insightface==0.7.3 onnxruntime-gpu==1.15.1 -q', "--- Installing GPU and Enhancement libraries ---")
    run_command('pip install --upgrade numpy==1.26.4 jax==0.4.23 jaxlib==0.4.23 -q', "--- Applying NumPy Compatibility Fix ---")
    if not os.path.exists('inswapper_128.onnx'):
        run_command('wget -q -P . https://huggingface.co/ai-forever/inswapper/resolve/main/inswapper_128.onnx', "--- Downloading Face Swapper Model ---")
    if not os.path.exists('GFPGANv1.3.pth'):
        run_command('wget -q https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P .', "--- Downloading GFPGAN Model ---")
    os.chdir(original_dir)
    print("\n\n✅✅✅ Full setup complete. The environment is ready. ✅✅✅\n")

# --- Main Application Logic ---
def process_images(args):
    # Set execution provider
    os.environ['ONNXRUNTIME_PROVIDERS'] = 'CUDAExecutionProvider' if 'cuda' in args.execution_provider else 'CPUExecutionProvider'
    print(f"--- Setting Execution Provider to: {os.environ['ONNXRUNTIME_PROVIDERS']} ---")

    from roop.core import process_image
    from roop.face_analyser import get_one_face
    from roop.utilities import resolve_relative_path

    # Validate paths and get image lists
    source_images = glob.glob(os.path.join(args.source_dir, "*"))
    target_images = glob.glob(os.path.join(args.target_dir, "*"))
    if not source_images or not target_images:
        print("Error: No images found in source or target directories.")
        return
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize enhancer if requested
    use_enhancer = 'face_enhancer' in [p.strip() for p in args.frame_processor.split(',')]
    enhancer = None
    if use_enhancer:
        print("--- Initializing GFPGAN (face_enhancer) ---")
        from gfpgan import GFPGANer
        enhancer = GFPGANer(model_path=resolve_relative_path('roop/GFPGANv1.3.pth'), upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)

    # --- Create job list based on user arguments ---
    jobs = []
    if args.random_targets:
        if args.num_jobs <= 0:
            print("Error: --num-jobs must be greater than 0 for random mode.")
            return
        print(f"--- Assigning {args.num_jobs} random jobs ---")
        for _ in range(args.num_jobs):
            jobs.append((random.choice(source_images), random.choice(target_images)))
    else:
        print("--- Assigning jobs sequentially ---")
        target_index = 0
        for source_path in source_images:
            for _ in range(2): # Each source image gets 2 target images
                if target_index >= len(target_images): break
                jobs.append((source_path, target_images[target_index]))
                target_index += 1
        # If --num-jobs is specified in sequential mode, trim the list
        if args.num_jobs > 0:
            jobs = jobs[:args.num_jobs]

    print(f"--- Created {len(jobs)} swap jobs to process ---")

    # --- Process all jobs ---
    for source_path, target_path in tqdm(jobs, desc="Processing Images"):
        try:
            source_face = get_one_face(cv2.imread(source_path))
            if not source_face:
                print(f"Warning: No face in {os.path.basename(source_path)}. Skipping.")
                continue
            target_img = cv2.imread(target_path)
            result_img = process_image(source_face, target_img)
            
            output_filename = f"{os.path.splitext(os.path.basename(source_path))[0]}_to_{os.path.splitext(os.path.basename(target_path))[0]}.png"
            output_path = os.path.join(args.output_dir, output_filename)
            
            if enhancer:
                _, _, result_img = enhancer.enhance(result_img, has_aligned=False, only_center_face=False, paste_back=True)

            cv2.imwrite(output_path, result_img)
        except Exception as e:
            print(f"Error on job ({os.path.basename(source_path)} to {os.path.basename(target_path)}): {e}")

    print(f"\n✅ Processing complete. Results saved in '{args.output_dir}'")

def main():
    parser = argparse.ArgumentParser(description="Final Batch Face Swapper")
    # Add all desired arguments
    parser.add_argument('--run-setup', action='store_true', help="Run the one-time setup.")
    parser.add_argument('--source-dir', type=str, help="Directory with source face images.")
    parser.add_argument('--target-dir', type=str, help="Directory with target images.")
    parser.add_argument('--output-dir', type=str, default="batch_output", help="Directory for saved results.")
    parser.add_argument('--num-jobs', type=int, default=0, help="Total jobs to run. 0 means all possible pairs in sequential mode.")
    parser.add_argument('--random-targets', action='store_true', help="Use random source/target pairs.")
    parser.add_argument('--frame-processor', type=str, default='face_swapper', help="Processes: 'face_swapper' and optionally ',face_enhancer'.")
    parser.add_argument('--execution-provider', type=str, default='cuda', help="Provider: 'cuda' or 'cpu'.")
    # Dummy arguments to prevent errors
    parser.add_argument('--temp-frame-quality', type=int, help="[Ignored]")
    parser.add_argument('--max-memory', type=int, help="[Ignored]")
    
    args = parser.parse_args()

    repo_dir = os.path.abspath('roop')
    if repo_dir not in sys.path: sys.path.insert(0, repo_dir)

    if args.run_setup:
        setup_environment()
    elif args.source_dir and args.target-dir:
        if not os.path.exists(repo_dir):
            print("Roop directory not found. Please run with '--run-setup' first.")
            return
        process_images(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
