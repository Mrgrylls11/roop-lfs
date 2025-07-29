import os
import sys
import subprocess
import argparse
import glob
import random
from tqdm import tqdm
import cv2

def run_command(command, description):
    """Runs a shell command and checks for errors."""
    print(description)
    try:
        # Using subprocess.run for better error handling
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to run command: '{command}'")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)

def setup_environment():
    """Clones repo, installs dependencies, and downloads models."""
    base_dir = os.path.abspath(os.getcwd())
    repo_dir = os.path.join(base_dir, 'roop')

    # Add roop to Python path to allow imports from anywhere
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    if not os.path.exists(repo_dir):
        run_command('git clone https://github.com/s0md3v/roop.git', "--- Cloning the roop repository ---")

    # Change to roop directory to run installations from within it
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
    
    # Return to the original directory
    os.chdir(original_dir)
    print("\n\n✅✅✅ Full setup complete. The environment is ready. ✅✅✅\n")


# --- Main Application Logic ---
def process_images(args):
    # Set the provider to CUDA for GPU acceleration
    os.environ['ONNXRUNTIME_PROVIDERS'] = 'CUDAExecutionProvider'
    
    from roop.core import process_image
    from roop.face_analyser import get_one_face
    from roop.face_swapper import get_face_swapper
    from roop.utilities import resolve_relative_path

    # Validate paths
    if not os.path.isdir(args.source_dir) or not os.path.isdir(args.target_dir):
        print(f"Error: Source ('{args.source_dir}') or Target ('{args.target_dir}') directory not found.")
        return

    source_images = glob.glob(os.path.join(args.source_dir, "*"))
    target_images = glob.glob(os.path.join(args.target_dir, "*"))

    if not source_images or not target_images:
        print("Error: No images found in source or target directories.")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)

    # Load GFPGAN enhancer only if requested
    enhancer = None
    if args.enhance_with_gfpgan:
        print("--- Initializing GFPGAN for enhancement ---")
        from gfpgan import GFPGANer
        enhancer_model_path = resolve_relative_path('roop/GFPGANv1.3.pth')
        enhancer = GFPGANer(model_path=enhancer_model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)

    # --- Create job list ---
    jobs = []
    if args.random_targets:
        print(f"--- Assigning {args.num_jobs} random jobs ---")
        if args.num_jobs <= 0:
            print("Error: --num-jobs must be greater than 0 for random mode.")
            return
        for _ in range(args.num_jobs):
            jobs.append((random.choice(source_images), random.choice(target_images)))
    else:
        print("--- Assigning jobs sequentially ---")
        target_index = 0
        for source_path in source_images:
            # Each source image swaps with two target images
            for _ in range(2):
                if target_index >= len(target_images): break
                jobs.append((source_path, target_images[target_index]))
                target_index += 1
    
    # Trim jobs if a specific number is requested for sequential mode
    if not args.random_targets and args.num_jobs > 0:
        jobs = jobs[:args.num_jobs]

    print(f"--- Created {len(jobs)} swap jobs to process ---")
    
    # --- Process all jobs ---
    for source_path, target_path in tqdm(jobs, desc="Swapping faces"):
        try:
            source_face = get_one_face(cv2.imread(source_path))
            if not source_face:
                print(f"Warning: No face found in source: {os.path.basename(source_path)}. Skipping.")
                continue

            target_img = cv2.imread(target_path)
            result_img = process_image(source_face, target_img)
            
            # Construct a unique output path
            source_name = os.path.splitext(os.path.basename(source_path))[0]
            target_name = os.path.splitext(os.path.basename(target_path))[0]
            output_filename = f"{source_name}_to_{target_name}.png"
            output_path = os.path.join(args.output_dir, output_filename)
            
            # Enhance with GFPGAN if requested
            if enhancer:
                _, _, result_img = enhancer.enhance(result_img, has_aligned=False, only_center_face=False, paste_back=True)

            cv2.imwrite(output_path, result_img)
        except Exception as e:
            print(f"Error processing job ({os.path.basename(source_path)} to {os.path.basename(target_path)}): {e}")


    print(f"\n✅ Processing complete. Results saved in '{args.output_dir}'")

def main():
    parser = argparse.ArgumentParser(description="Batch Face Swapping with GFPGAN Enhancement")
    parser.add_argument('--run-setup', action='store_true', help="IMPORTANT: Run this first to clone the repo and install all dependencies.")
    parser.add_argument('--source-dir', type=str, help="Directory containing source face images.")
    parser.add_argument('--target-dir', type=str, help="Directory containing target images.")
    parser.add_argument('--output-dir', type=str, default="batch_output", help="Directory to save the results. Default: batch_output")
    parser.add_argument('--num-jobs', type=int, default=0, help="Total jobs to run. In sequential mode, 0 means all possible pairs. In random mode, this sets the number of random pairs to generate.")
    parser.add_argument('--random-targets', action='store_true', help="Use random source/target pairs instead of the sequential method.")
    parser.add_argument('--enhance-with-gfpgan', action='store_true', help="Enhance the swapped faces with GFPGAN for better quality.")

    args = parser.parse_args()

    # Add roop to Python path to ensure it can be imported
    repo_dir = os.path.abspath('roop')
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    if args.run_setup:
        setup_environment()
        # After setup, show help if no other args were passed, so user knows what to do next.
        if len(sys.argv) == 2:
            parser.print_help()
    elif args.source_dir and args.target_dir:
        if not os.path.exists(repo_dir):
            print("Roop directory not found. Please run with '--run-setup' first.")
            return
        process_images(args)
    else:
        # Show help if the script is called with no arguments
        parser.print_help()

if __name__ == '__main__':
    main()
