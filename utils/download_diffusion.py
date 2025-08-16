#!/usr/bin/env python3
"""
Script to download Sana 600M checkpoint from HuggingFace.
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("huggingface-hub not installed. Installing...")
    os.system(f"{sys.executable} -m pip install huggingface-hub")
    from huggingface_hub import snapshot_download, hf_hub_download


def download_sana_checkpoint(
    model_size: str = "600M",
    resolution: str = "512px",
    output_dir: str = "./checkpoints",
):
    """
    Download Sana checkpoint from HuggingFace.

    Args:
        model_size: Model size (600M or 1600M)
        resolution: Resolution (512px or 1024px)
        output_dir: Directory to save the checkpoint
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Repository ID
    repo_id = f"Efficient-Large-Model/Sana_{model_size}_{resolution}"

    print(f"Downloading {repo_id} from HuggingFace...")

    try:
        # Download all files to a temporary location
        cache_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=output_dir / "temp_cache",
            resume_download=True,
        )

        print(f"Downloaded to cache: {cache_dir}")

        # Find the .pth or .pt file
        checkpoint_file = None
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                if file.endswith(('.pth', '.pt', '.ckpt')):
                    checkpoint_file = os.path.join(root, file)
                    break
            if checkpoint_file:
                break

        if checkpoint_file:
            # Define output filename
            output_file = output_dir / f"sana_{model_size.lower()}.pt"

            # Copy or move the checkpoint
            import shutil
            shutil.copy2(checkpoint_file, output_file)

            print(f"✓ Checkpoint saved to: {output_file}")

            # Clean up cache if desired
            response = input("Remove temporary cache files? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(output_dir / "temp_cache")
                print("✓ Cache cleaned up")
        else:
            print("❌ No checkpoint file found in the downloaded repository")
            print("Files downloaded:")
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    print(f"  - {os.path.relpath(os.path.join(root, file), cache_dir)}")

    except Exception as e:
        print(f"❌ Error downloading checkpoint: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Download Sana checkpoint from HuggingFace")
    parser.add_argument(
        "--model-size",
        type=str,
        default="600M",
        choices=["600M", "1600M"],
        help="Model size to download",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="512px",
        choices=["512px", "1024px"],
        help="Model resolution",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save the checkpoint",
    )

    args = parser.parse_args()

    success = download_sana_checkpoint(
        model_size=args.model_size,
        resolution=args.resolution,
        output_dir=args.output_dir,
    )

    if success:
        print("\n✅ Download completed successfully!")
        print(f"You can now run inference with:")
        print(f"  python inference.py --checkpoint {args.output_dir}/sana_{args.model_size.lower()}.pt")
    else:
        print("\n❌ Download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
