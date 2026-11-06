"""Download PathoROB Tolkach ESCA from the Hugging Face Hub and save it to disk.

    python scripts/download_data.py --out datasets/tolkach_esca
"""
import argparse

from datasets import load_dataset

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--out", default="datasets/tolkach_esca")
args = p.parse_args()

ds = load_dataset("bifold-pathomics/PathoROB-tolkach_esca", split="train")
print(ds)
ds.save_to_disk(args.out)
