import sys
import os

sys.path.append(os.path.abspath("ALMA-Matters/compression/pruning/TransformerCompression"))

from TransformerCompression.experiments.run_slicegpt import slicing_args_parser, process_slicing_args, slicing_main

args = [
    "--model", "haoranxu/ALMA-7B",
    "--save-dir", "ALMA-Matters/models/",
    "--sparsity", "0.25",
    "--device", "cuda:0",
    "--eval-baseline",
    "--no-wandb"
]


if __name__ == "__main__":
    sys.argv = args
    slicing_args = slicing_args_parser()
    process_slicing_args(slicing_args)
    slicing_main(slicing_args) 
