"""
Apply certain compression technique to a model
"""

import argparse
import torch

def apply_compression(model, compression_technique):
    """
    Apply a specific compression technique to a model.

    Args:
        model (torch.nn.Module): The model to compress.
        compression_technique (str): The technique to apply, e.g., "quantisation", "pruning", "distillation".

    Returns:
        torch.nn.Module: The compressed model.
    """
    if compression_technique == "quantisation":
        # return quantise_model(model)
        return model
    elif compression_technique == "pruning":
        # return prune_model(model)
        return model
    elif compression_technique == "distillation":
        # return distill_model(model)
        return model
    else:
        raise ValueError(f"Unsupported compression technique: {compression_technique}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply compression to a model")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--technique", type=str, required=True, help="Compression technique to apply")
    args = parser.parse_args()

    # Load the model
    model = torch.load(args.model)

    # Apply compression technique
    compressed_model = apply_compression(model, args.technique)

    # Save the compressed model
    torch.save(compressed_model, f"{args.model}_{args.technique}.pth")