import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def transpose(input_array):
    # Transpose the arrays from (N, H, W, C) to (N, C, H, W)
    transposed_array = np.transpose(input_array, (0, 3, 1, 2))

    return transposed_array


def main(input_file, output_dir):
    # Load the input NumPy array
    input_array = np.load(input_file)

    # Split and transpose the image pairs
    result1 = transpose(input_array)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the resulting NumPy arrays
    np.save(os.path.join(output_dir, "y_transposed.npy"), result1)

    print("Input shape:", input_array.shape)
    print("Output 1 shape:", result1.shape)
    print(f"Results saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transpose masks from a NumPy dataset."
    )
    parser.add_argument(
        "input_file", help="Path to the input NumPy dataset file (.npy)"
    )
    parser.add_argument(
        "output_dir", help="Directory to save the resulting NumPy datasets"
    )
    args = parser.parse_args()

    main(args.input_file, args.output_dir)
