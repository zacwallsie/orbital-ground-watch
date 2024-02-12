import argparse
import torch
import numpy as np
from utils.dataset import create_data_loaders
import os
from models.sunet import SNUNet_ECAM



def load_model(model_path, device, config_path):
    model = SNUNet_ECAM(in_channels=6, num_classes=1, base_channel=72)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def predict(model, prediction_loader, threshold=0.5, device="mps"):
    model.eval()
    all_images = []
    all_masks = []
    all_predicted_masks = []

    with torch.no_grad():
        for batch, mask in prediction_loader:
            batch = batch.to(device)

            xA, xB = torch.split(batch, 6, dim=1)
            output = model(xA, xB)
            pred = torch.where(
                output > threshold,
                torch.ones_like(output),
                torch.zeros_like(output),
            ).long()
            pred = pred.permute(0, 2, 3, 1)
            all_images.append(batch.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            all_predicted_masks.append(pred.cpu().numpy())

    return np.concatenate(all_images, axis=0),\
        np.concatenate(all_masks, axis=0),\
        np.concatenate(all_predicted_masks, axis=0)


def save_arrays(predictions, output_path):
    np.save(output_path, predictions)
    print(f"Predicted masks saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Model Prediction Script")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the saved model file."
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input numpy file."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="orbo/config.json",
        help="Path to the model configuration file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="prediction_output",
        help="Path to save the predicted masks.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for selecting the top class.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for prediction."
    )

    parser.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="The path to the .npy file containing the un-enhanced image data",
    )

    args = parser.parse_args()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = load_model(args.model_path, device, args.config_path)

    # Prepare dataset
    train_loader, test_loader, val_loader = create_data_loaders(
        args.input_path, args.mask_path, args.batch_size
    )

    x, y, predicted_masks = predict(model, val_loader, args.threshold, device)

    save_arrays(x, os.path.join(args.output_path, "x.npy"))
    save_arrays(y, os.path.join(args.output_path, "y.npy"))
    save_arrays(predicted_masks, os.path.join(args.output_path, "predicted_masks.npy"))


if __name__ == "__main__":
    main()
