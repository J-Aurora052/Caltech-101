import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from dataset import Caltech101Dataset
from models import FineTunedResNet18


def load_model(model_path, num_classes):
    model = FineTunedResNet18(num_classes=num_classes, pretrained=False)


    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)

    model.eval()
    return model


def inverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.clamp_(0, 1)


def visualize_random_prediction(model, dataset, classes):
    idx = random.randint(0, len(dataset) - 1)
    image, true_label = dataset[idx]


    input_tensor = image.unsqueeze(0)


    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_label = predicted.item()
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]


    image = inverse_normalize(image).permute(1, 2, 0).numpy()


    plt.figure(figsize=(14, 6))


    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"True: {classes[true_label]}\nPredicted: {classes[predicted_label]}")
    plt.axis('off')


    plt.subplot(1, 2, 2)
    topk = min(5, len(classes))
    topk_probs, topk_indices = torch.topk(probabilities, topk)
    plt.barh(range(topk), topk_probs.numpy(), align='center')
    plt.yticks(range(topk), [classes[i] for i in topk_indices.numpy()])
    plt.xlabel("Probability")
    plt.title("Top Predictions")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    data_dir = './caltech101'
    model_path = './results/best_pretrained_standard.pth'


    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        full_dataset = Caltech101Dataset(
            root_dir=data_dir,
            transform=transform,
            exclude_background=True
        )


        model = load_model(model_path, num_classes=len(full_dataset.classes))

        print(f"Loaded model from {model_path}")
        print(f"Dataset contains {len(full_dataset)} images across {len(full_dataset.classes)} classes")
        print("Running on CPU")

        while True:
            visualize_random_prediction(model, full_dataset, full_dataset.classes)
            user_input = input("Press Enter to see another prediction, or 'q' to quit: ")
            if user_input.lower() == 'q':

                break

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the following paths:")
        print(f"- Data directory: {os.path.abspath(data_dir)}")
        print(f"- Model path: {os.path.abspath(model_path)}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")