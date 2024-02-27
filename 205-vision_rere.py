import os
import time
import sys
from collections import namedtuple
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import torch
from IPython.display import HTML, FileLink, display
import matplotlib.pyplot as plt

from notebook_utils import load_image
from model.u2net import U2NET, U2NETP

model_config = namedtuple(
    "ModelConfig", ["name", "url", "model", "model_args"])

u2net_lite = model_config(
    name="u2net_lite",
    url="https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy",
    model=U2NETP,
    model_args=(),
)
u2net = model_config(
    name="u2net",
    url="https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
    model=U2NET,
    model_args=(3, 1),
)
u2net_human_seg = model_config(
    name="u2net_human_seg",
    url="https://drive.google.com/uc?id=1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P",
    model=U2NET,
    model_args=(3, 1),
)

# Set u2net_model to one of the three configurations listed above.
u2net_model = u2net_lite

# The filenames of the downloaded and converted models.
MODEL_DIR = "model"
model_path = Path(MODEL_DIR) / u2net_model.name / \
    Path(u2net_model.name).with_suffix(".pth")

if not model_path.exists():
    import gdown

    os.makedirs(name=model_path.parent, exist_ok=True)
    print("Start downloading model weights file... ")
    with open(model_path, "wb") as model_file:
        gdown.download(url=u2net_model.url, output=model_file)
        print(f"Model weights have been downloaded to {model_path}")

# Load the model.
net = u2net_model.model(*u2net_model.model_args)
net.eval()

# Load the weights.
print(f"Loading model weights from: '{model_path}'")
net.load_state_dict(state_dict=torch.load(model_path, map_location="cpu"))

model_ir = ov.convert_model(net, example_input=torch.zeros(
    (1, 3, 512, 512)), input=([1, 3, 512, 512]))

# Initialize the video capture from the camera (change the index to your camera number if needed)
cap = cv2.VideoCapture(0)

core = ov.Core()
device = 'CPU'
compiled_model_ir = core.compile_model(
    model=model_ir, device_name=device)

input_layer_ir = compiled_model_ir.input(0)
output_layer_ir = compiled_model_ir.output(0)

BACKGROUND_FILE = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/wall.jpg"
OUTPUT_DIR = "output"

os.makedirs(name=OUTPUT_DIR, exist_ok=True)

background_image = cv2.cvtColor(src=load_image(
    BACKGROUND_FILE), code=cv2.COLOR_BGR2RGB)

# Resize the background image to match the size of resized_result
background_image = cv2.resize(
    src=background_image, dsize=(512, 512))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the captured frame to RGB format
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the captured frame to the input size of the model
    resized_image = cv2.resize(src=image, dsize=(512, 512))

    # Convert the image shape to the expected format for OpenVINO IR model: (1, 3, 512, 512)
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

    # Perform inference on the input image using OpenVINO
    result = compiled_model_ir([input_image])[output_layer_ir]

    # Resize the network result to the image shape
    resized_result = np.rint(cv2.resize(src=np.squeeze(result), dsize=(
        image.shape[1], image.shape[0]))).astype(np.uint8)

    # Create a copy of the image and set all background values to 255 (white)
    bg_removed_result = image.copy()
    bg_removed_result[resized_result == 0] = 255

    # Set all the foreground pixels from the result to 0 in the background image
    background_image[resized_result == 1] = 0

    # Add the image with the background removed to the background image
    new_image = background_image + bg_removed_result

    # Display the original frame and the image with the new background
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Frame with New Background", new_image)

    # Save the generated image
    new2NET, U2NETP

model_config = namedtuple(
    "ModelConfig", ["name", "url", "model", "model_args"])

u2net_lite = model_config(
    name="u2net_lite",
    url="https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy",
    model=U2NETP,
    model_args=(),
)

u2net_model = u2net_lite

MODEL_DIR = "model"
model_path = Path(MODEL_DIR) / u2net_model.name / \
    Path(u2net_model.name).with_suffix(".pth")

if not model_path.exists():
    import gdown

    os.makedirs(name=model_path.parent, exist_ok=True)
    print("Start downloading model weights file... ")
    with open(model_path, "wb") as model_file:
        gdown.download(url=u2net_model.url, output=model_file)
        print(f"Model weights have been downloaded to {model_path}")

net = u2net_model.model(*u2net_model.model_args)
net.eval()

print(f"Loading model weights from: '{model_path}'")
net.load_state_dict(state_dict=torch.load(model_path, map_location="cpu"))

model_ir = ov.convert_model(net, example_input=torch.zeros(
    (1, 3, 512, 512)), input=([1, 3, 512, 512]))

input_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
input_scale = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)

cap = cv2.VideoCapture(0)

core = ov.Core()
device = "CPU"
compiled_model_ir = core.compile_model(model=model_ir, device_name=device)
input_layer_ir = compiled_model_ir.input(0)
output_layer_ir = compiled_model_ir.output(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(src=image, dsize=(512, 512))
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
    input_image = (input_image - input_mean) / input_scale

    result = compiled_model_ir([input_image])[output_layer_ir]
    resized_result = np.rint(cv2.resize(src=np.squeeze(result), dsize=(
        image.shape[1], image.shape[0]))).astype(np.uint8)

    bg_removed_result = image.copy()
    bg_removed_result[resized_result == 0] = 255

    BACKGROUND_FILE = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/wall.jpg"
    OUTPUT_DIR = "output"
    os.makedirs(name=OUTPUT_DIR, exist_ok=True)

    background_image = cv2.cvtColor(src=load_image(
        BACKGROUND_FILE), code=cv2.COLOR_BGR2RGB)
    background_image = cv2.resize(
        src=background_image, dsize=(image.shape[1], image.shape[0]))

    background_image = cv2.resize(
        src=background_image, dsize=(resized_result.shape[1], resized_result.shape[0]))

    background_image[resized_result == 1] = 0
    new_image = background_image + bg_removed_result

    new_image_path = Path(
        f"{OUTPUT_DIR}/{Path(IMAGE_URI).stem}-{Path(BACKGROUND_FILE).stem}.jpg")
    cv2.imwrite(filename=str(new_image_path),
                img=cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Frame with New Background", new_image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
