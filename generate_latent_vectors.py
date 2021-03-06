"""Train script.

Usage:
    infer_celeba.py <hparams> <dataset_root> <z_dir>
"""
import os
import cv2
import pickle
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import vision
import numpy as np
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.config import JsonConfig


def select_index(name, l, r, description=None):
    index = None
    while index is None:
        print("Select {} with index [{}, {}),"
              "or {} for random selection".format(name, l, r, l - 1))
        if description is not None:
            for i, d in enumerate(description):
                print("{}: {}".format(i, d))
        try:
            line = int(input().strip())
            if l - 1 <= line < r:
                index = line
                if index == l - 1:
                    index = random.randint(l, r - 1)
        except Exception:
            pass
    return index


def run_z(graph, z):
    graph.eval()
    x = graph(z=torch.tensor([z]).cuda(), eps_std=0.3, reverse=True)
    img = x[0].permute(1, 2, 0).detach().cpu().numpy()
    img = img[:, :, ::-1]
    img = cv2.resize(img, (256, 256))
    return img


def save_images(images, names):
    if not os.path.exists("pictures/infer/"):
        os.makedirs("pictures/infer/")
    for img, name in zip(images, names):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite("pictures/infer/{}.png".format(name), img)
        cv2.imshow("img", img)
        cv2.waitKey()


def draw_original(d, i):
    img = d.permute(1, 2, 0).contiguous().numpy()
    img = cv2.resize(img, (256, 256))
    cv2.imshow(f"img_{i}", img)


def show_selection_belnd(z_base1, z_base2, z_base):
    base_img1 = (run_z(graph, z_base1))
    base_img2 = (run_z(graph, z_base2))
    blend_img = (run_z(graph, z_base))
    cv2.imshow("base_img1", base_img1)
    cv2.waitKey()
    cv2.imshow("base_img2", base_img2)
    cv2.waitKey()
    cv2.imshow("blend_img", blend_img)
    cv2.waitKey()


if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset_root = args["<dataset_root>"]
    z_dir = args["<z_dir>"]
    assert os.path.exists(dataset_root), (
        "Failed to find root dir `{}` of dataset.".format(dataset_root))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))
    if not os.path.exists(z_dir):
        print("Generate Z to {}".format(z_dir))
        os.makedirs(z_dir)
        generate_z = True
    else:
        print("Load Z from {}".format(z_dir))
        generate_z = False

    hparams = JsonConfig("hparams/celeba.json")
    dataset = vision.Datasets["celeba"]
    # set transform of dataset
    transform = transforms.Compose([
        transforms.CenterCrop(hparams.Data.center_crop),
        transforms.Resize(hparams.Data.resize),
        transforms.ToTensor()])
    # build
    graph = build(hparams, False)["graph"]
    dataset = dataset(dataset_root, transform=transform)

    # get Z
    if not generate_z:
        # try to load
        try:
            delta_Z = []
            for i in range(hparams.Glow.y_classes):
                z = np.load(os.path.join(z_dir, "detla_z_{}.npy".format(i)))
                delta_Z.append(z)
        except FileNotFoundError:
            # need to generate
            generate_z = True
            print("Failed to load {} Z".format(hparams.Glow.y_classes))
            quit()
    if generate_z:
        delta_Z = graph.generate_attr_deltaz(dataset)
        for i, z in enumerate(delta_Z):
            np.save(os.path.join(z_dir, "detla_z_{}.npy".format(i)), z)
        print("Finish generating")

    mp = {}

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             num_workers=4,
                             shuffle=True,
                             drop_last=True)

    progress = tqdm(data_loader)
    graph.eval()
    for i_batch, batch in enumerate(progress):

        img = np.squeeze(batch["x"], axis=0)
        path = np.squeeze(batch["path"], axis=0)
        path = str(path).split('/')[-1]

        z = graph.generate_z(img)

        mp[path] = z

        if i_batch % 1000 == 0:
            with open('z_path_map_celeba.pkl', 'wb+') as f:
                pickle.dump(mp, f, pickle.HIGHEST_PROTOCOL)

        continue
        base_index1 = select_index("base image1", 0, len(dataset))
        base_index2 = select_index("base image2", 0, len(dataset))

        print(f"Using base indicees {base_index1} {base_index2}")

        z_base1 = graph.generate_z(dataset[base_index1]["x"])
        z_base2 = graph.generate_z(dataset[base_index2]["x"])

        print(z_base2.shape)

        draw_original(dataset[base_index1]["x"], 1)
        draw_original(dataset[base_index2]["x"], 2)

        z_blend = (z_base1 + z_base2) / 2
        blend_img = run_z(graph, z_blend)

        show_selection_belnd(z_base1, z_base2, z_blend)

        cv2.destroyAllWindows()

        print(z_base1.shape)

        for i in range(10):
            print(z_base1.shape)
            modified_image = (run_z(graph, z_base1))
            cv2.imshow(f"blend_img_{i}", modified_image)
            cv2.waitKey()
            z_base1 += 0.02

        continue

        interplate_n = 10
        for i in range(-interplate_n, interplate_n + 1, 2):
            d = z_delta * float(i) / float(interplate_n)
            modified_image = run_z(graph, z_blend - d)
            cv2.imshow(f"blend_img_{i}", modified_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
