"""Train script.

Usage:
    infer_celeba.py <hparams> <dataset_root> <z_dir>
"""
import os
import cv2
import random
import torch
import vision
import numpy as np
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.config import JsonConfig
import imageio
from PIL import Image
from skimage import img_as_ubyte


def select_index(name, l, r, description=None, rand=True):
    index = random.randint(l, r - 1) if rand else None
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
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
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
    img = d.permute(1, 2, 0).detach().cpu().numpy()
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow(f"img_{i}", img)

def show_selection_belnd(z_base1, z_base2, z_base):
    base_img1 = (run_z(graph, z_base1) )
    base_img2 = (run_z(graph, z_base2) )
    blend_img = (run_z(graph, z_base) )
    cv2.imshow("Reconstructed Image 1", base_img1)
    cv2.imshow("Reconstructed Image 2", base_img2)
    cv2.imshow("blend_img", blend_img)


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

    hparams = JsonConfig("hparams/cars.json")
    dataset = vision.Datasets["stanfordcars"]
    # set transform of dataset
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
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

    if generate_z:
        delta_Z = graph.generate_attr_deltaz(dataset)
        for i, z in enumerate(delta_Z):
            np.save(os.path.join(z_dir, "detla_z_{}.npy".format(i)), z)
        print("Finish generating")

    # interact with user]
    graph.eval()
    use_indicees = False
    while True:
        if use_indicees:
            base_index1 = select_index("base image1", 0, len(dataset), rand=True)
            base_index2 = select_index("base image2", 0, len(dataset), rand=True)

            print(f"Using base indicees {base_index1} {base_index2}")

            print(dataset[base_index1]["x"].shape)

            i1 = dataset[base_index1]["x"]
            i2 = dataset[base_index2]["x"]

            z_base1 = graph.generate_z(i1)
            z_base2 = graph.generate_z(i2)

            print(dataset[base_index1]["path"])
            print(dataset[base_index2]["path"])

            draw_original(dataset[base_index1]["x"], 1)
            draw_original(dataset[base_index2]["x"], 2)

            z_blend = (z_base1 + z_base2) / 2
            blend_img = run_z(graph, z_blend)

            show_selection_belnd(z_base1, z_base2, z_blend)

            #cv2.waitKey(0)

            cv2.destroyAllWindows()

            print(z_base1.shape)
        else:
            print('Type file names without extension')

            base_index1 = input()
            base_index2 = input()

            base_image1 = Image.open('/home/mert/Downloads/cars_train/{}.jpg'.format(base_index1))
            base_image1 = transform(base_image1)
            draw_original(base_image1, 1)

            base_image2 = Image.open('/home/mert/Downloads/cars_train/{}.jpg'.format(base_index2))
            base_image2 = transform(base_image2)
            draw_original(base_image2, 2)

            z_base1 = graph.generate_z(base_image1)
            z_base2 = graph.generate_z(base_image2)

            show_selection_belnd(z_base1, z_base2, (z_base1+z_base2)/2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # 4377 7818
        # 10597 14371

        images = []

        z_diff = z_base2 - z_base1
        z_int = z_base1
        for i in range(50):
            z_int += z_diff/50
            modified_image = (run_z(graph, z_int) )
            resized_image = cv2.resize(modified_image, (512, 512), interpolation=cv2.INTER_CUBIC)
            norm_image = cv2.normalize(resized_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            norm_image = norm_image.astype(np.uint8)
            images.append(norm_image)
            if i == 0 or i == 49:
                for j in range(3):
                    images.append(norm_image)
            #cv2.imshow(f"blend_img_{i}", modified_image)
            #cv2.waitKey()

        try:
            imageio.mimsave('gifs/{}_{}.gif'.format(base_index1, base_index2), images)
        except Exception as e:
            print('some went wrong', e)
            pass
        continue

        interplate_n = 10
        for i in range(-interplate_n, interplate_n + 1, 2):
            d = z_delta * float(i) / float(interplate_n)
            modified_image = run_z(graph, z_blend - d)
            cv2.imshow(f"blend_img_{i}", modified_image)
        cv2.waitKey()
        cv2.destroyAllWindows()