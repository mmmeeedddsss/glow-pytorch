from .datasets import StanfordCars
from .datasets import CelebADataset

Datasets = {
    "celeba": CelebADataset,
    'stanfordcars': StanfordCars,
}
