from PIL import Image
import os
from tqdm import tqdm

def rescale_images(directory):

    for img in tqdm(os.listdir(directory)):
        path = os.path.join(directory, img)
        img = Image.open(path).convert('RGB')
        img.resize((600, 600))
        img.save(path, 'JPEG')
if __name__ == '__main__':
    path = "/home/ashish/models/research/dataset/preprocessing/images/test"
    rescale_images(path)