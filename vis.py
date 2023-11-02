from model.PSP import PSPNet
import torch
from PIL import Image
from data.CDDataset import get_standard_transformations
import matplotlib.pyplot as plt
import numpy as np
from tools import IouCal


def show_single(image, location=None, save=False, name=None):
    # show single image
    image = np.array(image, dtype=np.uint8)
    fig = plt.figure()
    plt.imshow(image, cmap="gray")

    fig.set_size_inches(2048/100.0, 1024/100.0) #输出width*height像素
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    if save:
        plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.show()


def main():
    device = torch.device("cuda:1")
    model = PSPNet(2)
    checkpoint = torch.load("saved_model/" + dataset + "_" + str(epoch) + ".pt", map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()


    B_path = root + data_name
    l_path = root + data_name2
    std = get_standard_transformations()
    img_B = std(Image.open(B_path).convert("RGB"))
    img_B = img_B.to(device, dtype=torch.float32).unsqueeze(0)
    outputs = model(img_B)
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().detach().numpy()[0]
    preds[preds == 1] = 255

    img_lbl = np.array(Image.open(l_path))

    img_lbl[img_lbl > 0] = 255

    show_single(preds)
    show_single(img_lbl)


if __name__ == '__main__':
    dataset = "LEVIR-CD256"
    root = "/home/wangbowen/DATA/" + dataset + "/"
    data_name = "B/train_34_4.png"
    data_name2 = "label/train_34_4.png"
    epoch = 14
    main()