import torch
import yaml
import sys
import os

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.vit import ViT_model
from transformers import ViTImageProcessor

def main():
    print('ViT for CIFAR10 evaluation')

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = './config/cifar10_vit.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    model_name = params['model']
    image_processor = ViTImageProcessor.from_pretrained(model_name)
    
    # GPU 사용이 가능하면 사용하고, 불가능하면 CPU 활용
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    timestamp = "1749038439"
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp)))

    # 데이터 로드
    if params['task'] == "ImageNet":
        pass
    elif params['task'] == "CIFAR10":
        transforms_test = transforms.Compose([
            transforms.Lambda(lambda x: image_processor(images=x, return_tensors="pt", do_rescale=True, do_normalize=True))
        ])

    imgs = ImageFolder('scripts/example', transform=transforms_test)
    print("imgs:", imgs)
    inference_loader = torch.utils.data.DataLoader(imgs, batch_size=1)

    print("test_loader:", inference_loader)
    print(inference_loader.dataset)
    # 학습 모델 생성
    model = ViT_model(model_name).to(device)  # 모델을 지정한 device로 올려줌, dropout x

    # test 시작
    model.eval()

    # 저장된 state 불러오기
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints/best.pth"))

    # TODO : 세팅값 마다 save_path를 바꾸어 로드
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for thisimg, label in inference_loader:
        plt.imshow(thisimg['pixel_values'].squeeze().permute(1, 2, 0))
        plt.show()

        thisimg = thisimg.to(device)
        thisimg['pixel_values'] = thisimg['pixel_values'].squeeze().unsqueeze(0)
        pred = model.forward(thisimg)
        _, top_pred = torch.topk(pred, k=1, dim=-1)
        top_pred = top_pred.squeeze(dim=1)
        print("--------------------------------------")
        print("truth:", classes[label])
        print("model prediction:", classes[top_pred])

if __name__ == "__main__":
    main()