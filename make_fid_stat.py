import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import torch
import os

from torchvision.models import inception_v3
os.environ["CUDA_VISIBLE_DEVICES"] = "6" 
def get_activations(images, model, batch_size=64, device='cuda'):
    model.eval()
    n_images = images.shape[0]
    pred_arr = np.empty((n_images, 2048))
    tf = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    for i in tqdm(range(0, n_images, batch_size)):
        batch = images[i:i+batch_size]
        imgs = [tf(Image.fromarray(img.astype(np.uint8))) for img in batch]
        imgs = torch.stack(imgs).to(device)
        with torch.no_grad():
            pred = model(imgs)
            pred = pred.view(pred.size(0), -1).cpu().numpy()
        pred_arr[i:i+batch.shape[0]] = pred
    return pred_arr

def main():
    npz_path = '/data/PengBoYi/PSOGAN/fid_stat/ImageNet32_train_all.npz'  # 修改为你的路径
    save_path = '/data/PengBoYi/PSOGAN/fid_stat/fid_stat_imagenet32.npz'   # 修改为你的保存路径

    data = np.load(npz_path)
    images = data['arr_0']  # 根据你的npz实际key修改

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()  # 去掉最后的分类层

    feats = get_activations(images, inception, batch_size=64, device=device)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    np.savez(save_path, mu=mu, sigma=sigma)
    print(f'Saved FID stats to {save_path}')

if __name__ == '__main__':
    main()