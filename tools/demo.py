import torch
from torch import nn
from PIL import Image
import torchvision.transforms as T
from modeling.baseline import Baseline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def euclidean_dist_rank(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def process_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    img = img.to(device)
    return img

normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = T.Compose([T.Resize([256, 128]),T.ToTensor(),normalize_transform])

def main():
    model = Baseline(num_classes = 702,last_stride =1, model_path = ' ', stn = 'no', model_name = 'resnet50_ibn_a', pretrain_choice = ' ')
    model.load_param('models/resnet50_ibn_a/duke_resnet50_ibn_a_model.pth')
    model.to(device)
    model.eval()

    feats = []
    with torch.no_grad():
        img1 = process_img('/home/zzg/Datasets/DukeReiD/DukeMTMC-reID/query/0033_c1_f0057706.jpg')
        feat1 = model(img1)
        feats.append(feat1)

        img2 = process_img('/home/zzg/Datasets/DukeReiD/DukeMTMC-reID/query/0033_c6_f0045755.jpg')
        feat2 = model(img2)
        feats.append(feat2)

        img3 = process_img('/home/zzg/Datasets/DukeReiD/DukeMTMC-reID/query/0034_c2_f0057453.jpg')
        feat3 = model(img3)
        feats.append(feat3)
    feats = torch.cat(feats, dim=0)
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    dist = euclidean_dist_rank(feats,feats)
    print(dist)

if __name__ == '__main__':
    main()
    
