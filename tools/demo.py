import torch
from torch import nn
from PIL import Image
import torchvision.transforms as T
from modeling.backbones.resnet import ResNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride):
        super(Baseline, self).__init__()
        self.base = ResNet(last_stride)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        
        self.rank_bn = nn.BatchNorm1d(self.in_planes)
        self.rank_bn.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

    def forward(self, x):

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

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
    model = Baseline(num_classes = 702,last_stride =1)
    model.load_param('work_sapcet1r0.7/resnet50_model_130.pth')
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
    dist = euclidean_dist_rank(feats,feats)
    print(dist)

if __name__ == '__main__':
    main()
    
