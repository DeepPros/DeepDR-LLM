import torch
from torch import nn
from tqdm import tqdm
import pandas as pd
import os
import torchvision.transforms as transforms
import numpy as np
from Models import DeepVit,pos_embed
from utils import metrics,imgPro
from PIL import Image
from torch.utils.data import Dataset

class ListDataset(Dataset):
    def __init__(self, list_path,transform = None):
        self.transform = transform
        self.img_files = list_path

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_files)

INPUT_SIZE= 448
MEAN = [108.64628601 / 255, 75.86886597 / 255, 54.34005737 / 255]
STD = [70.53946096 / 255, 51.71475228 / 255, 43.03428563 / 255]
transform_test = transforms.Compose([
    transforms.Resize((INPUT_SIZE,INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(tuple(MEAN), tuple(STD))
])

Weights ={"DR":['path/to/DR.pth',5],
"DME":['path/to/DME.pth',2],
"Qua":['path/to/QUA.pth',3],
"RDR":['path/to/RDR.pth',2],
"Seg":['path/to/Seg.pth',5],
}

sets_root = 'path/to/dataDir'
datasets_names = os.listdir(sets_root)
imgs = []
dat = pd.DataFrame()
for i in datasets_names:
    path = os.path.join(sets_root,i)
    imgs.append(path)
rdrs = []
drs = []
dmes = []
quas = []
##DR
DR_model = DeepVit.vit_base_patch16(img_size=INPUT_SIZE, weight_init="nlhb",  cla_num_classes=Weights["DR"][1])
DR_model.load_state_dict(torch.load(Weights["DR"][0]), strict=True)
DR_model = DR_model.cuda()
##DME
DME_model = DeepVit.vit_base_patch16(img_size=INPUT_SIZE, weight_init="nlhb",  cla_num_classes=Weights["DME"][1])
DME_model.load_state_dict(torch.load(Weights["DME"][0]), strict=True)
DME_model = DME_model.cuda()
##RDR
RDR_model = DeepVit.vit_base_patch16(img_size=INPUT_SIZE, weight_init="nlhb",  cla_num_classes=Weights["RDR"][1])
RDR_model.load_state_dict(torch.load(Weights["RDR"][0]), strict=True)
RDR_model = RDR_model.cuda()
#Qua
Qua_model = DeepVit.vit_base_patch16(img_size=INPUT_SIZE, weight_init="nlhb",  cla_num_classes=Weights["Qua"][1])
Qua_model.load_state_dict(torch.load(Weights["Qua"][0]), strict=True)
Qua_model = Qua_model.cuda()
#Seg
Seg_model = DeepVit.vit_base_patch16(img_size=INPUT_SIZE, weight_init="nlhb",  seg_num_classes=Weights["Seg"][1])
Seg_model.load_state_dict(torch.load(Weights["Seg"][0]), strict=True)
Seg_model = Seg_model.cuda()

val_data = ListDataset(imgs,transform=transform_test)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1,shuffle=False,num_workers = 0)
for i,img in enumerate(val_loader):
    DR = torch.argmax(DR_model(img.cuda()),dim=1).squeeze().detach().cpu().numpy()
    DME = torch.argmax(DME_model(img.cuda()),dim=1).squeeze().detach().cpu().numpy()
    RDR = torch.argmax(RDR_model(img.cuda()),dim=1).squeeze().detach().cpu().numpy()

    output = imgPro.split_image_to_4(img)
    merge_out = []
    for i in range(len(output)):
        merge_out.append(Seg_model(output[i])[-1][0])
    result = imgPro.merge_4_image(merge_out[0],merge_out[1],merge_out[2],merge_out[3])
    result = torch.argmax(result, dim=0).detach().cpu().numpy()
    np.savetxt(f'./{i}.npy', result)
    drs += list(DR)
    dmes += list(DME)
    rdrs += list(RDR)
    QUA = torch.argmax(Qua_model(img.cuda()),dim=1).squeeze().detach().cpu().numpy()
    quas += list(QUA)
    print(f'now dataset is {i}-th/{len(val_loader)} imgs.')
dat["img"] = imgs
dat["DR pred"] = drs
dat["DME pred"] = dmes
dat["Qua pred"] = quas
dat["RDR pred"] = rdrs
dat.to_csv('./preds.csv')
