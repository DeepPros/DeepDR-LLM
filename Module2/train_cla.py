import torch
from torch import nn
import os
from ourdataset import ListDataset
import torchvision.transforms as transforms
import numpy as np
from Models import DeepVit,pos_embed
import cv2
from utils import metrics,imgPro
import torch.nn.functional as F
from torchvision import datasets
import datetime
from torch.utils.tensorboard import SummaryWriter  

input_size= 448
class_nums = 5

MEAN = [108.64628601 / 255, 75.86886597 / 255, 54.34005737 / 255]
STD = [70.53946096 / 255, 51.71475228 / 255, 43.03428563 / 255]

##data augmentation
data_aug={
    'scale': (1 / 1.15, 1.15),
    'stretch_ratio': (0.7561, 1.3225),  # (1/(1.15*1.15) and 1.15*1.15)
    'ratation': (-180, 180),
    'translation_ratio': (40 / 448, 40 / 448),  # 40 pixel in the report
    'sigma': 0.5
}
transform_train = transforms.Compose([
transforms.RandomResizedCrop(
    size=input_size,
    scale=data_aug['scale'],
    ratio=data_aug['stretch_ratio']
),
transforms.RandomAffine(
    degrees=data_aug['ratation'],
    translate=data_aug['translation_ratio'],
    scale=None,
    shear=None
),
transforms.RandomHorizontalFlip(),
transforms.RandomVerticalFlip(),
transforms.ToTensor(),
transforms.Normalize(tuple(MEAN), tuple(STD)),
imgPro.KrizhevskyColorAugmentation(sigma=data_aug['sigma'])
])

transform_test = transforms.Compose([
    transforms.Resize((input_size,input_size)),
    transforms.ToTensor(),
    transforms.Normalize(tuple(MEAN), tuple(STD))
])


#data loader
train_data = ListDataset('../Module2/Datalist/Clatrain.txt',transform=transform_train)
val_data = ListDataset('../Module2/Datalist/Clatest.txt',transform=transform_test)

#default bs = 1, workers = 0
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1,shuffle=True, num_workers = 0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1,shuffle=False, num_workers = 0)

#model
model = DeepVit.vit_base_patch16(img_size=input_size, weight_init="nlhb",  cla_num_classes=class_nums)
checkpoint_model = torch.load('path/to/vit_base.pth')['model']
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]
# interpolate position embedding
checkpoint_model = pos_embed.interpolate_pos_embed(model, checkpoint_model)
# load pre-trained model
model.load_state_dict(checkpoint_model, strict=False)

#settings
epochs_num = 40
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
lossf = nn.CrossEntropyLoss() 
net_name = './cla_results'
writer = SummaryWriter(net_name+'/logs')


#training
print("Start Training!")
step = 0
model = model.cuda()
with open(net_name+"/acc.txt", "w") as f:
    for epoch in range(epochs_num):
        print(net_name+'\nEpoch: %d' % (epoch + 1))
        model.train()
        correct = 0
        total = 0
        correct = float(correct)
        total = float(total)
        for i, (_, inputs, labels) in enumerate(train_loader):
            start_time = datetime.datetime.now()
            batch = len(inputs)
            inputs = inputs.cuda()
            labels = (labels).cuda() 
            optimizer.zero_grad()
            # forward + backward
            outputs = model(inputs)
            loss = lossf(outputs, labels)
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('lr',optimizer.state_dict()['param_groups'][0]['lr'],step)
            loss.backward()
            optimizer.step()
            end_time = datetime.datetime.now()
            print('time:',end_time-start_time)
            step += 1
            print('batch:%d/%d, loss:%.4f, epoch:%d.'%(i,len(train_loader),loss, epoch))
        scheduler.step()

        trues = []
        preds = []

        # eval
        if True:
            print("Waiting Test!")
            if epoch % 1 == 0:
                print('Saving model......')
                if not os.path.exists(net_name+'/finals'):
                    os.mkdir(net_name+'/finals')
                torch.save(model.state_dict(), '%s/net_%03d.pth' % (net_name+'/finals', epoch + 1))
            with torch.no_grad():
                for i, (paths, images, labels)  in enumerate(val_loader):
                    model.eval()
                    images, labels = images.cuda(), labels.cuda()
                    outputs = model(images)
                    predicted = torch.argmax(outputs, 1)
                    trues.append(labels)
                    preds.append(predicted)
                    print("test_batch:{:5d}/{:5d}".format(i,len(val_loader)))
                trues = torch.cat(trues,dim=0).cpu().numpy()
                preds = torch.cat(preds,dim=0).cpu().numpy()
                if class_nums > 2:
                    classify_result = metrics.compute_prec_recal_mul(preds,trues)
                else:
                    classify_result = metrics.compute_prec_recal_binary(preds,trues)
                print('EPOCH:{:5d},Acc:{:.3f},Precision:{:.3f},Recall:{:.3f},F1:{:.3f}.\n'.format(epoch + 1,classify_result[0],classify_result[1],classify_result[2],classify_result[3]))
                f.write('EPOCH:{:5d},Acc:{:.3f},Precision:{:.3f},Recall:{:.3f},F1:{:.3f}.\n'.format(epoch + 1,classify_result[0],classify_result[1],classify_result[2],classify_result[3]))
                f.flush()
print("Training Finished!!!")