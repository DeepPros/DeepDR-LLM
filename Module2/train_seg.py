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
import albumentations as A
from albumentations.pytorch import ToTensorV2

input_size= 448
class_nums = 5

##data augmentation
transform_train = A.Compose([
    A.LongestMaxSize(max_size=896),
    A.PadIfNeeded(min_height=896, min_width=896, border_mode=0),
    # A.CenterCrop(height=int(448), width=int(1024*0.7)), 
    A.RandomCrop(height=448, width=448, p=0.5),
    A.Resize(height=input_size, width=input_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(),
    ToTensorV2()
])
transform_test = A.Compose([
    A.LongestMaxSize(max_size=896),
    A.PadIfNeeded(min_height=896, min_width=896, border_mode=0),
    A.Normalize(),
    ToTensorV2()
])


#data loader
train_data = ListDataset('../Module2/Datalist/Segtrain.txt',transform=transform_train, seg_flag=True)
val_data = ListDataset('../Module2/Datalist/Segtest.txt',transform=transform_test, seg_flag=True)
#default bs = 1, workers = 0
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1,shuffle=True, num_workers = 0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1,shuffle=False, num_workers = 0)

#model
model = DeepVit.vit_base_patch16(img_size=input_size, weight_init="nlhb",  seg_num_classes=class_nums)
checkpoint_model = torch.load('path/to/vit_base.pth')['model']
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model:
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
net_name = './seg_results'
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
            loss_main = lossf(outputs[-1], labels)
            aux_loss_1 = lossf(outputs[0], labels)
            aux_loss_2 = lossf(outputs[1], labels)
            aux_loss_3 = lossf(outputs[2], labels)
            loss = loss_main + 0.2*aux_loss_1 + 0.3*aux_loss_2 + 0.4*aux_loss_3

            writer.add_scalar('loss', loss, step)
            writer.add_scalar('lr',optimizer.state_dict()['param_groups'][0]['lr'],step)
            loss.backward()
            optimizer.step()
            end_time = datetime.datetime.now()
            print('time:',end_time-start_time)
            step += 1
            print('batch:%d/%d, loss:%.4f, epoch:%d.'%(i,len(train_loader),loss, epoch))
        scheduler.step()

        # eval
        labels_group = []
        outputs_group = []

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
                    images, labels = images.cuda(), labels[0].cuda()
                    output = imgPro.split_image_to_4(images)
                    merge_out = []
                    for i in range(len(output)):
                        merge_out.append(model(output[i])[-1][0])

                    result = imgPro.merge_4_image(merge_out[0],merge_out[1],merge_out[2],merge_out[3])
                    result = torch.argmax(result, dim=0)
                    labels = torch.argmax(labels, dim=0)

                    labels_group.append(labels.detach().cpu().numpy())
                    outputs_group.append(result.detach().cpu().numpy())
                ious, f1_scores = metrics.compute_dataset_metrics(labels_group, outputs_group, num_classes=class_nums)
                for idx in range(class_nums):
                    print(f"Epoch:{epoch}, Class {idx} - IoU: {ious[idx]:.4f}, F1 score: {f1_scores[idx]:.4f}.")
                    f.write(f"Epoch:{epoch}, Class {idx} - IoU: {ious[idx]:.4f}, F1 score: {f1_scores[idx]:.4f}.")
                f.flush()
        break

print("Training Finished!!!")