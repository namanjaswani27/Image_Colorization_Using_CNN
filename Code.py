
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2lab, rgb2grey, lab2rgb
from tqdm import tqdm
from shutil import copyfile
import time

import os
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets





os.makedirs('images/train/class/', exist_ok=True)
os.makedirs('images/val/class/', exist_ok=True)

st = time.time()

for i, file in tqdm(enumerate(os.listdir('Images_Places/images/'))):
    if i > 1000 and i <= 2000: 
        copyfile(src='Images_Places/images/'+file, dst='images/val/class/'+file)
    else: 
        copyfile(src='Images_Places/images/'+file, dst='images/train/class/'+file)
print(time.time()-st)




BATCH_SIZE=128
SHUFFLE=True
EPOCHS=20

# --------------------------------- Preprocessing -----------------------

class LABImageGeneration(datasets.ImageFolder):
    
    def __getitem__(self, idx):
        path, target = self.imgs[idx]
        img = self.loader(path)
#         print(type(img))
        
        if self.transform is not None:
            img = np.asarray(self.transform(img))
            img_lab = rgb2lab(img)
            grey_img = img_lab[:,:,0]
            img_ab = img_lab[:,:,1:3] / 128
            img_ab = torch.from_numpy(img_ab.transpose((2,0,1))).float()
            grey_img = torch.from_numpy(grey_img).unsqueeze(0).float()  #[20 x 30] -> [1,20,30]
            
        return grey_img, img_ab, target
            

# --------------------------------- Data Loader -----------------------

train_transform = transforms.Compose(
                            [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip()])
custom_dataset_train = LABImageGeneration('images/train',train_transform)
trainloader = torch.utils.data.DataLoader(custom_dataset_train, batch_size=BATCH_SIZE, shuffle=SHUFFLE)


valid_transform = transforms.Compose(
                            [transforms.Resize(256),
                            transforms.CenterCrop(224)])
custom_dataset_val = LABImageGeneration('images/val',valid_transform)
validloader = torch.utils.data.DataLoader(custom_dataset_val, batch_size=BATCH_SIZE, shuffle=True)



# --------------------------------- MODEL -----------------------


class ColorCNN(nn.Module):
    def __init__(self):
        super(ColorCNN, self).__init__()
        model = models.resnet18(pretrained=True)
        model.conv1.weight = nn.Parameter(model.conv1.weight.sum(axis=1).unsqueeze(1))
        layers = list(model.children())[0:6]
        self.ourmodel = nn.Sequential(*layers)
        
        
        
        self.upsample =nn.Sequential(     
                              nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.ReLU(),
                              nn.Upsample(scale_factor=2),
                              nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.ReLU(),
                              nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.ReLU(),
                              nn.Upsample(scale_factor=2),
                              nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Tanh(),
                              nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
                              nn.Upsample(scale_factor=2),
                            )
    
    def forward(self, input):
        middle = self.ourmodel(input)
        output = self.upsample(middle)
        return output
          

# --------------------------------- Post Processing step -----------------------

def to_rgb(grayscale_input, ab_input, ab_true, save_path=None, save_name=None):
  
  plt.clf()  
  color_image = torch.cat((grayscale_input, ab_input), 0).numpy()
  color_image = color_image.transpose((1, 2, 0))  
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 128
  color_image = lab2rgb(color_image.astype(np.float64))
  
  true_img = torch.cat((grayscale_input, ab_true), 0).numpy() 
  true_img = true_img.transpose((1, 2, 0))  
  true_img[:, :, 1:3] = true_img[:, :, 1:3] * 128 
  true_img = lab2rgb(true_img.astype(np.float64))


  grayscale_input = grayscale_input.squeeze().numpy()
  if save_path is not None and save_name is not None: 
    plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
    plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))
    fig, axes = plt.subplots(1,3,figsize=(10,10))
    axes[0].imshow(grayscale_input, cmap='gray')
    axes[1].imshow(color_image)
    axes[2].imshow(true_img)
    plt.show()


DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

model = ColorCNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

model = model.to(DEVICE)
criterion = criterion.to(DEVICE)


def train(trainloader, model, criterion, optimizer, epoch):
    print('Training.....')
    model.train()
    train_loss = 0
    
    for i ,(grey_img, img_ab, target) in enumerate(trainloader):
        grey_img, img_ab = grey_img.to(DEVICE), img_ab.to(DEVICE)
        
        output_ab = model(grey_img)
        loss = criterion(output_ab, img_ab)
        
        optimizer.zero_grad() # To clear gradients computed in previous backward pass
        loss.backward()
        optimizer.step()
        train_loss += loss/len(trainloader)
        if i%16==0:
            print(f"EPOCH: {epoch},                 train_loss: {loss}")
        
    print(f"EPOCH: {epoch},                 Total_train_Loss: {train_loss}")
    return train_loss

        


def valid(validloader, model, optimizer, save_images, epoch):
    print('Validation.....')
    model.eval()
    valid_loss = 0
    already_saved_images = False
    
    for i ,(grey_img, input_ab, target) in enumerate(validloader):
        print(i)
        grey_img, input_ab = grey_img.to(DEVICE), input_ab.to(DEVICE)
        
        output_ab = model(grey_img)
        loss = criterion(output_ab, input_ab)
        valid_loss += loss/len(validloader)
        # Save images to file
    if save_images and not already_saved_images:
        already_saved_images = True
        for j in range(min(len(output_ab), 5)): # save at most 5 images
            save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
            save_name = 'img-{}-epoch-{}.jpg'.format(i * validloader.batch_size + j, epoch)
            to_rgb(grey_img[j].cpu(), ab_input=output_ab[j].detach().cpu(), ab_true=input_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)
        
        if i%16==0:
            print(i)
        
    print(f"EPOCH: {epoch},                  valid_loss: {valid_loss}")
    return valid_loss
    


os.makedirs('outputs/color', exist_ok=True)
os.makedirs('outputs/gray', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
save_images = True


# --------------------------------- Training and Validation -----------------------

train_loss = []
valid_loss = []
best_loss = 1e10

for epoch in tqdm(range(EPOCHS)):
    loss_t = train(trainloader, model, criterion, optimizer, epoch)
    train_loss.append(loss_t)
    with torch.no_grad():
        loss_v = valid(validloader, model, criterion, save_images, epoch)
        if loss_v<best_loss:
            best_loss = loss_v
            
            torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch+1,loss_v))
            
        valid_loss.append(loss_v)
print(time.time()/3600)


train_loss_new = []
valid_loss_new = []
for i in range(len(train_loss)):
    train_loss_new.append(train_loss[i].to("cpu").detach().numpy().item())
    valid_loss_new.append(valid_loss[i].to("cpu").detach().numpy().item())    




loss = {'Train_loss':np.asarray(train_loss_new), 'Valid_loss':np.asarray(valid_loss_new)}
loss_df = pd.DataFrame(loss)

print(loss_df)


loss_df.to_csv('MSE_Loss.csv', index=False)


loss_df.plot(figsize=(15,10))
plt.show()


