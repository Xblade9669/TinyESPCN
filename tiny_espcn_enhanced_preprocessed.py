
import os
import glob
import random
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


training_images_folder = "/kaggle/input/low-resolution-photographs/dataset"  # update if needed
all_images = glob.glob(os.path.join(training_images_folder, "*.*"))
all_images = [f for f in all_images if f.lower().endswith((".png",".jpg",".jpeg"))]
print(f"Found {len(all_images)} images in: {training_images_folder}")


class CropDataset(Dataset):
    def __init__(self, root, scale=2, crop_size=64, max_images=2000):
        self.scale = scale
        self.crop_size = crop_size
        self.files = glob.glob(os.path.join(root, "*.*"))
        self.files = [f for f in self.files if f.lower().endswith((".png",".jpg",".jpeg"))][:max_images]
        if len(self.files) == 0:
            raise ValueError("No valid images found in folder.")

        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20)
        ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.augmentations(img)
        w, h = img.size
        if w < self.crop_size or h < self.crop_size:
            img = img.resize((max(w,self.crop_size), max(h,self.crop_size)), Image.BICUBIC)

        x = random.randint(0, img.width - self.crop_size)
        y = random.randint(0, img.height - self.crop_size)
        hr = img.crop((x, y, x+self.crop_size, y+self.crop_size))
        lr = hr.resize((self.crop_size//self.scale, self.crop_size//self.scale), Image.BICUBIC)

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)
        return lr, hr


scale = 2
crop_size = 64
batch_size = 16
epochs = 50

dataset = CropDataset(training_images_folder, scale=scale, crop_size=crop_size, max_images=2000)
print(f"Dataset created with {len(dataset)} samples")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
print("DataLoader ready")


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y


class TinyESPCNEnhanced(nn.Module):
    def __init__(self, scale=2, use_attention=True):
        super().__init__()
        self.scale = scale
        self.use_attention = use_attention

        self.conv1 = nn.Conv2d(3,64,7,1,3)
        self.res_blocks = nn.Sequential(*[nn.Sequential(nn.Conv2d(64,64,3,1,1), nn.ReLU()) for _ in range(10)])
        if use_attention:
            self.attention = ChannelAttention(64)
        self.conv2 = nn.Conv2d(64, 3*(scale**2), 3,1,1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        lr_input = x
        x1 = F.relu(self.conv1(x))
        x2 = self.res_blocks(x1)
        if self.use_attention:
            x2 = self.attention(x2)
        x = self.pixel_shuffle(self.conv2(x2+x1))
        lr_up = F.interpolate(lr_input, scale_factor=self.scale, mode='bicubic', align_corners=False)
        return torch.clamp(x+lr_up,0,1)

class EnhancedLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters(): p.requires_grad=False
        self.vgg = vgg.to(device)
        self.device = device
        self.layers = [2,7,12]

        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        laplacian = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32)

        self.sobel_x = sobel_x.view(1,1,3,3).repeat(3,1,1,1).to(device)
        self.sobel_y = sobel_y.view(1,1,3,3).repeat(3,1,1,1).to(device)
        self.laplacian = laplacian.view(1,1,3,3).repeat(3,1,1,1).to(device)

    def forward(self,sr,hr):
        sr = torch.clamp(sr,0,1)
        hr = torch.clamp(hr,0,1)

        mean = torch.tensor([0.485,0.456,0.406],device=self.device).view(1,3,1,1)
        std = torch.tensor([0.229,0.224,0.225],device=self.device).view(1,3,1,1)
        sr_vgg = (sr-mean)/std
        hr_vgg = (hr-mean)/std

        loss=0
        sr_f, hr_f = sr_vgg, hr_vgg
        for i,layer in enumerate(self.vgg):
            sr_f = layer(sr_f)
            hr_f = layer(hr_f)
            if i in self.layers:
                loss += F.l1_loss(sr_f, hr_f)

        grad_x_sr = F.conv2d(sr, self.sobel_x, padding=1, groups=3)
        grad_y_sr = F.conv2d(sr, self.sobel_y, padding=1, groups=3)
        grad_x_hr = F.conv2d(hr, self.sobel_x, padding=1, groups=3)
        grad_y_hr = F.conv2d(hr, self.sobel_y, padding=1, groups=3)
        edge_loss = F.l1_loss(grad_x_sr,grad_x_hr)+F.l1_loss(grad_y_sr,grad_y_hr)

        lap_sr = F.conv2d(sr,self.laplacian,padding=1,groups=3)
        lap_hr = F.conv2d(hr,self.laplacian,padding=1,groups=3)
        edge_loss += F.l1_loss(lap_sr, lap_hr)

        loss += 0.2 * edge_loss

        sr_lab = rgb_to_lab(sr)
        hr_lab = rgb_to_lab(hr)
        loss += 0.1 * F.l1_loss(sr_lab, hr_lab)

        return loss

def rgb_to_lab(tensor):
    from skimage import color
    B,C,H,W = tensor.shape
    lab=[]
    for i in range(B):
        img = tensor[i].detach().permute(1,2,0).cpu().numpy()
        lab_img = color.rgb2lab(img)
        lab.append(torch.tensor(lab_img, device=tensor.device).permute(2,0,1))
    return torch.stack(lab)


def train_model(model, dataloader, epochs=50, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = EnhancedLoss(device=device)
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for lr_img, hr_img in pbar:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            optimizer.zero_grad()
            sr = model(lr_img)
            loss = criterion(sr, hr_img)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch+1}/{epochs} Loss:{loss.item():.6f}")
    return model


def upscale_images(model, files, out_folder="upscaled", device='cuda', sharpen=True):
    os.makedirs(out_folder, exist_ok=True)
    model.eval()
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    with torch.no_grad():
        for path in tqdm(files):
            img = Image.open(path).convert("RGB")
            lr = to_tensor(img).unsqueeze(0).to(device)
            sr = model(lr).clamp(0,1).cpu().squeeze(0)
            out_img = to_pil(sr)
            if sharpen:
                out_img = out_img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=100, threshold=1))
            out_img.save(os.path.join(out_folder, os.path.basename(path)))

def msaa_postprocess(input_folder="upscaled", output_folder="upscaled_aa", supersample=2):
    os.makedirs(output_folder, exist_ok=True)
    files = glob.glob(os.path.join(input_folder,"*.*"))
    files = [f for f in files if f.lower().endswith((".png",".jpg",".jpeg"))]
    for path in tqdm(files):
        img = Image.open(path).convert("RGB")
        w,h = img.size
        img = img.resize((w*supersample,h*supersample), Image.LANCZOS)
        img = img.filter(ImageFilter.GaussianBlur(0.5))
        img = img.resize((w,h), Image.LANCZOS)
        img.save(os.path.join(output_folder, os.path.basename(path)))


model = TinyESPCNEnhanced(scale=scale)
print("Model initialized and ready for training or inference")
