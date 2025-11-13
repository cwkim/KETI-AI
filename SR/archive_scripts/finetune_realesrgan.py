#!/usr/bin/env python3
"""
Real-ESRGAN Fine-tuning Script
Fine-tune Real-ESRGAN on microdisk images
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader

sys.path.append('AI_models/Real-ESRGAN')
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


class MicrodiskDataset(Dataset):
    """Dataset for microdisk images"""
    def __init__(self, hr_dir, lr_dir, patch_size=128):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.patch_size = patch_size

        self.hr_images = sorted(list(self.hr_dir.glob('*.png')))
        self.lr_images = sorted(list(self.lr_dir.glob('*.png')))

        assert len(self.hr_images) == len(self.lr_images), "HR and LR image counts must match"

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        # Load images
        hr_img = cv2.imread(str(self.hr_images[idx]))
        lr_img = cv2.imread(str(self.lr_images[idx]))

        # Convert BGR to RGB
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)

        # Resize to fixed size (instead of random crop)
        h, w = hr_img.shape[:2]

        if h >= self.patch_size and w >= self.patch_size:
            # Random crop if image is large enough
            h_start = random.randint(0, h - self.patch_size)
            w_start = random.randint(0, w - self.patch_size)
            hr_patch = hr_img[h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]

            lr_h_start = h_start // 4
            lr_w_start = w_start // 4
            lr_patch_size = self.patch_size // 4
            lr_patch = lr_img[lr_h_start:lr_h_start+lr_patch_size, lr_w_start:lr_w_start+lr_patch_size]
        else:
            # Resize to fixed patch size
            hr_patch = cv2.resize(hr_img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
            lr_patch = cv2.resize(lr_img, (self.patch_size//4, self.patch_size//4), interpolation=cv2.INTER_CUBIC)

        # Random augmentation
        if random.random() > 0.5:
            hr_patch = np.fliplr(hr_patch).copy()
            lr_patch = np.fliplr(lr_patch).copy()

        if random.random() > 0.5:
            hr_patch = np.flipud(hr_patch).copy()
            lr_patch = np.flipud(lr_patch).copy()

        # Convert to tensor and normalize to [0, 1]
        hr_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1).float() / 255.0
        lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).float() / 255.0

        return lr_tensor, hr_tensor


def train_realesrgan(model, train_loader, num_epochs, device, save_path, start_weights):
    """Train Real-ESRGAN model"""

    # Loss functions
    criterion_pixel = nn.L1Loss().to(device)

    # Optimizer with lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, num_epochs//3), gamma=0.5)

    model.train()

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Total iterations: {len(train_loader) * num_epochs}")

    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, (lr_imgs, hr_imgs) in enumerate(progress_bar):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # Forward pass
            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)

            # Calculate loss
            loss = criterion_pixel(sr_imgs, hr_imgs)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.6f}, LR: {current_lr:.6f}')

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f'  -> Best model saved with loss: {best_loss:.6f}')

        scheduler.step()

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = save_path.parent / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'  -> Checkpoint saved: {checkpoint_path}')

    print(f"\nTraining completed! Best loss: {best_loss:.6f}")
    return model


def main():
    base_dir = Path('/home/keti/cwkim/KETI-AI/SR')
    results_dir = base_dir / 'results'
    train_hr_dir = results_dir / 'train' / 'hr'
    train_lr_dir = results_dir / 'train' / 'lr'
    weights_dir = base_dir / 'AI_models' / 'Real-ESRGAN' / 'weights'

    pretrained_weights = weights_dir / 'RealESRGAN_x4plus.pth'
    finetuned_weights = weights_dir / 'RealESRGAN_x4plus_finetuned.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create model
    print("\nCreating RRDBNet model...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                   num_block=23, num_grow_ch=32, scale=4)

    # Load pre-trained weights
    print(f"Loading pre-trained weights from {pretrained_weights}...")
    pretrained_dict = torch.load(pretrained_weights, map_location='cpu')

    # Handle different checkpoint formats
    if 'params_ema' in pretrained_dict:
        pretrained_dict = pretrained_dict['params_ema']
    elif 'params' in pretrained_dict:
        pretrained_dict = pretrained_dict['params']

    model.load_state_dict(pretrained_dict, strict=True)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Create dataset and dataloader
    print("\nPreparing dataset...")
    dataset = MicrodiskDataset(train_hr_dir, train_lr_dir, patch_size=128)
    print(f"Dataset size: {len(dataset)}")

    batch_size = 4 if device.type == 'cuda' else 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)

    # Train model with more epochs for better convergence
    num_epochs = 50  # Increased for better fine-tuning
    model = train_realesrgan(model, dataloader, num_epochs, device,
                            finetuned_weights, pretrained_weights)

    print(f"\nFine-tuned model saved to: {finetuned_weights}")

    # Verify the saved model
    if finetuned_weights.exists():
        print(f"Fine-tuned model file size: {finetuned_weights.stat().st_size / 1e6:.2f} MB")
        print("\nFine-tuning completed successfully!")
    else:
        print("\nError: Fine-tuned model not saved!")


if __name__ == '__main__':
    main()
