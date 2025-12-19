import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 1))
        )
    
    def forward(self, x):
        return self.model(x)

def discriminator_loss(real_pred, fake_pred, loss_type='hinge'):
    if loss_type == 'hinge':
        loss_real = torch.mean(F.relu(1.0 - real_pred))
        loss_fake = torch.mean(F.relu(1.0 + fake_pred))
    else:
        loss_real = F.binary_cross_entropy_with_logits(
            real_pred, torch.ones_like(real_pred)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            fake_pred, torch.zeros_like(fake_pred)
        )
    return loss_real + loss_fake

def generator_adversarial_loss(fake_pred, loss_type='hinge'):
    if loss_type == 'hinge':
        return -torch.mean(fake_pred)
    else:
        return F.binary_cross_entropy_with_logits(
            fake_pred, torch.ones_like(fake_pred)
        )

def reconstruction_loss(pred, target, loss_type='l1'):
    if loss_type == 'l1':
        return F.l1_loss(pred, target)
    else:
        return F.mse_loss(pred, target)

def gradient_penalty(discriminator, real_data, fake_data, device='cuda'):
    """
    Compute gradient penalty for WGAN-GP style training
    Helps stabilize discriminator training
    """
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_data)
    
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_(True)
    
    d_interpolates = discriminator(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty