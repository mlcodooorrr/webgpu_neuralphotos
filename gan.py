import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: need to test clamping, closed loop vs open loop issue with drift in pixels with generated frames
# TODO: need to make ModelConfig to test different sizes of models, reduce duplicate code

class ConvBlock(nn.Module):
    """Basic conv block: Conv -> LeakyRELU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DownBlock0(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_past = ConvBlock(12,16,3,2,1)        #   process past_3: (B, 12, 256, 192) -> (B, 16, 128, 96)
        self.c_ctrl = ConvBlock(16,16,1,1,0)        #   process controls: (B, 16, 1, 1) -> (B, 16, 128, 96)
        self.ctrl_mask = nn.Parameter(torch.ones(1,1,128,96))  # mask will broadcast automatically

    def forward(self, controls, past_3):
        past_feat = self.c_past(past_3) # (B, 16, 128, 96)
        ctrl_feat = self.c_ctrl(controls) # (B, 16, 1, 1)
        ctrl_feat = ctrl_feat * self.ctrl_mask  # (B, 16, 128, 96)
        output = torch.cat([past_feat, ctrl_feat], dim=1) # (B, 32, 128, 96)
        return output

class DownBlock1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_skip = nn.Sequential(                # process skip: (B, 32, 128, 96) -> (B, 64, 32, 24)
            ConvBlock(32, 32, 3, 2, 1),
            ConvBlock(32, 64, 3, 2, 1)
        )
        self.c_past = ConvBlock(24, 64, 3, 2, 1)    # process past_2: (B, 24, 64, 48) -> (B, 64, 32, 24)
        self.c_ctrl = ConvBlock(16,64,1,1,0)        # process controls: (B, 16, 1, 1) -> (B, 64, 1, 1)
        self.ctrl_mask = nn.Parameter(torch.ones(1,1,32,24))        # mask at output size (32, 24) for broadcasting
        self.c_out = ConvBlock(192, 64, 1, 1, 0) # Project concatenated: 192 → 64 channels for output

    def forward(self, controls, past_2, skip):
        skip_feat = self.c_skip(skip)               # (B, 64, 32, 24)
        past_feat = self.c_past(past_2)             # (B, 64, 32, 24)
        ctrl_feat = self.c_ctrl(controls)           # (B, 64, 1, 1)
        ctrl_feat = ctrl_feat * self.ctrl_mask      # Pytorch handles broadcasting automatically to (B, 64, 32, 24)
        combined = torch.cat([skip_feat, past_feat, ctrl_feat], dim=1)     # concatenate all 3: (B, 192, 32, 24)
        out = self.c_out(combined)      # (B, 64, 32, 24)
        return out

class DownBlock2(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_skip = nn.Sequential(
            ConvBlock(64,64,3,2,1),             # process skip: (B, 64, 32, 24) -> (B, 128, 8, 6)
            ConvBlock(64,128,3,2,1)             # each stride=2 halves the width/height, in out channels determine how mich bigger
        )
        self.c_past = ConvBlock(48,128,3,2,1)        # process past_1: (B, 48, 16, 12) -> (B, 128, 8, 6)
        self.c_ctrl = ConvBlock(16,128,1,1,0)       # process control: (B, 16, 1, 1) -> (B, 128, 1, 1)
        self.ctrl_mask = nn.Parameter(torch.ones(1,1,8,6))      # mask at output size (8, 6) for broadcasting# mask at output size (8, 6)
        self.c_out = ConvBlock(384,128,1,1,0)       # Project concatenated: 384 (128+128+128) → 128 channels

    def forward(self, controls, past_1, skip):
        skip_feat = self.c_skip(skip)       # (B, 128, 8, 6)
        past_feat = self.c_past(past_1)     # (B, 128, 8, 6)
        ctrl_feat = self.c_ctrl(controls)   # (B, 128, 1, 1)
        ctrl_feat = ctrl_feat * self.ctrl_mask      # broadcasts to (B, 128, 8, 6)
        combined = torch.cat([skip_feat, past_feat, ctrl_feat], dim=1)      # (B, 384, 8, 6)
        out = self.c_out(combined)      # (B, 128, 8, 6)
        return out

class DownBlock3(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_skip = ConvBlock(128,256,3,2,1)     # process skip: (B, 128, 8, 6) -> (B, 256, 4, 3)
        self.c_past = ConvBlock(96,256,3,1,1)     # process past_0: (B, 96, 4, 3) -> (B, 256, 4, 3), stride=1 here, no halving
        self.c_ctrl = ConvBlock(16, 256, 1, 1, 0)     # process control:: (B, 16, 1, 1) -> (B, 256, 1, 1)
        self.ctrl_mask = nn.Parameter(torch.ones(1,1,4,3))     # mask at output size (4, 3) for broadcasting
        self.c_out = ConvBlock(768,256,1,1,0)     # Project concatenated: 768 (256+256+256) → 256 channels

    def forward(self, controls, past_0, skip):
        skip_feat = self.c_skip(skip)               # (B, 256, 4, 3)
        past_feat = self.c_past(past_0)         # (B, 256, 4, 3)
        ctrl_feat = self.c_ctrl(controls)       # (B, 256, 1, 1)
        ctrl_feat = ctrl_feat * self.ctrl_mask      # Pytorch handles broadcasting automatically to (B, 256, 4, 3)
        combined = torch.cat([skip_feat, past_feat, ctrl_feat], dim=1)     # concatenate all 3: (B, 768, 4, 3)
        out = self.c_out(combined)      # (B, 256, 4, 3)
        return out

class Bottleneck(nn.Module):
    """
    Bottleneck processing at lowest resolution (4×3)
    """
    def __init__(self):
        super().__init__()
        self.c_cell = nn.Sequential(
            ConvBlock(257, 256, 1, 1, 0),           # process enc_3 + noise_0 (B, 257, 4, 3) -> (B, 256, 4, 3)
            ConvBlock(256, 256, 3, 1, 1),           # (B, 256, 4, 3) -> (B, 256, 4, 3)
            ConvBlock(256, 256, 3, 1, 1)            # (B, 256, 4, 3) -> (B, 256, 4, 3)
        )
        self.c_cell_2 = nn.Sequential(
            ConvBlock(513, 256, 1, 1, 0),           # process enc_3 + noise_0 + c_cell (B, 513, 4, 3) -> (B, 256, 4, 3)
            ConvBlock(256, 256, 3, 1, 1),           # (B, 256, 4, 3) -> (B, 256, 4, 3)
            ConvBlock(256, 256, 3, 1, 1),           # (B, 256, 4, 3) -> (B, 256, 4, 3)
        )
        self.c_out = ConvBlock(256, 256, 3, 1, 1)   # (B, 256, 4, 3) -> (B, 256, 4, 3)

    def forward(self, enc_3, noise_0):
        combined  = torch.cat([enc_3, noise_0], dim=1)                  # concat enc_3 + noise_0 (B, 256, 4, 3)
        cell_1_out = self.c_cell(combined)                              # (B, 256, 4, 3)
        combined_2 = torch.cat([cell_1_out, enc_3, noise_0], dim=1)     # concat cell_1_out + enc_3 + noise_0 (B, 514, 4, 3)     
        cell_2_out = self.c_cell_2(combined_2)                          # (B, 256, 4, 3)
        output = self.c_out(cell_2_out)                                 # (B, 256, 4, 3)
        return output

class UpBlock1(nn.Module):
    """
    Upsample from 4x3 -> 8x6 -> 16x12
    Input: x (256, 4, 3), skip (128, 8, 6), noise_1 (1, 16, 12)
    Output: (128, 16, 12)
    """
    def __init__(self):
        super().__init__()
        # c_x: Conv to 512 channels
        self.c_x = ConvBlock(256, 512, 1, 1, 0)
        # c_cell: process x_up, skip , noise_down
        self.c_cell = nn.Sequential(
            ConvBlock(260, 128, 1, 1, 0),
            ConvBlock(128, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1)
        )

        self.c_cell_2 = nn.Sequential(
            ConvBlock(388, 128, 1, 1, 0),
            ConvBlock(128, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1)
        )
        self.c_out = ConvBlock(128, 512, 1, 1, 0)

    def forward(self, x_up, skip, noise_up):
        """
        x: (256, 4, 3) - from bottleneck
        skip: (128, 8, 6) - from enc_2
        noise: (B, 1, 16, 12) - from noise_1
        """
        x_up = self.c_x(x_up)
        x_up = F.pixel_shuffle(x_up, 2)
        noise_down = F.pixel_unshuffle(noise_up, 2)
        combined = torch.cat([x_up, skip, noise_down], dim=1)
        cell_out = self.c_cell(combined)
        combined_2 = torch.cat([cell_out, x_up, skip, noise_down], dim=1)
        cell_2_out = self.c_cell_2(combined_2)
        output = self.c_out(cell_2_out)
        output = F.pixel_shuffle(output, 2)
        return output

class UpBlock2(nn.Module):
    """
    Upsample from 16,12 → 32,24 → 64,48
    Input: x (128, 16, 12), skip (64, 32, 24), noise_2 (1, 64, 48)
    Output: (64, 64, 48)
    """
    def __init__(self):
        super().__init__()
        self.c_x = ConvBlock(128, 256, 1, 1, 0)
        self.c_cell = nn.Sequential(
            ConvBlock(132, 64, 1, 1, 0),
            ConvBlock(64, 64, 3, 1, 1),
            ConvBlock(64, 64, 3, 1, 1)
        )        
        self.c_cell_2 = nn.Sequential(
            ConvBlock(196, 64, 1, 1, 0),
            ConvBlock(64, 64, 3, 1, 1),
            ConvBlock(64, 64, 3, 1, 1)
        )
        self.c_out = ConvBlock(64, 256, 1, 1, 0)

    def forward(self, x, skip, noise):
        """
        x: (B, 128, 16, 12) - from UpBlock1
        skip: (B, 64, 32, 24) - from enc_1
        noise: (B, 1, 64, 48) - noise_2 at final res
        """
        # (128, 16, 12) → (256, 16, 12) → (64, 32, 24)
        x_up = self.c_x(x)
        x_up = F.pixel_shuffle(x_up, 2)
        
        # Downsample noise: (1, 64, 48) → (4, 32, 24)
        noise_down = F.pixel_unshuffle(noise, 2)
        
        # [x_up, skip, noise_down] = 132 channels
        combined = torch.cat([x_up, skip, noise_down], dim=1)
        cell_out = self.c_cell(combined)
        
        # [cell_out, x_up, skip, noise_down] = 196 channels
        combined_2 = torch.cat([cell_out, x_up, skip, noise_down], dim=1)
        cell_2_out = self.c_cell_2(combined_2)
        
        # (64, 32, 24) → (256, 32, 24) → (64, 64, 48)
        output = self.c_out(cell_2_out)
        output = F.pixel_shuffle(output, 2)
        
        return output

class UpBlock3(nn.Module):
    """
    Upsample from 64×48 → 128×96 → 256×192 (final RGB output)
    Input: x (64, 64, 48), skip (32, 128, 96), noise_3 (1, 256, 192)
    Output: (3, 256, 192) RGB image
    """
    def __init__(self):
        super().__init__()
        # c_x: Conv to 128 channels (for 2× pixel shuffle)
        self.c_x = ConvBlock(64, 128, 1, 1, 0)
        
        # 32 + 32 + 4 = 68
        self.c_cell = nn.Sequential(
            ConvBlock(68, 32, 1, 1, 0),
            ConvBlock(32, 32, 3, 1, 1)
        )
        
        # output rgb
        self.c_rgb = nn.Conv2d(32, 12, 3, 1, 1)

    def forward(self, x, skip, noise):
        """
        x: (B, 64, 64, 48) - from UpBlock2
        skip: (B, 32, 128, 96) - from enc_0
        noise: (B, 1, 256, 192) - noise_3 at final res
        """
        
        (64, 64, 48) → (128, 64, 48) → (32, 128, 96)
        x_up = self.c_x(x)
        x_up = F.pixel_shuffle(x_up, 2)

        # (1, 256, 192) → (4, 128, 96)
        noise_down = F.pixel_unshuffle(noise, 2)
        
        # Concat: [x_up, skip, noise_down] = 68 channels
        combined = torch.cat([x_up, skip, noise_down], dim=1)
        cell_out = self.c_cell(combined)
        
        # (32, 128, 96) → (12, 128, 96) → (3, 256, 192)
        rgb = self.c_rgb(cell_out)
        rgb = F.pixel_shuffle(rgb, 2)

        rgb = torch.tanh(rgb)

        return rgb

class GAN(nn.Module):
    """
    4-scale UNet generator for neural world model
    Takes controls, multi-scale memory buffers, and noise as input
    Outputs a predicted frame
    """
    def __init__(self):
        super(GAN, self).__init__()
        
        # Encoder Blocks
        self.encoder_scale3 = DownBlock0()
        self.encoder_scale2 = DownBlock1()
        self.encoder_scale1 = DownBlock2()
        self.encoder_scale0 = DownBlock3()

        # Bottleneck
        self.bottleneck = Bottleneck()

        # Decoder Blocks
        self.decoder_scale0 = UpBlock1()
        self.decoder_scale1 = UpBlock2()
        self.decoder_scale2 = UpBlock3()
        
        self.output_conv = None

    def forward(self, controls, past_0, past_1, past_2, past_3, noise_0, noise_1, noise_2, noise_3):
        """
        Args:
            controls: (B, 16) - control vector
            past_0: (B, 32, 3, 4, 3) - oldest frames
            past_1: (B, 16, 3, 16, 12)
            past_2: (B, 8, 3, 64, 48)
            past_3: (B, 4, 3, 256, 192) - newest frames
            noise_0-3: Noise at various resolutions
        Returns:
            (B, 3, 256, 192) - predicted RGB frame
        """
        B = controls.shape[0]

        # reshape past frames -> (B, T, C, H, W) -> (B, T*C, H, W)
        past_0 = past_0.reshape(B, -1, 4, 3)        # (B, 96, 4, 3)
        past_1 = past_1.reshape(B, -1, 16, 12)      # (B, 48, 16, 12)
        past_2 = past_2.reshape(B, -1, 64, 48)      # (B, 24, 64, 48)
        past_3 = past_3.reshape(B, -1, 256, 192)    # (B, 12, 256, 192)

        # reshape controls for broadcasting (B, 16) -> (B, 16, 1, 1)
        controls_spatial = controls.reshape(B, 16, 1, 1)

        # DownBlocks, 1-3 have skips
        enc_0 = self.encoder_scale3(controls_spatial, past_3)  
        enc_1 = self.encoder_scale2(controls_spatial, past_2, enc_0)  
        enc_2 = self.encoder_scale1(controls_spatial, past_1, enc_1) 
        enc_3 = self.encoder_scale0(controls_spatial, past_0, enc_2)

        bottleneck_out = self.bottleneck(enc_3, noise_0)  # (B, 256, 4, 3)

        # UpBlocks
        dec_1 = self.decoder_scale0(bottleneck_out, enc_2, noise_1)  # (B, 128, 16, 12)
        dec_2 = self.decoder_scale1(dec_1, enc_1, noise_2)  # (B, 64, 64, 48)
        output = self.decoder_scale2(dec_2, enc_0, noise_3)  # (B, 3, 256, 192)

        return output
