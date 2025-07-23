"""
Model for using the Resnet Encoder-Decoder Model for the traces reconstructions.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    """
    encoder: 2 1-d convolution layers in one block. There are 3 blocks in the encoder.

    decoder: 2 1-d convotranspose layers in one block.  There are 3 blocks in the decoder.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Adjust channels in skip connection if necessary
        self.adjust_channels = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Apply the skip connection
        if self.adjust_channels is not None:
            identity = self.adjust_channels(identity)

        out += identity
        out = self.relu(out)
        return out
    
class DecoderResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, output_padding):
        super(DecoderResidualBlock, self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=output_padding)
        self.conv2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)

        # Adjust channels in skip connection if necessary
        self.adjust_channels = nn.ConvTranspose1d(in_channels, out_channels, 1, stride=2, output_padding=output_padding) if in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        # Apply the skip connection
        if self.adjust_channels is not None:
            identity = self.adjust_channels(identity)

        # Ensure output and identity have the same shape before adding
        if out.size() != identity.size():
            diff = out.size(2) - identity.size(2)
            identity = F.pad(identity, (0, diff))

        out += identity
        return out
    
class TimeDomainBranch(nn.Module):
    def __init__(self, in_channels=3, conv_channels=32, res_channels=(128, 256)):
        super(TimeDomainBranch, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=conv_channels, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.res_block1 = ResidualBlock(in_channels=conv_channels, out_channels=res_channels[0], kernel_size=3, stride=1, padding=1)
        self.res_block2 = ResidualBlock(in_channels=res_channels[0], out_channels=res_channels[1], kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.output_channels = res_channels[1] 

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.res_block1(x)
        x = self.pool(x)
        x = self.res_block2(x)
        return x
    
class SingleBranchAutoencoder(nn.Module):
    def __init__(self, model_config):
        super(SingleBranchAutoencoder, self).__init__()
        # Extract hyperparameters from model_config
        time_branch_params = model_config.get('time_branch', {})
        decoder_channels = model_config.get('decoder_channels', [128, 64, 32, 3])

        self.time_branch = TimeDomainBranch(**time_branch_params)

        # Calculate the total number of channels after concatenation
        total_channels = self.time_branch.output_channels
        # Adjust decoder_channels to match the computed total_channels
        decoder_channels = decoder_channels.copy()  # Ensure we don't modify the original list
        decoder_channels[0] = total_channels


        self.decoder = nn.Sequential(
            DecoderResidualBlock(
                in_channels=decoder_channels[0],
                out_channels=decoder_channels[1],
                kernel_size=3,
                padding=1,
                output_padding=1
            ),
            DecoderResidualBlock(
                in_channels=decoder_channels[1],
                out_channels=decoder_channels[2],
                kernel_size=3,
                padding=1,
                output_padding=1
            ),
            DecoderResidualBlock(
                in_channels=decoder_channels[2],
                out_channels=decoder_channels[3],
                kernel_size=5,
                padding=2,
                output_padding=1
            )
        )

    def forward(self, x):
        original_size = x.shape[-1]
        x = self.time_branch(x)
        reconstructed_signal = self.decoder(x)
        # Ensure the reconstructed signal matches the input size
        if reconstructed_signal.shape[-1] != original_size:
            reconstructed_signal = F.interpolate(reconstructed_signal, size=original_size, mode='linear', align_corners=False)
        return reconstructed_signal
    
    
    
