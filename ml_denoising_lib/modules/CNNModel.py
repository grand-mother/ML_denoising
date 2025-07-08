import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual Block for 1D convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)

        # Adjust the input channels if they do not match
        self.adjust_channels = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Adjust the input channels if needed
        if self.adjust_channels is not None:
            identity = self.adjust_channels(identity)

        out += identity
        out = self.relu(out)
        return out

class DecoderResidualBlock(nn.Module):
    """
    Residual Block for Decoder with Transposed 1D Convolution layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, output_padding):
        super(DecoderResidualBlock, self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=output_padding)
        self.activation = nn.ReLU() 
        self.conv2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)

        # Adjust channels in skip connection if necessary
        self.adjust_channels = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1, stride=2, output_padding=output_padding) if in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)

        # Apply the skip connection
        if self.adjust_channels is not None:
            identity = self.adjust_channels(identity)

        # Ensure output and identity have the same shape before adding
        if out.size() != identity.size():
            diff = out.size(2) - identity.size(2)
            identity = F.pad(identity, (0, diff))

        out += identity
        #out = self.activation(out)
        return out

class TimeDomainBranch(nn.Module):
    def __init__(self, in_channels=3, conv_channels=64*1, res_channels=(32*1, 16*1)):
        super(TimeDomainBranch, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=conv_channels, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        # Residual blocks
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

class FrequencyDomainBranch(nn.Module):
    def __init__(self, in_channels=6, conv_channels=64*1, res_channels=(32*1, 16*1)):
        super(FrequencyDomainBranch, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=conv_channels, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        # Residual blocks
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

class DualBranchAutoencoder(nn.Module):
    def __init__(self, xy_mode=False, time_only=True, decoder_channels=[32, 64, 128, 3]):
        super(DualBranchAutoencoder, self).__init__()
        if xy_mode:
            nb_chan_in = 2
        else:
            nb_chan_in = 3
        self.time_only = time_only

        self.time_branch = TimeDomainBranch(in_channels=nb_chan_in)
        if not (self.time_only):
            self.freq_branch = FrequencyDomainBranch(in_channels=2*nb_chan_in)

        decoder_channels[-1] = nb_chan_in
        if self.time_only:
            total_channels = self.time_branch.output_channels 
        else:
            total_channels = self.time_branch.output_channels + self.freq_branch.output_channels
         # Adjust decoder_channels to match the computed total_channels
        decoder_channels = decoder_channels.copy()  # Ensure we don't modify the original list
        decoder_channels[0] = total_channels  # Set the first element to total_channels
        
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels=decoder_channels[0], out_channels=decoder_channels[1], kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(in_channels=decoder_channels[1], out_channels=decoder_channels[2], kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(in_channels=decoder_channels[2], out_channels=decoder_channels[3], kernel_size=5, stride=2, padding=2, output_padding=1)
        # )
        # Decoder with residual blocks
        self.decoder = nn.Sequential(
            DecoderResidualBlock(in_channels=decoder_channels[0], out_channels=decoder_channels[1], kernel_size=3, padding=1, output_padding=1),
            nn.ReLU(),
            DecoderResidualBlock(in_channels=decoder_channels[1], out_channels=decoder_channels[2], kernel_size=3, padding=1, output_padding=1),
            nn.ReLU(),
            DecoderResidualBlock(in_channels=decoder_channels[2], out_channels=decoder_channels[3], kernel_size=5, padding=2, output_padding=1)
        )

    def forward(self, x):
        # Time domain processing
        time_features = self.time_branch(x)  # Input shape: [batch_size, 3, signal_length]
        if not (self.time_only):
            # Frequency domain processing
            x_fft = torch.fft.rfft(x, dim=-1)  # Perform FFT along the last dimension
            
            # Split FFT into real and imaginary parts
            x_fft_real = x_fft.real
            x_fft_imag = x_fft.imag
            
            # Concatenate real and imaginary parts along the channel dimension
            x_fft_combined = torch.cat((x_fft_real, x_fft_imag), dim=1)  # Shape: [batch_size, 6, signal_length//2 + 1]

            # Process frequency features
            freq_features = self.freq_branch(x_fft_combined)

            # Align the output sizes by interpolation if needed
            if time_features.size(2) != freq_features.size(2):
                min_size = min(time_features.size(2), freq_features.size(2))
                time_features = F.interpolate(time_features, size=min_size, mode='linear', align_corners=False)
                freq_features = F.interpolate(freq_features, size=min_size, mode='linear', align_corners=False)

        # Combine features from both branches
            combined_features = torch.cat((time_features, freq_features), dim=1)  # Concatenate along the channel dimension
        else:
            combined_features = time_features
            
        # Decode to reconstruct the clean signal in the time domain
        reconstructed_signal = self.decoder(combined_features)
        
        # Ensure the reconstructed signal matches the input size
        if reconstructed_signal.shape[-1] != x.shape[-1]:
            reconstructed_signal = F.interpolate(reconstructed_signal, size=x.shape[-1], mode='linear', align_corners=False)

        return reconstructed_signal