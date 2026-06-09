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
        self.conv2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)

        # Adjust channels in skip connection if necessary
        self.adjust_channels = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1, stride=2, output_padding=output_padding) if in_channels != out_channels else None

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

class FrequencyDomainBranch(nn.Module):
    def __init__(self, in_channels=6, conv_channels=16, res_channels=(128, 256)):
        super(FrequencyDomainBranch, self).__init__()
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

class DualBranchAutoencoder(nn.Module):
    def __init__(self, model_config):
        super(DualBranchAutoencoder, self).__init__()
        # Determine whether to use the frequency branch (default is True)
        self.use_freq_branch = model_config.get('use_freq_branch', True)

        # Extract hyperparameters from model_config
        time_branch_params = model_config.get('time_branch', {})
        freq_branch_params = model_config.get('freq_branch', {})
        decoder_channels = model_config.get('decoder_channels', [128, 64, 32, 3])

        self.time_branch = TimeDomainBranch(**time_branch_params)

        if self.use_freq_branch:
            # Define magnitude and phase branches
            self.magnitude_branch = FrequencyDomainBranch(in_channels=3, **freq_branch_params)  # 3 channels for X,Y,Z magnitude
            self.phase_branch = FrequencyDomainBranch(in_channels=3, **freq_branch_params)     # 3 channels for X,Y,Z phase

            # Calculate total channels: time + magnitude + phase
            total_channels_before_fusion = (self.time_branch.output_channels + 
                             self.magnitude_branch.output_channels + 
                             self.phase_branch.output_channels)
        else:
            # Only time domain channels
            total_channels_before_fusion = self.time_branch.output_channels

        # Add fusion layer to learn optimal combination (or just process time features if freq is disabled)
        fusion_channels = total_channels_before_fusion // 2  # Reduce dimensionality
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(total_channels_before_fusion, fusion_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Adjust decoder_channels to match the computed total_channels
        decoder_channels = decoder_channels.copy()  # Ensure we don't modify the original list
        decoder_channels[0] = fusion_channels  # Set the first element to fusion_channels

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
        # Check for NaN in input
        if torch.isnan(x).any():
            # Replace NaN with zeros
            x = torch.nan_to_num(x, nan=0.0)
        
        # Time domain processing
        time_features = self.time_branch(x)  # Input shape: [batch_size, 3, signal_length]

        if self.use_freq_branch:
            # Frequency domain processing
            x_fft = torch.fft.rfft(x, dim=-1)  # Perform FFT along the last dimension

            # Process magnitude and phase separately
            magnitude = torch.abs(x_fft)
            phase = torch.angle(x_fft)     

            # Process each component
            mag_features = self.magnitude_branch(magnitude)
            phase_features = self.phase_branch(phase)

            # Convert frequency features to global context via Adaptive Avg Pooling
            mag_global = F.adaptive_avg_pool1d(mag_features, 1)      # [B, C, 1]
            phase_global = F.adaptive_avg_pool1d(phase_features, 1)  # [B, C, 1]
            
            # Expand the global frequency context across the entire temporal sequence
            mag_expanded = mag_global.expand(-1, -1, time_features.size(2))
            phase_expanded = phase_global.expand(-1, -1, time_features.size(2))

            # Safely concatenate: the temporal structure is perfectly preserved
            combined_features = torch.cat((time_features, mag_expanded, phase_expanded), dim=1)
        else:
            combined_features = time_features
        
        # Apply fusion layer
        fused_features = self.fusion_layer(combined_features)

        # Decode to reconstruct the clean signal in the time domain
        reconstructed_signal = self.decoder(fused_features)
        
        # Ensure the reconstructed signal matches the input size
        if reconstructed_signal.shape[-1] != x.shape[-1]:
            reconstructed_signal = F.interpolate(reconstructed_signal, size=x.shape[-1], mode='linear', align_corners=False)

        return reconstructed_signal

class TimeOnlyAutoencoder(nn.Module):
    def __init__(self, model_config):
        super(TimeOnlyAutoencoder, self).__init__()
        # Extract hyperparameters from model_config
        time_branch_params = model_config.get('time_branch', {})
        decoder_channels = model_config.get('decoder_channels', [128, 64, 32, 3])

        self.time_branch = TimeDomainBranch(**time_branch_params)

        total_channels = self.time_branch.output_channels
        
        # Adjust decoder_channels to match the computed total_channels
        decoder_channels = decoder_channels.copy()
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
        # Check for NaN in input
        if torch.isnan(x).any():
            # Replace NaN with zeros
            x = torch.nan_to_num(x, nan=0.0)
        
        # Time domain processing
        time_features = self.time_branch(x)  # Input shape: [batch_size, 3, signal_length]

        # Decode to reconstruct the clean signal in the time domain
        reconstructed_signal = self.decoder(time_features)
        
        # Ensure the reconstructed signal matches the input size
        if reconstructed_signal.shape[-1] != x.shape[-1]:
            reconstructed_signal = F.interpolate(reconstructed_signal, size=x.shape[-1], mode='linear', align_corners=False)

        return reconstructed_signal
class FreqOnlyAutoencoder(nn.Module):
    def __init__(self, model_config):
        super(FreqOnlyAutoencoder, self).__init__()
        # Extract hyperparameters from model_config
        freq_branch_params = model_config.get('freq_branch', {})
        decoder_channels = model_config.get('decoder_channels', [128, 64, 32, 3])

        # Define magnitude and phase branches
        self.magnitude_branch = FrequencyDomainBranch(in_channels=3, **freq_branch_params)  
        self.phase_branch = FrequencyDomainBranch(in_channels=3, **freq_branch_params)     

        total_channels_before_fusion = (self.magnitude_branch.output_channels + 
                         self.phase_branch.output_channels)
        
        # Add fusion layer to learn optimal combination
        fusion_channels = total_channels_before_fusion // 2  # Reduce dimensionality
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(total_channels_before_fusion, fusion_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Adjust decoder_channels to match the computed total_channels
        decoder_channels = decoder_channels.copy()
        decoder_channels[0] = fusion_channels

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
        # Check for NaN in input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        # Frequency domain processing
        x_fft = torch.fft.rfft(x, dim=-1)  

        # Process magnitude and phase separately
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)  

        # Process each component
        mag_features = self.magnitude_branch(magnitude)
        phase_features = self.phase_branch(phase)

        # Align all features to the same size
        min_size = min(mag_features.size(2), phase_features.size(2))
        mag_features = F.interpolate(mag_features, size=min_size, mode='linear', align_corners=False)
        phase_features = F.interpolate(phase_features, size=min_size, mode='linear', align_corners=False)

        # Combine features from all branches
        combined_features = torch.cat((mag_features, phase_features), dim=1)
        
        # Apply fusion layer
        fused_features = self.fusion_layer(combined_features)

        # Decode
        reconstructed_signal = self.decoder(fused_features)
        
        # Ensure the reconstructed signal matches the input size
        if reconstructed_signal.shape[-1] != x.shape[-1]:
            reconstructed_signal = F.interpolate(reconstructed_signal, size=x.shape[-1], mode='linear', align_corners=False)

        return reconstructed_signal
