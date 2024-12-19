import torch
import torchaudio
import torch.nn.functional as F

class WhisperFeatureExtractorTorch:
    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=-0.7739,  # Set padding value here
        device="cpu",
    ):
        self.device = device
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.padding_value = padding_value

        # Calculate expected number of samples and target frames for the chunk length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length

        # Initialize Mel filter bank with torchaudio.transforms.MelScale
        self.mel_filters = torchaudio.transforms.MelScale(
            n_mels=feature_size,
            sample_rate=sampling_rate,
            f_min=0.0,
            f_max=sampling_rate / 2,
            n_stft=n_fft // 2 + 1,
            norm="slaney",
            mel_scale="slaney"
        ).to(device)

    def extract_fbank_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract log-mel spectrogram features from a waveform and pad or truncate output to match target shape.
        
        Args:
            waveform (torch.Tensor): Input waveform tensor (1D or 2D for batch).
        
        Returns:
            torch.Tensor: Log-mel spectrogram with shape (1, feature_size, nb_max_frames).
        """
        # Ensure the waveform is 2D and pad if necessary
        """if waveform.dim() == 1:
            print("uh oh")
            waveform = waveform.unsqueeze(0)
        if waveform.size(1) < self.n_samples:
            # Pad waveform to expected length with the fixed padding value
            waveform = F.pad(waveform, (0, self.n_samples - waveform.size(1)), value=self.padding_value)
        else:
            # Truncate waveform if longer than expected length
            print("this is the issue")
            waveform = waveform[:, :self.n_samples]"""

        # Compute STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft).to(self.device),
            return_complex=True
        )

        # Compute magnitude spectrogram and ensure it's float32
        magnitudes = stft.abs().pow(2).float()

        # Apply mel filter bank
        mel_spec = self.mel_filters(magnitudes)

        # Convert to log scale
        log_mel_spec = torch.clamp(mel_spec, min=1e-10).log10()
        
        # Normalize to align with the expected range
        max_val = log_mel_spec.amax(dim=-1, keepdim=True)
        log_mel_spec = torch.maximum(log_mel_spec, max_val - 8.0)
        log_mel_spec = (log_mel_spec + 4.0) / 4.0

        # Pad or truncate along time dimension to match target shape (1, feature_size, nb_max_frames)
        if log_mel_spec.size(-1) < self.nb_max_frames:
            # Pad log_mel_spec to expected length with the fixed padding value
            log_mel_spec = F.pad(
                log_mel_spec, (0, self.nb_max_frames - log_mel_spec.size(-1)), value=self.padding_value
            )
        else:
            log_mel_spec = log_mel_spec[:, :, :self.nb_max_frames]

        # Add batch dimension and ensure the shape is (1, feature_size, nb_max_frames)
        log_mel_spec = log_mel_spec.unsqueeze(0)

        return log_mel_spec
