# src/eeg_bci/augment.py
import math, random
from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F

Tensor = torch.Tensor

def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

# ---------- Basic time-domain ----------
class AddGaussianNoise:
    def __init__(self, sigma: float = 0.01, p: float = 0.5):
        self.sigma, self.p = sigma, p
    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p: return x
        return x + torch.randn_like(x) * self.sigma

class AmplitudeScale:
    def __init__(self, low: float = 0.9, high: float = 1.1, p: float = 0.5):
        self.low, self.high, self.p = low, high, p
    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p: return x
        g = (self.high - self.low) * torch.rand(x.size(0), device=x.device) + self.low
        return x * g.view(-1, 1)

class TimeShift:
    """Circular shift by up to +/- max_shift samples."""
    def __init__(self, max_shift: int = 32, p: float = 0.5):
        self.max_shift, self.p = max_shift, p
    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p or self.max_shift <= 0: return x
        s = random.randint(-self.max_shift, self.max_shift)
        return torch.roll(x, shifts=s, dims=-1)

class TimeMask:
    """Mask a contiguous time span with zeros (SpecAugment-like time masking)."""
    def __init__(self, max_width: int = 64, p: float = 0.3):
        self.max_width, self.p = max_width, p
    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p or self.max_width <= 0: return x
        T = x.size(-1); w = random.randint(1, min(self.max_width, T))
        t0 = random.randint(0, T - w)
        x = x.clone()
        x[..., t0:t0 + w] = 0.0
        return x

class TimeWarp:
    """Lightweight elastic warping using linear resampling on a random grid."""
    def __init__(self, max_warp: float = 0.2, num_knots: int = 4, p: float = 0.3):
        self.max_warp, self.num_knots, self.p = max_warp, num_knots, p
    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p or self.max_warp <= 0: return x
        C, T = x.shape
        # random control points
        knots = torch.linspace(0, 1, self.num_knots, device=x.device)
        delta = (torch.rand(self.num_knots, device=x.device) * 2 - 1) * self.max_warp
        src = knots
        dst = torch.clamp(knots + delta, 0.0, 1.0).sort().values
        # build interpolation grid
        grid_t = torch.linspace(0, 1, T, device=x.device)
        # piecewise-linear mapping t' = f(t)
        idx = torch.bucketize(grid_t, dst[1:-1])
        left = dst[idx]; right = dst[torch.clamp(idx+1, max=self.num_knots-1)]
        left_src = src[idx]; right_src = src[torch.clamp(idx+1, max=self.num_knots-1)]
        alpha = (grid_t - left) / torch.clamp(right - left, 1e-6)
        tprime = left_src + alpha * (right_src - left_src)
        # resample with linear interp (per channel)
        t_idx = tprime * (T - 1)
        t0 = torch.clamp(torch.floor(t_idx).long(), 0, T - 1)
        t1 = torch.clamp(t0 + 1, 0, T - 1)
        w = (t_idx - t0.float()).unsqueeze(0)
        xw = (1 - w) * x[:, t0] + w * x[:, t1]
        return xw

# ---------- Frequency-aware ----------
class FreqShift:
    """Small frequency shift via phase ramp in FFT domain."""
    def __init__(self, max_shift_hz: float = 1.0, fs: float = 128.0, p: float = 0.3):
        self.max_shift_hz, self.fs, self.p = max_shift_hz, fs, p
    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p or self.max_shift_hz <= 0: return x
        C, T = x.shape
        f = (random.uniform(-self.max_shift_hz, self.max_shift_hz)) / self.fs
        t = torch.arange(T, device=x.device).float()
        ramp = torch.exp(2j * math.pi * f * t)  # complex ramp
        X = torch.fft.rfft(x, dim=-1)
        ramp_f = torch.fft.rfft(ramp, dim=0)
        Y = X * ramp_f  # convolution in time ~ mult in freq of ramps; approximation
        y = torch.fft.irfft(Y, n=T, dim=-1)
        return y.real

class BandstopDropout:
    """Zero a random narrow frequency band (simulated notch)."""
    def __init__(self, fs: float = 128.0, width_hz: float = 2.0, low_hz: float = 6.0, high_hz: float = 30.0, p: float = 0.3):
        self.fs, self.width_hz, self.low_hz, self.high_hz, self.p = fs, width_hz, low_hz, high_hz, p
    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p: return x
        C, T = x.shape
        X = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(T, 1.0 / self.fs).to(x.device)
        center = random.uniform(self.low_hz, self.high_hz)
        mask = (freqs >= (center - self.width_hz / 2)) & (freqs <= (center + self.width_hz / 2))
        X[..., mask] = 0
        return torch.fft.irfft(X, n=T, dim=-1)

# ---------- Spatial/channel ----------
class ChannelDropout:
    """Randomly drop (zero) a subset of channels to simulate cap variability."""
    def __init__(self, max_drop: int = 2, p: float = 0.3):
        self.max_drop, self.p = max_drop, p
    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p or self.max_drop <= 0: return x
        C, T = x.shape
        k = random.randint(1, min(self.max_drop, C-1))
        idx = torch.randperm(C, device=x.device)[:k]
        x = x.clone(); x[idx, :] = 0.0
        return x

# ---------- Mix-based (need special loss handling) ----------
class MixupBatch:
    """Applies mixup to a batch: returns mixed x and (y_a, y_b, lam)."""
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    def __call__(self, x: Tensor, y: Tensor):
        if self.alpha <= 0: return x, y, None
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        perm = torch.randperm(x.size(0), device=x.device)
        x_m = lam * x + (1 - lam) * x[perm]
        return x_m, (y, y[perm]), lam

class CutCatBatch:
    """Cut-and-concatenate (CutCat): splice two segments from two trials."""
    def __init__(self, seg_len: int = 64, p: float = 0.5):
        self.seg_len, self.p = seg_len, p
    def __call__(self, x: Tensor, y: Tensor):
        if random.random() > self.p: return x, y, None
        B, C, T = x.shape
        perm = torch.randperm(B, device=x.device)
        t0 = random.randint(0, max(0, T - self.seg_len))
        # x_new = [x[:,:,:t0] | x_perm[:,:,t0:t0+L] | x[:,:,t0+L:]]
        x2 = x[perm]
        x_new = torch.cat([x[:, :, :t0], x2[:, :, t0:t0 + self.seg_len], x[:, :, t0 + self.seg_len:]], dim=-1)
        lam = 0.5 * (self.seg_len / T) + 0.5  # crude mixing coefficient
        return x_new, (y, y[perm]), lam

# ---------- Compose ----------
class Compose:
    def __init__(self, ops: List):
        self.ops = ops
    def __call__(self, x: Tensor) -> Tensor:
        for op in self.ops:
            x = op(x)
        return x
