
# -*- coding: utf-8 -*-
"""
    Integrated Signal Processing Framework
    with Added Trumpet Chirp Functionality
"""

import numpy as np
import torch
from torch.utils.data import IterableDataset
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Dict, Callable, Optional, Union
from abc import ABC, abstractmethod
import os
import hashlib

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- Original utility functions ----

def rms(sig: np.ndarray, axis=None) -> float:
    """Calculate the root mean square of a signal."""
    return np.sqrt(np.mean(sig**2, axis=axis))

def whitenoise(numsamples: int, rmsnorm: bool = False) -> np.ndarray:
    """Generate white noise."""
    noise = np.random.randn(numsamples)
    if rmsnorm:
        noise *= (1/rms(noise))
    else:
        noise *= (1/np.max(np.abs(noise)))
    return noise 

def pinknoise(numsamples: int, rmsnorm: bool = False) -> np.ndarray:
    """Generate pink noise."""
    X = np.fft.rfft(whitenoise(numsamples))
    S = np.sqrt(np.arange(len(X))+1.)  # +1 to avoid divide by zero
    X = X / S
    noise = np.fft.irfft(X)
    if rmsnorm:
        noise *= (1/rms(noise))
    else:
        noise *= (1/np.max(np.abs(noise)))
    return noise[:numsamples]  # Ensure the output length matches the input

def brownoise(numsamples: int, rmsnorm: bool = False) -> np.ndarray:
    """Generate brown noise."""
    noise = np.cumsum(whitenoise(numsamples))
    noise -= np.mean(noise)
    if rmsnorm:
        noise *= (1/rms(noise))
    else:
        noise *= (1/np.max(np.abs(noise)))
    return noise

# ---- Modulator classes ----

class Modulator(ABC):
    """Abstract base class for signal modulators."""
    @abstractmethod
    def __call__(self, timevec: np.ndarray) -> np.ndarray:
        pass

class Linear(Modulator):
    """Linear modulator."""
    def __init__(self, y0: float, y1: float):
        self.y0 = y0
        self.y1 = y1
        self.dy = y1 - y0

    def __call__(self, timevec: np.ndarray) -> np.ndarray:
        dt = timevec[-1] - timevec[0]
        alpha = self.dy / dt
        return alpha * timevec + self.y0

class Exponential(Modulator):
    """Exponential modulator."""
    def __init__(self, y0: float, y1: float):
        self.y0 = y0
        self.y1 = y1

    def __call__(self, timevec: np.ndarray) -> np.ndarray:
        dt = timevec[-1] - timevec[0]
        rate = np.log(self.y1 / self.y0) / dt
        return self.y0 * np.exp(rate * timevec)

class LinearChirp(Modulator):
    """Linear frequency sweeping modulator (chirp)."""
    def __init__(self, f0: float, f1: float):
        self.f0 = f0  # Start frequency
        self.f1 = f1  # End frequency

    def __call__(self, timevec: np.ndarray) -> np.ndarray:
        return np.linspace(self.f0, self.f1, len(timevec))  # Linear frequency sweep

class ExponentialChirp(Modulator):
    """Exponential frequency sweeping modulator (chirp)."""
    def __init__(self, f0: float, f1: float):
        self.f0 = f0
        self.f1 = f1

    def __call__(self, timevec: np.ndarray) -> np.ndarray:
        return self.f0 * (self.f1 / self.f0) ** (timevec / timevec[-1])  # Exponential sweep

class InverseExponential(Modulator):
    """Inverse exponential modulator."""
    def __init__(self, y0: float, y1: float):
        self.y0 = y0
        self.y1 = y1

    def __call__(self, timevec: np.ndarray) -> np.ndarray:
        dt = timevec[-1] - timevec[0]
        rate = np.log(self.y0 / self.y1) / dt
        return self.y0 * np.exp(-rate * timevec)

class Sinusoidal(Modulator):
    """Sinusoidal modulator."""
    def __init__(self, y0: float, y1: float, f0: float):
        self.y0 = y0
        self.y1 = y1
        self.f0 = f0

    def __call__(self, timevec: np.ndarray) -> np.ndarray:
        amplitude = (self.y1 - self.y0) / 2
        offset = (self.y1 + self.y0) / 2
        return amplitude * np.sin(2 * np.pi * self.f0 * timevec) + offset

# ---- NEW: Trumpet Chirp classes and functions ----

class TrumpetChirp(Modulator):
    """Trumpet-shaped exponential chirp modulator with custom amplitude envelope."""
    def __init__(self, f0: float, f1: float, exp_factor: float = 3.0):
        self.f0 = f0  # Start frequency
        self.f1 = f1  # End frequency
        self.exp_factor = exp_factor  # Controls the exponential curve shape

    def __call__(self, timevec: np.ndarray) -> np.ndarray:
        """Returns instantaneous frequency values for a trumpet-shaped chirp."""
        t_norm = timevec / timevec[-1]
        return self.f0 * (self.f1 / self.f0) ** (t_norm ** self.exp_factor)

def generate_trumpet_envelope(timevec: np.ndarray, 
                             initial_amp: float = 0.3,
                             final_amp: float = 0.5,
                             initial_section: float = 0.20,
                             middle_section: float = 0.40) -> np.ndarray:
    """
    Generate a trumpet-shaped amplitude envelope.
    
    Args:
        timevec: Time vector
        initial_amp: Initial amplitude value
        final_amp: Final amplitude value
        initial_section: Proportion of signal for initial section
        middle_section: Proportion of signal for initial + middle sections
    
    Returns:
        Amplitude envelope array
    """
    num_samples = len(timevec)
    t_norm = timevec / timevec[-1]
    env = np.ones(num_samples)
    
    # Calculate indices
    initial_idx = int(initial_section * num_samples)
    middle_idx = int(middle_section * num_samples)
    
    # 1. Initial section (constant with slight decrease)
    env[:initial_idx] = initial_amp * (1 - 0.1 * t_norm[:initial_idx] / initial_section)
    
    # 2. Middle section (slight dip)
    middle_t = (t_norm[initial_idx:middle_idx] - initial_section) / (middle_section - initial_section)
    middle_dip = 0.8 * initial_amp  # Lowest amplitude in the dip
    env[initial_idx:middle_idx] = (initial_amp * 0.9) - ((initial_amp * 0.9) - middle_dip) * np.sin(np.pi * middle_t)
    
    # 3. Final section (exponential increase)
    final_t = (t_norm[middle_idx:] - middle_section) / (1 - middle_section)
    # Use squared function for more gradual initial increase and steeper final increase
    env[middle_idx:] = middle_dip + (final_amp - middle_dip) * final_t**2
    
    # Apply slight smoothing to avoid abrupt transitions
    env = gaussian_filter1d(env, sigma=num_samples/500)
    
    return env

def generate_trumpet_chirp_signal(params: Dict, modulator: TrumpetChirp) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a trumpet-shaped chirp signal with custom frequency and amplitude envelopes.
    
    Args:
        params: Dictionary with signal parameters
        modulator: TrumpetChirp modulator instance
    
    Returns:
        Tuple containing:
            - Generated signal
            - Instantaneous frequency
    """
    N = params['N']
    tvec = np.linspace(0, params['duration'], N, endpoint=False)
    
    # Get instantaneous frequency over time
    inst_freq = modulator(tvec)
    
    # Calculate phase by integrating frequency
    dt = 1.0 / params['Fs']
    phase = 2 * np.pi * np.cumsum(inst_freq) * dt
    
    # Generate base chirp signal
    signal = np.sin(phase)
    
    # Create trumpet-shaped amplitude envelope
    env = generate_trumpet_envelope(
        tvec,
        initial_amp=params.get('initial_amp', 0.3),
        final_amp=params.get('final_amp', 0.5)
    )
    
    # Apply envelope to signal
    signal = signal * env
    
    # Scale signal
    signal *= params.get('chirp_amplitude', 1.0) / max(np.max(np.abs(signal)), 1e-10)
    
    return signal, inst_freq

# ---- Existing signal generation functions ----

def get_signal_params() -> Dict:
    """Generate random parameters for signal generation."""
    param_space = {
        'Fs': np.random.uniform(40000, 50000),
        'duration': 3.0, 
        'Amplitude': np.random.uniform(20, 40),  # For AM/FM signals
        'chirp_amplitude': np.random.uniform(0.2, 0.8),  # For chirp amplitude
        'm': np.random.uniform(0.3, 1.0),
        'a0': np.random.uniform(0.8, 1.2),
        'alpha': np.random.uniform(3, 7),
        'freq_AM': np.random.uniform(0, 12500),  # Fs/4
        'freq_FM': np.random.uniform(0.1, 5000),  # Fs/10
        'modulating_freqs': np.random.uniform(3, 7),
        'num_signals': np.random.randint(1, 5),
        'am_ymin': 0.5,
        'am_ymax': 2.0,
        'am_fmin': 1,
        'am_fmax': 100,
        'fm_fmin': 1,
        'fm_fmax': 100,
        'fm_cfmin': 1000,
        'fm_cfmax': 5000,
        # NEW: Additional parameters for trumpet chirp
        'initial_amp': np.random.uniform(0.2, 0.4),
        'final_amp': np.random.uniform(0.4, 0.6),
        'exp_factor': np.random.uniform(2.5, 3.5)
    }
    
    param_space['T'] = 1.0 / param_space['Fs']
    param_space['N'] = int(np.ceil(param_space['Fs'] * param_space['duration']))

    return param_space

def fm_am_sin(a0: float, mdepth: float, phi_am: np.ndarray, phi_fm: np.ndarray) -> np.ndarray:
    """Returns frequency and amplitude modulated sinusoidal signal."""
    ainst = a0
    if mdepth > 0:
        ainst += mdepth * np.cos(phi_am)
    return ainst * np.sin(phi_fm)

def convert_inst_freq_to_phase(inst_freq: np.ndarray, initial_phase: float) -> np.ndarray:
    """Convert instantaneous frequency to phase."""
    return np.cumsum(inst_freq) + initial_phase

def generate_chirp_signal(params: Dict, modulator_fm: Modulator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a frequency sweeping (chirp) signal with improved stability."""
    N = params['N']
    tvec = np.linspace(0, params['duration'], N, endpoint=False)

    inst_freq = modulator_fm(tvec)  # Get instantaneous frequency over time

    # Compute phase correctly (fixed)
    dt = 1.0 / params['Fs']
    phase = 2 * np.pi * np.cumsum(inst_freq) * dt  # Correct phase calculation
    signal = np.sin(phase)  # Generate chirp signal

    # Apply proper amplitude scaling (fixed)
    signal *= params['chirp_amplitude'] / (np.max(np.abs(signal)) + 1e-8)

    # Use a smooth amplitude envelope instead of random scaling (fixed)
    amplitude_envelope = np.hanning(len(signal))  # Hann window for smoothness
    signal *= amplitude_envelope

    return signal, inst_freq


def generate_modulated_signal(params: Dict, modulator_am: Modulator, modulator_fm: Modulator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a modulated signal based on given parameters and modulators."""
    N = params['N']
    tvec = np.linspace(0, params['duration'], N, endpoint=False)
    
    am_mod = modulator_am(tvec)
    fm_mod = modulator_fm(tvec)
    
    phi_am = 2 * np.pi * params['freq_AM'] * tvec
    phi_fm = 2 * np.pi * params['freq_FM'] * tvec + params['alpha'] * np.cumsum(fm_mod) * params['T']
    
    signal = fm_am_sin(params['a0'], params['m'], phi_am, phi_fm)
    
    return signal, phi_am, phi_fm

def generate_spectrograms(signal: np.ndarray, noise: np.ndarray, Fs: float, nperseg: int = 256, noverlap: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    min_length = min(len(signal), len(noise))
    signal = signal[:min_length]
    noise = noise[:min_length]

    freq, time, Sxx = spectrogram(signal + noise, fs=Fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Sxx_signal = spectrogram(signal, fs=Fs, nperseg=nperseg, noverlap=noverlap)

    return Sxx, Sxx_signal, freq, time

def compute_log_power_spectrum(Sxx: np.ndarray) -> np.ndarray:
    """Compute the log power spectrum from a spectrogram."""
    return 10 * np.log10(np.maximum(Sxx, 1e-10))

# ---- Updated Dataset class with trumpet chirp support ----

class InfiniteDataset(IterableDataset):
    def __init__(self, batch_size: int = 10000, spectrogram_size: Tuple[int, int] = (129, 256)):
        """Initialize dataset parameters."""
        self.batch_size = batch_size
        self.modulators = [Linear, Exponential, InverseExponential, Sinusoidal, LinearChirp, ExponentialChirp]
        self.noise_funcs = [whitenoise, pinknoise, brownoise]
        self.spectrogram_size = spectrogram_size

    def __iter__(self):
        """Allows Dataset to be used as an infinite iterable."""
        return iter(self.generator())

    def generator(self):
        """Generator function that runs indefinitely."""
        while True:
            yield self.generate_data()

    def scale_noise_to_snr(self, signal: np.ndarray, noise: np.ndarray, target_snr: float) -> np.ndarray:
        """Scale noise to match a target SNR level."""
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        
        # Avoid division by zero
        if noise_power == 0:
            print("⚠ Warning: Noise power is zero. Adding small noise to avoid division by zero.")
            noise_power = 1e-10
        
        desired_noise_power = signal_power * 10**(-target_snr / 10)
        scaling_factor = np.sqrt(desired_noise_power / noise_power)
        
        return noise * scaling_factor

    def generate_data(self) -> Dict:
        """Generate a single data point with random noise levels and spectrograms."""
        params = get_signal_params()
        
        signal_type = np.random.choice(['am_fm', 'chirp', 'trumpet_chirp'], p=[0.4, 0.3, 0.3])
        
        if signal_type == 'trumpet_chirp':
            trumpet_chirp = TrumpetChirp(
                f0=np.random.uniform(0.5, 1.0),
                f1=np.random.uniform(60.0, 100.0),
                exp_factor=params['exp_factor']
            )
            signal, inst_freq = generate_trumpet_chirp_signal(params, trumpet_chirp)
            phi_am, phi_fm = None, None
            signal_subtype = 'trumpet_chirp'
        
        elif signal_type == 'chirp':
            fm_mod_class = np.random.choice([LinearChirp, ExponentialChirp])
            fm_modulator = fm_mod_class(f0=np.random.uniform(500, 2000), f1=np.random.uniform(3000, 8000))
            signal, inst_freq = generate_chirp_signal(params, fm_modulator)
            phi_am, phi_fm = None, None
            signal_subtype = 'standard_chirp'
        
        else:
            am_mod_class = np.random.choice(self.modulators[:-2])  
            fm_mod_class = np.random.choice(self.modulators[:-2])

            if am_mod_class == Sinusoidal:
                am_mod = am_mod_class(params['am_ymin'], params['am_ymax'], 
                                      np.random.uniform(params['am_fmin'], params['am_fmax']))
            else:
                am_mod = am_mod_class(params['am_ymin'], params['am_ymax'])

            if fm_mod_class == Sinusoidal:
                fm_mod = fm_mod_class(params['fm_fmin'], params['fm_fmax'], 
                                      np.random.uniform(params['fm_fmin'], params['fm_fmax']))
            else:
                fm_mod = fm_mod_class(params['fm_fmin'], params['fm_fmax'])

            signal, phi_am, phi_fm = generate_modulated_signal(params, am_mod, fm_mod)
            inst_freq = None
            signal_subtype = 'am_fm'

        # Select a random noise function
        noise_func = np.random.choice(self.noise_funcs)
        noise = noise_func(params['N'], rmsnorm=True)

        # ---- APPLY RANDOM SNR BETWEEN -10 AND 30 dB ----
        target_snr = np.random.uniform(-10, 30)
        noise = self.scale_noise_to_snr(signal, noise, target_snr)

        # Generate spectrograms
        Sxx, Sxx_signal, freq, time = generate_spectrograms(signal, noise, params['Fs'])

        # Compute log power spectrograms
        log_power_Sxx = compute_log_power_spectrum(Sxx)
        log_power_Sxx_signal = compute_log_power_spectrum(Sxx_signal)

        # Print SNR values for verification
        computed_snr = 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))
        print(f"🎯 Target SNR: {target_snr:.2f} dB | Computed SNR: {computed_snr:.2f} dB")

        return {
            'SNR': target_snr,
            'SNR_computed': computed_snr,
            'clean_spectrogram': torch.FloatTensor((Sxx_signal - Sxx_signal.min()) / (Sxx_signal.max() - Sxx_signal.min())),
            'noisy_spectrogram': torch.FloatTensor((Sxx - Sxx.min()) / (Sxx.max() - Sxx.min())),
            'log_power_Sxx': log_power_Sxx,
            'log_power_Sxx_signal': log_power_Sxx_signal,
            'signal': signal,
            'noise': noise,
            'Sxx': Sxx,
            'Sxx_signal': Sxx_signal,
            'freq': freq,
            'time': time,
            'tvec': np.linspace(0, params['duration'], params['N'], endpoint=False),
            'N': params['N'],
            'Fs': params['Fs'],
            'T': params['T'],
            'inst_freq_AM': np.diff(phi_am) / (2 * np.pi * params['T']) if phi_am is not None else None,
            'inst_freq_FM': np.diff(phi_fm) / (2 * np.pi * params['T']) if phi_fm is not None else None,
            'inst_freq': inst_freq,
            'signal_type': signal_type,
            'signal_subtype': signal_subtype
        }


    def pad_or_crop_spectrogram(self, Sxx: np.ndarray) -> np.ndarray:
        """Pad or crop spectrogram to match target size."""
        target_height, target_width = self.spectrogram_size
        current_height, current_width = Sxx.shape

        if current_height < target_height:
            Sxx = np.pad(Sxx, ((0, target_height - current_height), (0, 0)), mode='constant')
        elif current_height > target_height:
            Sxx = Sxx[:target_height, :]
        
        if current_width < target_width:
            Sxx = np.pad(Sxx, ((0, 0), (0, target_width - current_width)), mode='constant')
        elif current_width > target_width:
            Sxx = Sxx[:, :target_width]
        
        return Sxx


# ---- Updated plotting functions ----

def plot_chirp_signal(signal: np.ndarray, tvec: np.ndarray, Sxx: np.ndarray, freq: np.ndarray, time: np.ndarray, signal_type: str = 'standard_chirp', save_path=None):
    """Plot chirp signals with full visibility in time domain and spectrogram views."""
    
    plt.figure(figsize=(12, 6), dpi=300)  

    # Determine the full duration of the signal
    duration = tvec[-1]  
    Fs = len(tvec) / duration  # Estimated sampling rate

    # Ensure we plot enough samples
    plot_samples = int(Fs * duration)  # Capture the full signal

    # Time domain plot
    plt.subplot(2, 1, 1)
    plt.plot(tvec[:plot_samples], signal[:plot_samples], linewidth=1)#  [:plot_samples]
    title_prefix = "Trumpet" if signal_type == 'trumpet_chirp' else "Standard"
    plt.title(f"{title_prefix} Chirp Signal (Time Domain)")
    plt.xlabel("Time [s]")
    plt.xlim(0, duration)  # Show full signal duration
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Spectrogram plot
    plt.subplot(2, 1, 2)
    plt.pcolormesh(time, freq, 10 * np.log10(np.maximum(Sxx, 1e-10)), shading="gouraud")
    plt.title(f"{title_prefix} Chirp Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Intensity [dB]")

    plt.tight_layout()

    # Save the plot if needed
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()

def plot_all_spectograms(data: Dict):
    """Plot various aspects of the generated signal."""
    plt.figure(figsize=(20, 20), dpi=300)

    # Time domain plot
    plt.subplot(6, 1, 1)
    plt.plot(data['tvec'][:1000], data['signal'][:1000])
    
    # Set title based on signal type
    if data.get('signal_type') == 'trumpet_chirp':
        title = 'Trumpet Chirp Signal in Time Domain'
    elif data.get('signal_type') == 'chirp':
        title = 'Standard Chirp Signal in Time Domain'
    else:
        title = 'Modulated Signal in Time Domain'
        
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # Frequency domain plot
    plt.subplot(6, 1, 2)
    yf = np.fft.fft(data['signal'])
    xf = np.fft.fftfreq(data['N'], data['T'])[:data['N']//2]
    plt.plot(xf, 2.0/data['N'] * np.abs(yf[0:data['N']//2]))
    plt.title('Frequency Domain FFT')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')

    # Spectrogram for signal + noise
    plt.subplot(6, 1, 3)
    plt.pcolormesh(data['time'], data['freq'], 10 * np.log10(np.maximum(data['Sxx'], 1e-10)), shading='gouraud')
    plt.title(f"{data.get('signal_type', 'Signal')} + Noise Spectrogram")
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='Intensity [dB]')

    # Spectrogram for signal only
    plt.subplot(6, 1, 4)
    plt.pcolormesh(data['time'], data['freq'], 10 * np.log10(np.maximum(data['Sxx_signal'], 1e-10)), shading='gouraud')
    plt.title(f"Clean {data.get('signal_type', 'Signal')} Spectrogram")
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='Intensity [dB]')

    # Log Power Spectrogram for signal + noise
    plt.subplot(6, 1, 5)
    plt.pcolormesh(data['time'], data['freq'], data['log_power_Sxx'], shading='gouraud')
    plt.title(f"Log Power Spectrogram of {data.get('signal_type', 'Signal')} + Noise")
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='Log Power Intensity [dB]')

    # Log Power Spectrogram for signal only
    plt.subplot(6, 1, 6)
    plt.pcolormesh(data['time'], data['freq'], data['log_power_Sxx_signal'], shading='gouraud')
    plt.title(f"Log Power Spectrogram of Clean {data.get('signal_type', 'Signal')}")
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='Log Power Intensity [dB]')

    plt.tight_layout()
    plt.show()

# ---- Testing functions ----

def check_infinite_signal_generation(num_samples: int = 10):
    """Continuously generate and plot signals for debugging."""
    print("🔄 Starting infinite signal generation...\n")
    
    dataset = InfiniteDataset(batch_size=num_samples)
    generated_signals, unique_signals = 0, set()

    try:
        for i, data in enumerate(dataset):
            # Ensure valid tensor conversion
            if isinstance(data['clean_spectrogram'], torch.Tensor):
                signal = data['clean_spectrogram'].detach().cpu().numpy()
            else:
                print("⚠ Warning: clean_spectrogram is not a tensor, skipping...")
                continue
            
            generated_signals += 1
            unique_signals.add(hashlib.md5(signal.tobytes()).hexdigest())

            print(f"✅ Generated signal {generated_signals}/{num_samples} | Unique: {len(unique_signals)}", flush=True)

            # Plot based on signal type
            if data['signal_type'] == 'trumpet_chirp':
                print(f"📡 Plotting Trumpet Chirp {generated_signals}...")
                Sxx, _, freq, time = generate_spectrograms(data['signal'], np.zeros_like(data['signal']), data['Fs'])
                plot_chirp_signal(data['signal'], data['tvec'], Sxx, freq, time, signal_type='trumpet_chirp')

            elif data['signal_type'] == 'chirp':
                print(f"📡 Plotting Standard Chirp {generated_signals}...")
                Sxx, _, freq, time = generate_spectrograms(data['signal'], np.zeros_like(data['signal']), data['Fs'])
                plot_chirp_signal(data['signal'], data['tvec'], Sxx, freq, time, signal_type='standard_chirp')

            else:
                print(f"📡 Plotting AM/FM {generated_signals}...")
                plot_all_spectograms(data)

            plt.show()  # Ensure plots are displayed

            if generated_signals >= num_samples:
                break  # Stop after reaching the desired number of samples

    except KeyboardInterrupt:
        print(f"\n❌ Stopped after {generated_signals} signals.")
        print(f"📊 Total Unique Signals: {len(unique_signals)}")
    except Exception as e:
        print(f"⚠ Error: {e}")

    return len(unique_signals) == generated_signals


def run_validation(num_iterations: int = 1, num_samples: int = 5):
    """Validate different signal types and check for anomalies."""
    print(f"🔍 Starting validation: {num_iterations} iterations, {num_samples} samples per iteration.")

    for iteration in range(num_iterations):
        print(f"\n🟢 Iteration {iteration + 1}/{num_iterations}...")

        dataset = InfiniteDataset(batch_size=num_samples)
        for i, data in enumerate(dataset):
            signal_type = data['signal_type']
            print(f"✅ Validating {signal_type} Signal {i + 1}")

            # Validate trumpet chirp
            if signal_type == 'trumpet_chirp':
                inst_freq = data['inst_freq']
                if inst_freq is not None and np.any(np.abs(np.diff(inst_freq)) > 500):
                    print(f"⚠ Warning: Sudden frequency jump in Trumpet Chirp {i + 1}")

                Sxx, _, freq, time = generate_spectrograms(data['signal'], data['noise'], data['Fs'])
                if np.max(Sxx) < 0.1:
                    print(f"⚠ Warning: Low energy in Trumpet Chirp {i + 1}")

            # Validate standard chirp
            elif signal_type == 'chirp':
                if np.std(data['signal']) < 0.1:
                    print(f"⚠ Warning: Low variance in Standard Chirp {i + 1}")

            # Validate AM/FM signal
            else:
                if data.get('inst_freq_FM') is not None and np.any(np.isnan(data['inst_freq_FM'])):
                    print(f"⚠ Warning: NaN values in FM instantaneous frequency in signal {i + 1}")

            # General validation
            if np.any(np.isnan(data['signal'])) or np.any(np.isinf(data['signal'])):
                print(f"⚠ Error: NaN or Inf values in {signal_type} signal {i + 1}")

            # Print progress every 10 samples
            if (i + 1) % 10 == 0:
                print(f"✓ Processed {i + 1}/{num_samples} samples in iteration {iteration + 1}")

        print(f"✅ Iteration {iteration + 1} complete!")

    print("🎉 Validation complete! All signal types generated successfully.")

# ---- Enhanced visualization functions ----

def plot_comparative_spectrograms(signals_dict: Dict[str, Dict]):
    """
    Plot comparative spectrograms for different signal types.
    
    Args:
        signals_dict: Dictionary mapping signal type names to their data dictionaries
    """
    num_signals = len(signals_dict)
    fig, axs = plt.subplots(num_signals, 2, figsize=(14, 5 * num_signals), dpi=150)
    
    for i, (signal_name, data) in enumerate(signals_dict.items()):
        # Time domain plot
        axs[i, 0].plot(data['tvec'], data['signal'])
        axs[i, 0].set_title(f"{signal_name} (Time Domain)")
        axs[i, 0].set_xlabel("Time [s]")
        axs[i, 0].set_ylabel("Amplitude")
        axs[i, 0].grid(True)
        
        # Spectrogram
        im = axs[i, 1].pcolormesh(
            data['time'], 
            data['freq'], 
            10 * np.log10(np.maximum(data['Sxx_signal'], 1e-10)), 
            shading='gouraud'
        )
        axs[i, 1].set_title(f"{signal_name} (Spectrogram)")
        axs[i, 1].set_xlabel("Time [s]")
        axs[i, 1].set_ylabel("Frequency [Hz]")
        fig.colorbar(im, ax=axs[i, 1], label="Intensity [dB]")
    
    plt.tight_layout()
    plt.show()

def visualize_trumpet_envelope(params: Dict = None):
    """
    Visualize the trumpet envelope shape.
    
    Args:
        params: Optional parameters for customizing the envelope
    """
    if params is None:
        params = {
            'duration': 3.0,
            'Fs': 44100,
            'initial_amp': 0.3,
            'final_amp': 0.6
        }
    
    N = int(params['duration'] * params['Fs'])
    tvec = np.linspace(0, params['duration'], N, endpoint=False)
    
    # Generate default envelope
    default_env = generate_trumpet_envelope(
        tvec,
        initial_amp=params['initial_amp'],
        final_amp=params['final_amp']
    )
    
    # Generate variations
    variation1 = generate_trumpet_envelope(
        tvec,
        initial_amp=params['initial_amp'] * 0.8,
        final_amp=params['final_amp'] * 1.2
    )
    
    variation2 = generate_trumpet_envelope(
        tvec,
        initial_amp=params['initial_amp'] * 1.2,
        final_amp=params['final_amp'] * 0.8
    )
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(tvec, default_env, label='Default Envelope')
    plt.plot(tvec, variation1, label='Stronger Final Amp')
    plt.plot(tvec, variation2, label='Stronger Initial Amp')
    plt.title('Trumpet Signal Envelope Variations')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """Main function to generate signals indefinitely, log SNR values, and visualize them."""
    #print("========== Infinite Signal Generation ==========")

    dataset = InfiniteDataset()

    try:
        for i, data in enumerate(dataset):
            print(f"🔄 Generating signal {i + 1} of type: {data['signal_type']}")

            # ✅ Fix KeyError: Ensure correct SNR keys
            if 'SNR' in data and 'SNR_computed' in data:
                print(f"✅ Target SNR: {data['SNR']:.2f} dB | Computed SNR: {data['SNR_computed']:.2f} dB")
            else:
                print("⚠ Warning: SNR values missing in dataset output.")

            # ✅ Plot visualization based on signal type
            if data['signal_type'] == 'trumpet_chirp':
                Sxx, _, freq, time = generate_spectrograms(data['signal'], np.zeros_like(data['signal']), data['Fs'])
                plot_chirp_signal(data['signal'], data['tvec'], Sxx, freq, time, signal_type='trumpet_chirp')
            elif data['signal_type'] == 'chirp':
                Sxx, _, freq, time = generate_spectrograms(data['signal'], np.zeros_like(data['signal']), data['Fs'])
                plot_chirp_signal(data['signal'], data['tvec'], Sxx, freq, time, signal_type='standard_chirp')
            else:
                plot_all_spectograms(data)

    except KeyboardInterrupt:
        print("\n❌ Stopping infinite signal generation.")

if __name__ == "__main__":
    main()




