import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, ifft
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Callable
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RadarParameters:
    fs: float = 10e6
    fc: float = 10e9
    bandwidth: float = 150e6
    pulse_width: float = 10e-6
    prf: float = 1000
    num_pulses: int = 64
    range_bins: int = 256
    doppler_bins: int = 64
    snr_db: float = 20

class WindowFunctionBenchmark:
    def __init__(self, params: RadarParameters):
        self.params = params
        self.window_functions = {
            'Rectangular': lambda N: np.ones(N),
            'Hanning': np.hanning,
            'Hamming': np.hamming,
            'Blackman': np.blackman,
            'Blackman-Harris': self.blackman_harris,
            'Kaiser β=3': lambda N: np.kaiser(N, 3),
            'Kaiser β=5': lambda N: np.kaiser(N, 5),
            'Kaiser β=7': lambda N: np.kaiser(N, 7),
            'Tukey α=0.25': lambda N: self.tukey(N, 0.25),
            'Tukey α=0.5': lambda N: self.tukey(N, 0.5),
            'Flat-top': self.flat_top,
            'Nuttall': self.nuttall,
            'Taylor': lambda N: self.taylor(N, 4, 30)
        }
    
    def benchmark_execution_time(self, window_func: Callable, size: int, iterations: int = 100) -> float:
        """Benchmark the execution time of a window function"""
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = window_func(size)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1e6)  # Convert to microseconds
        
        return np.mean(times)
    
    @staticmethod
    def blackman_harris(N):
        n = np.arange(N)
        return (0.35875 - 0.48829 * np.cos(2*np.pi*n/(N-1)) + 
                0.14128 * np.cos(4*np.pi*n/(N-1)) - 0.01168 * np.cos(6*np.pi*n/(N-1)))
    
    @staticmethod
    def tukey(N, alpha=0.5):
        n = np.arange(N)
        w = np.ones(N)
        transition = int(alpha * N / 2)
        for i in range(transition):
            w[i] = 0.5 * (1 + np.cos(np.pi * (2*i/(alpha*N) - 1)))
            w[N-i-1] = w[i]
        return w
    
    @staticmethod
    def flat_top(N):
        n = np.arange(N)
        a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
        w = a[0]
        for k in range(1, len(a)):
            w -= a[k] * np.cos(2*k*np.pi*n/(N-1))
        return w
    
    @staticmethod
    def nuttall(N):
        n = np.arange(N)
        a = [0.3635819, 0.4891775, 0.1365995, 0.0106411]
        w = a[0]
        for k in range(1, len(a)):
            w -= a[k] * np.cos(2*k*np.pi*n/(N-1))
        return w
    
    @staticmethod
    def taylor(N, nbar=4, sll=30):
        n = np.arange(N)
        w = np.ones(N)
        A = np.arccosh(10**(sll/20)) / np.pi
        for m in range(1, nbar):
            numerator = 1.0
            for n_val in range(1, nbar):
                if n_val != m:
                    numerator *= 1 - (m/n_val)**2
            fm = numerator / 2
            w += 2 * fm * np.cos(2 * np.pi * m * n / N)
        return w / np.max(w)

class RadarSignalProcessor:
    def __init__(self, params: RadarParameters):
        self.params = params
        self.range_resolution = 3e8 / (2 * params.bandwidth)
        self.max_range = 3e8 / (2 * params.prf)
        self.velocity_resolution = 3e8 / (2 * params.fc * params.num_pulses / params.prf)
        
    def generate_lfm_chirp(self) -> np.ndarray:
        t = np.linspace(0, self.params.pulse_width, 
                        int(self.params.pulse_width * self.params.fs))
        k = self.params.bandwidth / self.params.pulse_width
        return np.exp(1j * np.pi * k * t**2)
    
    def fft_correlate(self, x, y, mode='same'):
        N = len(x)
        M = len(y)
        L = N + M - 1
        X = fft(x, n=L)
        Y = fft(y[::-1], n=L)
        result = ifft(X * Y)
        if mode == 'same':
            start = (M - 1) // 2
            return result[start:start + N]
        return result
    
    def generate_target_return(self, range_m: float, velocity_ms: float, rcs_dbsm: float = 0) -> np.ndarray:
        chirp = self.generate_lfm_chirp()
        samples_per_pri = int(self.params.fs / self.params.prf)
        data_cube = np.zeros((self.params.num_pulses, samples_per_pri), dtype=complex)
        
        for pulse_idx in range(self.params.num_pulses):
            time_delay = 2 * (range_m + velocity_ms * pulse_idx / self.params.prf) / 3e8
            sample_delay = int(time_delay * self.params.fs)
            
            if 0 <= sample_delay < samples_per_pri - len(chirp):
                doppler_shift = 2 * velocity_ms * self.params.fc / 3e8
                phase_shift = np.exp(1j * 2 * np.pi * doppler_shift * pulse_idx / self.params.prf)
                amplitude = 10**(rcs_dbsm / 20) * phase_shift
                data_cube[pulse_idx, sample_delay:sample_delay + len(chirp)] = amplitude * chirp
                
        noise_power = 10**(-self.params.snr_db / 10)
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*data_cube.shape) + 
                                           1j * np.random.randn(*data_cube.shape))
        return data_cube + noise
    
    def pulse_compression(self, data: np.ndarray, window_func: Callable) -> np.ndarray:
        chirp = self.generate_lfm_chirp()
        window = window_func(len(chirp))
        matched_filter = np.conj(chirp[::-1]) * window
        compressed = np.zeros_like(data)
        for i in range(data.shape[0]):
            compressed[i] = self.fft_correlate(data[i], matched_filter, mode='same')
        return compressed
    
    def range_doppler_processing(self, data: np.ndarray, 
                               range_window: Callable, 
                               doppler_window: Callable) -> np.ndarray:
        compressed = self.pulse_compression(data, range_window)
        range_fft = fft(compressed, n=self.params.range_bins, axis=1)
        doppler_win = doppler_window(self.params.num_pulses)
        windowed_data = range_fft * doppler_win[:, np.newaxis]
        range_doppler = fftshift(fft(windowed_data, n=self.params.doppler_bins, axis=0), axes=0)
        return np.abs(range_doppler)**2

def generate_window_comparison_figure(benchmarker: WindowFunctionBenchmark):
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('Comprehensive Window Function Analysis for Radar Signal Processing', fontsize=18, y=0.96)
    
    window_names = list(benchmarker.window_functions.keys())[:8]
    
    # Position titles just above the upper edge of each subplot with perfect vertical alignment
    title_positions = [
        (0.125, 0.86), (0.375, 0.86), (0.625, 0.86), (0.875, 0.86),  # Row 1
        (0.125, 0.64), (0.375, 0.64), (0.625, 0.64), (0.875, 0.64),  # Row 2
        (0.125, 0.42), (0.375, 0.42), (0.625, 0.42), (0.875, 0.42),  # Row 3
        (0.125, 0.20), (0.375, 0.20), (0.625, 0.20), (0.875, 0.20),  # Row 4
    ]
    
    for idx, (name, func) in enumerate(list(benchmarker.window_functions.items())[:8]):
        # Window time domain plot
        ax = plt.subplot(4, 4, idx*2 + 1)
        N = 256
        window = func(N)
        ax.plot(window, 'b-', linewidth=2)
        ax.set_xlabel('Sample', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        ax.tick_params(labelsize=7)
        
        # Window frequency domain plot
        ax = plt.subplot(4, 4, idx*2 + 2)
        nfft = 2048
        W = fft(window, nfft)
        W_mag_db = 20 * np.log10(np.abs(W) / np.max(np.abs(W)) + 1e-300)
        freq_axis = np.linspace(0, 0.5, nfft // 2)
        ax.plot(freq_axis, W_mag_db[:nfft // 2], 'r-', linewidth=2)
        ax.set_ylim(-120, 5)
        ax.set_xlabel('Normalized Frequency', fontsize=8)
        ax.set_ylabel('Magnitude (dB)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        
        # Add titles above each subplot with proper spacing
        fig.text(title_positions[idx*2][0], title_positions[idx*2][1], 
                f'{name} Window', fontsize=11, ha='center', weight='bold')
        fig.text(title_positions[idx*2+1][0], title_positions[idx*2+1][1], 
                f'{name} Spectrum', fontsize=11, ha='center', weight='bold')
    
    plt.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.05, hspace=0.70, wspace=0.28)
    return fig

def generate_window_metrics_comparison(benchmarker: WindowFunctionBenchmark, params: RadarParameters):
    metrics_results = {}
    for name, func in benchmarker.window_functions.items():
        window = func(params.range_bins)
        N = len(window)
        nfft = 8 * N
        
        W = fft(window, nfft)
        W_mag = np.abs(W)
        W_mag_db = 20 * np.log10(W_mag / np.max(W_mag) + 1e-300)
        
        first_null = np.where(W_mag_db[1:nfft//2] < -60)[0]
        if len(first_null) > 0:
            first_null = first_null[0] + 1
        else:
            first_null = 10
            
        main_lobe_width = 2 * first_null / nfft * params.fs / 1e6
        
        side_lobe_start = min(first_null * 2, nfft // 4)
        if side_lobe_start < nfft // 2:
            peak_sidelobe = np.max(W_mag_db[side_lobe_start:nfft//2])
        else:
            peak_sidelobe = -80
        
        coherent_gain = np.sum(window)
        processing_gain = coherent_gain**2 / np.sum(window**2)
        enbw = N * np.sum(window**2) / np.sum(window)**2
        
        metrics_results[name] = {
            'main_lobe_width_mhz': main_lobe_width,
            'peak_sidelobe_db': peak_sidelobe,
            'processing_gain': processing_gain,
            'enbw': enbw
        }
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle('Window Function Performance Metrics', fontsize=18, y=0.96)
    
    # Add individual plot titles above graphs and below main title
    fig.text(0.25, 0.88, 'Main Lobe Width Comparison', fontsize=14, ha='center', weight='bold')
    fig.text(0.75, 0.88, 'Sidelobe Suppression Performance', fontsize=14, ha='center', weight='bold')
    fig.text(0.25, 0.43, 'Processing Gain Comparison', fontsize=14, ha='center', weight='bold')
    fig.text(0.75, 0.43, 'ENBW Comparison', fontsize=14, ha='center', weight='bold')
    
    window_names = list(metrics_results.keys())
    x_pos = np.arange(len(window_names))
    
    ax = axes[0, 0]
    values = [metrics_results[w]['main_lobe_width_mhz'] for w in window_names]
    bars = ax.bar(x_pos, values, color='skyblue', edgecolor='navy')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(window_names, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Main Lobe Width (MHz)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=10)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax = axes[0, 1]
    values = [metrics_results[w]['peak_sidelobe_db'] for w in window_names]
    bars = ax.bar(x_pos, values, color='lightcoral', edgecolor='darkred')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(window_names, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Peak Sidelobe Level (dB)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=10)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - abs(min(values))*0.05,
                f'{val:.1f}', ha='center', va='top', fontsize=10, weight='bold')
    
    ax = axes[1, 0]
    values = [metrics_results[w]['processing_gain'] for w in window_names]
    bars = ax.bar(x_pos, values, color='lightgreen', edgecolor='darkgreen')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(window_names, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Processing Gain', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=10)
    
    ax = axes[1, 1]
    values = [metrics_results[w]['enbw'] for w in window_names]
    bars = ax.bar(x_pos, values, color='plum', edgecolor='purple')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(window_names, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Equivalent Noise Bandwidth', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=10)
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.15, hspace=0.50, wspace=0.20)
    return fig, metrics_results

def generate_pulse_compression_demo(processor: RadarSignalProcessor, benchmarker: WindowFunctionBenchmark):
    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    fig.suptitle('Pulse Compression Performance with Different Windows', fontsize=18, y=0.96)
    
    # Add individual plot titles above graphs and below main title
    fig.text(0.25, 0.88, 'Rectangular Window', fontsize=14, ha='center', weight='bold')
    fig.text(0.75, 0.88, 'Hamming Window', fontsize=14, ha='center', weight='bold')
    fig.text(0.25, 0.58, 'Blackman Window', fontsize=14, ha='center', weight='bold')
    fig.text(0.75, 0.58, 'Kaiser β=5 Window', fontsize=14, ha='center', weight='bold')
    fig.text(0.25, 0.28, 'Taylor Window', fontsize=14, ha='center', weight='bold')
    fig.text(0.75, 0.28, 'Nuttall Window', fontsize=14, ha='center', weight='bold')
    
    target_range = 5000
    target_velocity = 0
    target_rcs = 10
    
    data = processor.generate_target_return(target_range, target_velocity, target_rcs)
    single_pulse = data[0]
    
    window_types = ['Rectangular', 'Hamming', 'Blackman', 'Kaiser β=5', 'Taylor', 'Nuttall']
    
    for idx, window_name in enumerate(window_types):
        ax = axes[idx // 2, idx % 2]
        
        window_func = benchmarker.window_functions[window_name]
        compressed = processor.pulse_compression(data, window_func)
        
        range_samples = np.arange(len(compressed[0]))
        range_meters = range_samples * 3e8 / (2 * processor.params.fs)
        
        compressed_db = 20 * np.log10(np.abs(compressed[0]) + 1e-10)
        ax.plot(range_meters/1000, compressed_db, 'b-', linewidth=2)
        
        peak_idx = np.argmax(np.abs(compressed[0]))
        peak_db = compressed_db[peak_idx]
        ax.axhline(y=peak_db - 3, color='r', linestyle='--', alpha=0.7, linewidth=2, label='3dB down')
        ax.axhline(y=peak_db - 13, color='g', linestyle='--', alpha=0.7, linewidth=2, label='13dB down')
        
        ax.set_xlabel('Range (km)', fontsize=12)
        ax.set_ylabel('Magnitude (dB)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(3, 7)
        ax.set_ylim(peak_db - 60, peak_db + 5)
        ax.legend(loc='upper right', fontsize=10)
        ax.tick_params(labelsize=10)
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.08, hspace=0.45, wspace=0.20)
    return fig

def generate_range_doppler_maps(processor: RadarSignalProcessor, benchmarker: WindowFunctionBenchmark):
    targets = [
        {'range': 5000, 'velocity': 100, 'rcs': 10},
        {'range': 8000, 'velocity': -50, 'rcs': 5},
        {'range': 12000, 'velocity': 200, 'rcs': 15},
        {'range': 15000, 'velocity': 0, 'rcs': 20}
    ]
    
    data_cube = np.zeros((processor.params.num_pulses, 
                        int(processor.params.fs / processor.params.prf)), dtype=complex)
    
    for target in targets:
        data_cube += processor.generate_target_return(
            target['range'], target['velocity'], target['rcs'])
    
    fig, axes = plt.subplots(2, 3, figsize=(22, 18))
    fig.suptitle('Range-Doppler Maps with Different Window Combinations', fontsize=18, y=0.96)
    
    # Add individual plot titles above graphs and below main title
    fig.text(0.17, 0.88, 'Rectangular & Rectangular', fontsize=14, ha='center', weight='bold')
    fig.text(0.5, 0.88, 'Hamming & Hamming', fontsize=14, ha='center', weight='bold')
    fig.text(0.83, 0.88, 'Blackman & Blackman', fontsize=14, ha='center', weight='bold')
    fig.text(0.17, 0.43, 'Kaiser β=5 & Kaiser β=5', fontsize=14, ha='center', weight='bold')
    fig.text(0.5, 0.43, 'Taylor & Taylor', fontsize=14, ha='center', weight='bold')
    fig.text(0.83, 0.43, 'Blackman & Taylor', fontsize=14, ha='center', weight='bold')
    
    window_pairs = [
        ('Rectangular', 'Rectangular'),
        ('Hamming', 'Hamming'),
        ('Blackman', 'Blackman'),
        ('Kaiser β=5', 'Kaiser β=5'),
        ('Taylor', 'Taylor'),
        ('Blackman', 'Taylor')
    ]
    
    range_axis = np.linspace(0, processor.max_range, processor.params.range_bins) / 1000
    velocity_axis = np.linspace(-processor.params.prf * 3e8 / (4 * processor.params.fc),
                               processor.params.prf * 3e8 / (4 * processor.params.fc),
                               processor.params.doppler_bins)
    
    for idx, (range_win_name, doppler_win_name) in enumerate(window_pairs):
        ax = axes[idx // 3, idx % 3]
        
        range_win = benchmarker.window_functions[range_win_name]
        doppler_win = benchmarker.window_functions[doppler_win_name]
        
        rd_map = processor.range_doppler_processing(data_cube, range_win, doppler_win)
        rd_map_db = 10 * np.log10(rd_map + 1e-10)
        
        im = ax.imshow(rd_map_db, aspect='auto',
                      extent=[range_axis[0], range_axis[-1],
                             velocity_axis[0], velocity_axis[-1]],
                      cmap='viridis', vmin=np.max(rd_map_db) - 60)
        
        ax.set_xlabel('Range (km)', fontsize=12)
        ax.set_ylabel('Velocity (m/s)', fontsize=12)
        ax.tick_params(labelsize=11)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Power (dB)', fontsize=11)
        cbar.ax.tick_params(labelsize=10)
        
        for target in targets:
            ax.plot(target['range']/1000, target['velocity'], 'r+', 
                   markersize=12, markeredgewidth=3, alpha=0.9)
    
    plt.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.08, hspace=0.50, wspace=0.30)
    return fig

def generate_ambiguity_function(processor: RadarSignalProcessor, benchmarker: WindowFunctionBenchmark):
    fig, axes = plt.subplots(2, 3, figsize=(22, 18))
    fig.suptitle('Ambiguity Functions for Different Window Functions', fontsize=18, y=0.96)
    
    # Add individual plot titles with proper vertical spacing
    fig.text(0.17, 0.85, 'Rectangular Window', fontsize=14, ha='center', weight='bold')
    fig.text(0.5, 0.85, 'Hamming Window', fontsize=14, ha='center', weight='bold')
    fig.text(0.83, 0.85, 'Blackman Window', fontsize=14, ha='center', weight='bold')
    fig.text(0.17, 0.40, 'Kaiser β=5 Window', fontsize=14, ha='center', weight='bold')
    fig.text(0.5, 0.40, 'Taylor Window', fontsize=14, ha='center', weight='bold')
    fig.text(0.83, 0.40, 'Flat-top Window', fontsize=14, ha='center', weight='bold')
    
    chirp = processor.generate_lfm_chirp()
    window_names = ['Rectangular', 'Hamming', 'Blackman', 'Kaiser β=5', 'Taylor', 'Flat-top']
    
    max_delay = 100
    max_doppler = 100
    delays = np.arange(-max_delay, max_delay)
    dopplers = np.linspace(-max_doppler, max_doppler, 201)
    
    for idx, window_name in enumerate(window_names):
        ax = axes[idx // 3, idx % 3]
        
        window_func = benchmarker.window_functions[window_name]
        window = window_func(len(chirp))
        windowed_chirp = chirp * window
        
        ambiguity = np.zeros((len(dopplers), len(delays)), dtype=complex)
        
        for d_idx, doppler in enumerate(dopplers):
            doppler_shift = np.exp(1j * 2 * np.pi * doppler * np.arange(len(chirp)) / processor.params.fs)
            shifted_signal = windowed_chirp * doppler_shift
            
            for t_idx, delay in enumerate(delays):
                if delay >= 0 and delay < len(chirp):
                    ambiguity[d_idx, t_idx] = np.sum(windowed_chirp[delay:] * 
                                                     np.conj(shifted_signal[:len(chirp)-delay]))
        
        ambiguity_mag = np.abs(ambiguity)
        ambiguity_db = 20 * np.log10(ambiguity_mag / np.max(ambiguity_mag) + 1e-10)
        
        im = ax.imshow(ambiguity_db, aspect='auto',
                      extent=[delays[0], delays[-1], dopplers[0], dopplers[-1]],
                      cmap='hot', vmin=-40, vmax=0)
        
        ax.set_xlabel('Time Delay (samples)', fontsize=12)
        ax.set_ylabel('Doppler Shift (Hz)', fontsize=12)
        ax.tick_params(labelsize=11)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Magnitude (dB)', fontsize=11)
        cbar.ax.tick_params(labelsize=10)
    
    plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08, hspace=0.50, wspace=0.30)
    return fig

def generate_window_selection_guide(metrics_results: Dict):
    fig, ax = plt.subplots(figsize=(20, 14))
    fig.suptitle('Window Function Selection Guide for Radar Applications', fontsize=18, y=0.96)
    
    # Add subtitle
    fig.text(0.5, 0.89, 'Higher values indicate better performance for each metric', 
             fontsize=14, ha='center', weight='bold')
    
    window_names = list(metrics_results.keys())
    categories = ['Resolution', 'Sidelobe\nSuppression', 'Processing\nGain', 'ENBW']
    
    data = np.zeros((len(window_names), len(categories)))
    
    for i, window in enumerate(window_names):
        data[i, 0] = 1 / metrics_results[window]['main_lobe_width_mhz']
        data[i, 1] = -metrics_results[window]['peak_sidelobe_db']
        data[i, 2] = metrics_results[window]['processing_gain']
        data[i, 3] = 1 / metrics_results[window]['enbw']
    
    for i in range(len(categories)):
        data[:, i] = (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]))
    
    im = ax.imshow(data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(window_names)))
    ax.set_xticklabels(window_names, rotation=45, ha='right', fontsize=12)
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories, fontsize=14)
    ax.tick_params(labelsize=12)
    
    for i in range(len(window_names)):
        for j in range(len(categories)):
            color = 'white' if data[i, j] < 0.5 else 'black'
            text = ax.text(i, j, f'{data[i, j]:.2f}',
                         ha='center', va='center', color=color, fontsize=11, weight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Performance (0=worst, 1=best)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    plt.subplots_adjust(left=0.10, right=0.92, top=0.85, bottom=0.18)
    return fig

def generate_doppler_resolution_demo(processor: RadarSignalProcessor, benchmarker: WindowFunctionBenchmark):
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle('Doppler Resolution Performance Analysis', fontsize=18, y=0.96)
    
    # Add individual plot titles above graphs and below main title
    fig.text(0.25, 0.88, 'Rectangular Window', fontsize=14, ha='center', weight='bold')
    fig.text(0.75, 0.88, 'Hamming Window', fontsize=14, ha='center', weight='bold')
    fig.text(0.25, 0.43, 'Blackman Window', fontsize=14, ha='center', weight='bold')
    fig.text(0.75, 0.43, 'Taylor Window', fontsize=14, ha='center', weight='bold')
    
    velocity_separation = 25
    targets = [
        {'range': 10000, 'velocity': 0, 'rcs': 10},
        {'range': 10000, 'velocity': velocity_separation, 'rcs': 10}
    ]
    
    window_names = ['Rectangular', 'Hamming', 'Blackman', 'Taylor']
    
    for idx, window_name in enumerate(window_names):
        ax = axes[idx // 2, idx % 2]
        
        data_cube = np.zeros((processor.params.num_pulses, 
                            int(processor.params.fs / processor.params.prf)), dtype=complex)
        
        for target in targets:
            data_cube += processor.generate_target_return(
                target['range'], target['velocity'], target['rcs'])
        
        window_func = benchmarker.window_functions[window_name]
        rd_map = processor.range_doppler_processing(data_cube, window_func, window_func)
        
        range_cut_idx = np.argmax(np.max(rd_map, axis=0))
        doppler_profile = rd_map[:, range_cut_idx]
        doppler_profile_db = 10 * np.log10(doppler_profile + 1e-10)
        
        velocity_axis = np.linspace(-processor.params.prf * 3e8 / (4 * processor.params.fc),
                                   processor.params.prf * 3e8 / (4 * processor.params.fc),
                                   processor.params.doppler_bins)
        
        ax.plot(velocity_axis, doppler_profile_db, 'b-', linewidth=2)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.7, linewidth=2)
        ax.axvline(x=velocity_separation, color='r', linestyle='--', alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Velocity (m/s)', fontsize=12)
        ax.set_ylabel('Power (dB)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-100, 100)
        ax.tick_params(labelsize=11)
        
        peak_val = np.max(doppler_profile_db)
        ax.set_ylim(peak_val - 50, peak_val + 5)
        
        # Add text annotation for target positions
        ax.text(0, peak_val - 5, 'Target 1', ha='center', va='top', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        ax.text(velocity_separation, peak_val - 5, 'Target 2', ha='center', va='top', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.10, hspace=0.50, wspace=0.20)
    return fig

def generate_computational_performance_analysis(benchmarker: WindowFunctionBenchmark, params: RadarParameters):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle('Computational Performance Analysis', fontsize=18, y=0.96)
    
    # Add individual plot titles with proper spacing below main title
    fig.text(0.25, 0.88, 'Window Generation Time vs Size', fontsize=14, ha='center', weight='bold')
    fig.text(0.75, 0.88, 'Pulse Compression Processing Time', fontsize=14, ha='center', weight='bold')
    fig.text(0.25, 0.45, 'Window Function Memory Footprint', fontsize=14, ha='center', weight='bold')
    fig.text(0.70, 0.45, 'System Performance Summary', fontsize=14, ha='center', weight='bold')
    
    sizes = [64, 128, 256, 512, 1024, 2048]
    window_types = ['Rectangular', 'Hamming', 'Blackman', 'Kaiser β=5', 'Taylor', 'Flat-top']
    
    # Plot 1: Window generation time vs size
    ax = axes[0, 0]
    for window_name in window_types:
        window_func = benchmarker.window_functions[window_name]
        times = []
        for size in sizes:
            exec_time = benchmarker.benchmark_execution_time(window_func, size, iterations=50)
            times.append(exec_time)
        ax.semilogy(sizes, times, 'o-', linewidth=2, markersize=8, label=window_name)
    
    ax.set_xlabel('Window Size (samples)', fontsize=12)
    ax.set_ylabel('Execution Time (μs)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    
    # Plot 2: Pulse compression processing time
    ax = axes[0, 1]
    processor = RadarSignalProcessor(params)
    processing_times = []
    
    for window_name in window_types[:4]:
        window_func = benchmarker.window_functions[window_name]
        
        data = processor.generate_target_return(5000, 0, 10)
        
        start_time = time.perf_counter()
        for _ in range(10):
            _ = processor.pulse_compression(data, window_func)
        proc_time = (time.perf_counter() - start_time) / 10 * 1000
        
        processing_times.append(proc_time)
    
    bars = ax.bar(range(len(window_types[:4])), processing_times, color='skyblue', edgecolor='navy')
    ax.set_xticks(range(len(window_types[:4])))
    ax.set_xticklabels(window_types[:4], rotation=0, ha='center', fontsize=11)
    ax.set_ylabel('Processing Time (ms)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=11)
    
    # Add value labels on bars
    for bar, val in zip(bars, processing_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(processing_times)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    # Plot 3: Memory usage
    ax = axes[1, 0]
    memory_usage = []
    for window_name in window_types:
        window_func = benchmarker.window_functions[window_name]
        window = window_func(params.range_bins)
        memory_bytes = window.nbytes
        memory_usage.append(memory_bytes / 1024)
    
    bars = ax.bar(range(len(window_types)), memory_usage, color='lightcoral', edgecolor='darkred')
    ax.set_xticks(range(len(window_types)))
    ax.set_xticklabels(window_types, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Memory Usage (KB)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=11)
    
    # Plot 4: System performance summary (positioned slightly left and up)
    ax = axes[1, 1]
    processor = RadarSignalProcessor(params)
    
    # Position the summary text slightly left and up from the upper left corner
    summary_text = f"""Processor Configuration:
    • Sampling Rate: {params.fs/1e6:.0f} MHz
    • Range Resolution: {processor.range_resolution:.2f} m
    • Velocity Resolution: {processor.velocity_resolution:.2f} m/s
    • Max Unambiguous Range: {processor.max_range/1000:.1f} km
    
    Processing Capabilities:
    • Range Bins: {params.range_bins}
    • Doppler Bins: {params.doppler_bins}
    • Pulses per CPI: {params.num_pulses}
    • SNR: {params.snr_db} dB"""
    
    ax.text(0.02, 0.98, summary_text, ha='left', va='top', 
            fontsize=12, transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    ax.axis('off')
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.12, hspace=0.35, wspace=0.25)
    return fig

if __name__ == "__main__":
    print("Advanced Radar Signal Processing Analysis")
    print("========================================\n")
    
    params = RadarParameters()
    processor = RadarSignalProcessor(params)
    benchmarker = WindowFunctionBenchmark(params)
    
    print("Generating comprehensive analysis outputs...\n")
    
    print("1. Creating window function comparison...")
    fig1 = generate_window_comparison_figure(benchmarker)
    
    print("2. Analyzing window performance metrics...")
    fig2, metrics_results = generate_window_metrics_comparison(benchmarker, params)
    
    print("3. Demonstrating pulse compression performance...")
    fig3 = generate_pulse_compression_demo(processor, benchmarker)
    
    print("4. Generating range-Doppler maps...")
    fig4 = generate_range_doppler_maps(processor, benchmarker)
    
    print("5. Computing ambiguity functions...")
    fig5 = generate_ambiguity_function(processor, benchmarker)
    
    print("6. Creating window selection guide...")
    fig6 = generate_window_selection_guide(metrics_results)
    
    print("7. Analyzing Doppler resolution...")
    fig7 = generate_doppler_resolution_demo(processor, benchmarker)
    
    print("8. Evaluating computational performance...")
    fig8 = generate_computational_performance_analysis(benchmarker, params)
    
    print("\nAll analyses complete. Displaying results...")
    plt.show()
    
    print("\nKey Findings:")
    print("-------------")
    
    best_resolution = min(metrics_results.items(), key=lambda x: x[1]['main_lobe_width_mhz'])
    best_sidelobe = min(metrics_results.items(), key=lambda x: x[1]['peak_sidelobe_db'])
    
    print(f"Best Resolution: {best_resolution[0]} (Main lobe width: {best_resolution[1]['main_lobe_width_mhz']:.2f} MHz)")
    print(f"Best Sidelobe Suppression: {best_sidelobe[0]} (Peak sidelobe: {best_sidelobe[1]['peak_sidelobe_db']:.1f} dB)")
    
    print("\nProcessing complete. Close all plots to exit.")