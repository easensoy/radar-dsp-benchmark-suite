# RadarScope

**Comprehensive signal processing analysis toolkit for radar systems with optimised window function evaluation and performance benchmarking.**

## Overview

RadarScope addresses the fundamental challenge of selecting optimal window functions for pulse compression and range-doppler processing in radar applications. The toolkit enables systematic evaluation of window function performance across multiple metrics including range resolution, sidelobe suppression, and computational efficiency.

## Features

- **13 Window Functions**: Rectangular, Hanning, Hamming, Blackman, Blackman-Harris, Kaiser (multiple β), Taylor, Tukey, Flat-top, and Nuttall
- **Performance Benchmarking**: Real-time execution time measurement across different window sizes
- **Pulse Compression Analysis**: Complete radar processing chain with synthetic target generation
- **Range-Doppler Processing**: 2D performance maps with configurable target parameters
- **Ambiguity Function Visualization**: Joint range-doppler uncertainty analysis
- **Quantitative Metrics**: Main lobe width, sidelobe suppression, processing gain, equivalent noise bandwidth

## Architecture

### Core Classes

**`RadarParameters`**: Dataclass defining radar system operating characteristics
- Configurable for different operational scenarios
- Standard radar parameter alignment

**`WindowFunctionBenchmark`**: Benchmarking engine with 13 optimised window functions
- Computational performance measurement
- Real-time execution analysis for embedded systems

**`RadarSignalProcessor`**: Core processing chain implementation
- Windowed matched filter application
- FFT-based correlation processing
- Range-doppler map generation

## Performance Analysis

### Key Findings

| Window Function | Main Lobe Width | Sidelobe Suppression | Processing Gain | ENBW Ratio |
|---|---|---|---|---|
| Rectangular | 0.1 MHz | -17.9 dB | 256 | 1.0× |
| Hamming | 0.2 MHz | -42.7 dB | 198 | 1.36× |
| Blackman | 0.3 MHz | -58.1 dB | 151 | 1.73× |
| Blackman-Harris | 0.3 MHz | -93.1 dB | 126 | 2.00× |
| Kaiser (β=3) | 0.2 MHz | -39.2 dB | 210 | 1.24× |
| Kaiser (β=7) | 0.3 MHz | -63.4 dB | 165 | 1.57× |
| Taylor | 0.3 MHz | -35.0 dB | 180 | 53.0× |

### Resolution vs. Interference Rejection Trade-offs

- **Rectangular**: Optimal resolution (0.1 MHz main lobe), poor sidelobe rejection (-17.9 dB)
- **Blackman-Harris**: Maximum sidelobe suppression (-93.1 dB), 200% wider main lobe
- **Kaiser**: Parametric control via β adjustment for balanced performance
- **Taylor**: Precise sidelobe shaping with computational overhead

## Analysis Outputs

The toolkit generates comprehensive visualisations:

1. **Time/Frequency Domain Analysis**: Window characteristics and spectral properties
2. **Quantitative Metrics Comparison**: Performance across 13 window functions
3. **Pulse Compression Performance**: Range profiles with sidelobe analysis
4. **Range-Doppler Maps**: 2D processing results with multi-target scenarios
5. **Ambiguity Functions**: Joint range-doppler resolution characteristics
6. **Selection Decision Matrix**: Normalised performance comparison guide
7. **Doppler Resolution Analysis**: Velocity discrimination capability
8. **Computational Performance**: Processing time and memory footprint analysis

## System Specifications

- **Sampling Rate**: 10 MHz
- **Range Resolution**: 1.0 meter
- **Maximum Unambiguous Range**: 150 km
- **Processing Time**: 107-112 ms for realistic scenarios
- **Memory Footprint**: 2KB coefficient storage per window function

## Window Function Selection Guide

**For Resolution-Critical Applications**:
- Rectangular (maximum resolution)
- Kaiser β=3 (balanced performance)

**For Interference-Limited Environments**:
- Blackman-Harris (maximum sidelobe suppression)
- Blackman (good suppression with acceptable resolution)

**For General Purpose**:
- Hamming (optimal compromise)
- Kaiser β=5 (parametric flexibility)

**For Specialised Applications**:
- Taylor (precise sidelobe control)
- Flat-top (ultra-low sidelobes)

## Installation

```bash
# Clone the repository
git clone https://github.com/username/radarscope.git
cd radarscope

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Analysis

Execute the complete RadarScope analysis suite:

```python
from radarscope import RadarParameters, RadarSignalProcessor, WindowFunctionBenchmark
import matplotlib.pyplot as plt

# Initialize system parameters
params = RadarParameters(
    pulse_width=1e-6,          # 1 microsecond pulse
    bandwidth=10e6,            # 10 MHz bandwidth  
    sampling_rate=20e6,        # 20 MHz sampling
    num_samples=1024,          # Sample count
    max_range=150e3            # 150 km max range
)

# Create analysis components
processor = RadarSignalProcessor(params)
benchmark = WindowFunctionBenchmark()

# Generate all eight analysis outputs
processor.generate_all_analysis()
```

### Individual Analysis Components

**Window Function Comparison:**
```python
# Benchmark all window functions
results = benchmark.compare_all_windows(window_size=256)
benchmark.plot_performance_metrics(results)
```

**Pulse Compression Analysis:**
```python
# Analyse pulse compression performance  
processor.pulse_compression_analysis(['rectangular', 'hamming', 'blackman'])
```

**Range-Doppler Processing:**
```python
# Generate range-doppler maps
targets = [(5000, 100), (8000, -50), (12000, 200), (15000, 0)]  # (range, velocity)
processor.range_doppler_analysis(targets, window_combinations=['rect-rect', 'blackman-blackman'])
```

**Ambiguity Function Analysis:**
```python
# Generate ambiguity functions
processor.ambiguity_function_analysis(['rectangular', 'kaiser_5', 'flat_top'])
```

## Analysis Outputs

### 1. Window Function Time and Frequency Domain Analysis

<img width="1919" height="1133" alt="Screenshot 2025-08-06 003058" src="https://github.com/user-attachments/assets/0c17faa8-fa54-4635-a70b-ef6ee5910524" />


**Description**: Comprehensive analysis of eight primary window functions showing time domain characteristics and frequency spectra. Demonstrates fundamental tradeoffs between main lobe width and sidelobe suppression.

**Key Insights**:
- Rectangular windowing: narrowest main lobe (-13.0 dB peak sidelobe)
- Blackman-Harris: maximum sidelobe suppression (-92.0 dB) with wider main lobe
- Kaiser family: parametric control via β adjustment (β=3: -39.2 dB, β=7: -63.4 dB)
- Enhanced sidelobe suppression always reduces frequency resolution

### 2. Quantitative Performance Metrics Comparison

<img width="1919" height="1088" alt="Screenshot 2025-08-06 003146" src="https://github.com/user-attachments/assets/7fcf4848-ca02-4094-bdf0-254c2f16dd88" />


**Description**: Quantitative comparison of main lobe width, sidelobe suppression, processing gain, and equivalent noise bandwidth across thirteen window functions.

**Key Insights**:
- Main lobe width: Rectangular (0.1 MHz) to Taylor (0.3 MHz) - 200% range resolution impact
- Sidelobe suppression: Blackman-Harris achieves 75 dB improvement over rectangular
- Processing gain: 3.1 dB loss with advanced windowing (256 to 126)
- ENBW impact: Taylor shows 53× higher noise bandwidth than rectangular

### 3. Pulse Compression Performance Analysis

<img width="1916" height="1089" alt="Screenshot 2025-08-06 003201" src="https://github.com/user-attachments/assets/48ff703d-92bb-49ea-9bef-52087c2314fd" />


**Description**: Range profile comparisons showing compressed pulse responses for six different window functions with 3dB and 13dB beamwidth references.

**Key Insights**:
- Rectangular: narrowest pulse width but high sidelobe structure above -20 dB
- Advanced windows (Blackman, Taylor): dramatic sidelobe suppression below -30 dB
- Nuttall: extreme sidelobe suppression near noise floor with increased main lobe width
- Enables weak target detection near strong reflectors

### 4. Range-Doppler Processing Results

<img width="1915" height="1132" alt="Screenshot 2025-08-06 003215" src="https://github.com/user-attachments/assets/090d6fe3-a44f-4dc6-b89c-46fc1ae45929" />


**Description**: Two-dimensional range-doppler processing results using six different window function combinations. Shows target detection performance across range and velocity dimensions with four targets at (5,8,12,15 km) and velocities (100,-50,200,0 m/s).

**Key Insights**:
- Rectangular: sharpest responses but cross-range sidelobe contamination
- Advanced combinations (Blackman-Blackman, Taylor-Taylor): superior target separation
- Mixed approach (Blackman-Taylor): optimised asymmetric complexity allocation
- Clear target detection despite proximity in range-doppler space

### 5. Ambiguity Function Visualisation

<img width="1914" height="1091" alt="Screenshot 2025-08-06 003227" src="https://github.com/user-attachments/assets/769f902b-5224-4fef-ae13-d4f832f3dc5f" />


**Description**: Range-doppler ambiguity functions showing fundamental resolution characteristics and uncertainty relationships for six different window functions with 0 to -40 dB magnitude scale.

**Key Insights**:
- Rectangular: thumbtack pattern with optimal resolution but extensive ridge structure
- Advanced windows: dramatically reduced sidelobe structures with increased main lobe extent
- Kaiser β=5: balanced performance in delay-doppler space
- Flat-top: near-ideal sidelobe suppression with main lobe broadening

### 6. Window Selection Decision Matrix

<img width="1913" height="1092" alt="Screenshot 2025-08-06 003239" src="https://github.com/user-attachments/assets/242bf6f8-1d14-4bc8-9e49-e7bcae737310" />


**Description**: Normalised performance comparison matrix (0-1 scale) enabling systematic window function selection based on operational priorities.

**Key Insights**:
- Rectangular: perfect resolution/processing gain (1.00), zero sidelobe suppression
- Blackman-Harris: maximum sidelobe suppression (1.00), reduced resolution (0.16)  
- Kaiser β=3: balanced performance across all metrics (>0.28 in all categories)
- Taylor: specialised solution with poor resolution (0.13) but moderate sidelobe control

### 7. Doppler Resolution Capability Analysis

<img width="1919" height="1090" alt="Screenshot 2025-08-06 003327" src="https://github.com/user-attachments/assets/dd663797-fdca-4ab4-a957-f112c9051570" />


**Description**: Doppler profile analysis showing resolution of closely spaced velocity targets separated by 25 m/s at positions 0 m/s and 25 m/s.

**Key Insights**:
- Rectangular: sharpest velocity resolution with significant sidelobe structure (>30 dB)
- Advanced windows (Blackman, Taylor): superior sidelobe suppression (<20 dB) with broader main lobes
- Hamming: optimal compromise between resolution and sidelobe control (<25 dB)
- Resolution vs. interference rejection tradeoff clearly demonstrated

### 8. Computational Performance Assessment

<img width="1919" height="1089" alt="Screenshot 2025-08-06 003401" src="https://github.com/user-attachments/assets/645f3eb7-0646-4454-8a74-6837be718306" />


**Description**: Processing time analysis, memory footprint comparison, and system performance summary showing execution times from sub-microsecond (rectangular) to >100 microseconds (Taylor) with system specifications (10 MHz sampling, 1.0m resolution, 150km range).

**Key Insights**:
- Execution time range: rectangular (sub-μs) to Taylor (100+ μs)
- Processing complexity: advanced functions require 10-100× more time
- Pulse compression: uniform 107-112 ms processing across all functions
- Memory footprint: consistent 2KB coefficient storage
- Real-time constraints manageable for most applications

## System Specifications

- **Sampling Rate**: 10 MHz
- **Range Resolution**: 1.0 meter
- **Maximum Unambiguous Range**: 150 km
- **Processing Time**: 107-112 ms for realistic scenarios
- **Memory Footprint**: 2KB coefficient storage per window function

## Window Function Selection Guide

**For Resolution-Critical Applications**:
- Rectangular (maximum resolution)
- Kaiser β=3 (balanced performance)

**For Interference-Limited Environments**:
- Blackman-Harris (maximum sidelobe suppression)
- Blackman (good suppression with acceptable resolution)

**For General Purpose**:
- Hamming (optimal compromise)
- Kaiser β=5 (parametric flexibility)

**For Specialised Applications**:
- Taylor (precise sidelobe control)
- Flat-top (ultra-low sidelobes)

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Pandas (for analysis)

## Citation

If you use RadarScope in your research, please cite:

```bibtex
@software{radarscope,
  title={RadarScope: Signal Processing Analysis Toolkit for Radar Systems},
  author={Your Name},
  year={2024},
  url={https://github.com/username/radarscope}
}
```
