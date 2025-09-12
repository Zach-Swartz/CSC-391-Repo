import matplotlib.pyplot as plt
import numpy as np

# Answers to written questions will be printed after the figures have been generated.

# Global parameters
duration = 2  # seconds
signal_freq = 5.0  # Hz
sampling_freq = 8  # Hz
num_bits = 3  # 3-bit quantization (8 levels)
min_signal = -1
max_signal = 1
mean = 0
std_dev = 0.1  # noise level


def original_signal(t):
    return np.sin(2 * np.pi * signal_freq * t)


def add_Gaussian_noise(signal, mean, std):
    mag = np.max(signal) - np.min(signal)
    noise = np.random.normal(mean, std * mag, len(signal))
    return signal + noise


def quantize(signal, num_bits, min_signal, max_signal):
    num_levels = 2 ** num_bits
    qs = np.round((signal - min_signal) /
                  (max_signal - min_signal) * (num_levels - 1))
    qv = min_signal + qs * (max_signal - min_signal) / (num_levels - 1)
    return qv


def compute_errors(original, noisy):
    mse = np.mean((original - noisy) ** 2)
    rmse = np.sqrt(mse)
    psnr = 10 * np.log10(np.max(np.abs(original)) ** 2 /
                         mse) if mse > 0 else float('inf')
    return mse, rmse, psnr


# plot signals
t_points = np.linspace(0, duration, 1000, endpoint=False)
cont_signal = original_signal(t_points)

plt.figure(figsize=(10, 5))
plt.plot(t_points, cont_signal, label='Continuous Signal')

# sampling
total_samples = int(sampling_freq * duration)
t_sampled = np.linspace(0, duration, total_samples, endpoint=False)
sampled_signal = original_signal(t_sampled)
plt.plot(t_sampled, sampled_signal, 'o', label='Sampled Signal')

# add noise
noisy_sampled_signal = add_Gaussian_noise(sampled_signal, mean, std_dev)
plt.plot(t_sampled, noisy_sampled_signal, 'x', label='Noisy Sampled Signal')

# quantize noisy signal
quantized_noisy = quantize(
    noisy_sampled_signal, num_bits, min_signal, max_signal)
plt.step(t_sampled, quantized_noisy, where='post',
         label=f'Quantized Noisy ({num_bits} bits)', color='r', linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sampling, Quantization, and Noise Analysis')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# print error metrics
mse, rmse, psnr = compute_errors(sampled_signal, noisy_sampled_signal)
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"PSNR: {psnr:.2f} dB")
