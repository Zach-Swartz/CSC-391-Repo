import numpy as np
import matplotlib.pyplot as plt

# Answers to written questions will be printed after the figures have been generated.

# Global parameters
duration = 2  # seconds
signal_freq = 5.0  # Hz
sampling_freq = 8  # Hz
num_bits = 3  # 3-bit quantization (8 levels)
min_signal = -1  # min signal value
max_signal = 1  # max signal value

# Original signal


def original_signal(t):
    return np.sin(2 * np.pi * signal_freq * t)


t_points = np.linspace(0, duration, 1000, endpoint=False)
cont_signal = original_signal(t_points)

plt.figure(figsize=(10, 5))
plt.plot(t_points, cont_signal, label='Continuous Signal')

# Sampling
total_samples = int(sampling_freq * duration)
t_sampled = np.linspace(0, duration, total_samples, endpoint=False)
sampled_signal = original_signal(t_sampled)
plt.plot(t_sampled, sampled_signal, 'o', label='Sampled Signal')

# Quantization
num_levels = 2 ** num_bits
qs = np.round((sampled_signal - min_signal) /
              (max_signal - min_signal) * (num_levels - 1))
qv = min_signal + qs * (max_signal - min_signal) / (num_levels - 1)
plt.step(t_sampled, qv, where='post',
         label=f'Quantized Signal ({num_bits} bits)', color='r', linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sampling and Quantization of a Sinusoidal Signal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("A sampling frequency should be at least twice the signal frequency (Nyquist-Shannon theorem) to capture the true shape of the signal.")
print("To minimize error, increase the sampling frequency and the number of quantization bits.")
