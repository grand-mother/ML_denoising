#### FFT transform 
import matplotlib.pyplot as plt
import numpy as np

# def fft(time, clean_signal, noisy_signal, plot = False):

#     dt = time[1] - time[0] # Time difference between consecutive samples in nanoseconds
#     fs = 1. / (dt *1e-9) # Convert time difference from ns to seconds and compute sampling frequency in Hz

#     # Step 3: Compute the FFT of the signal
#     fft_clean = np.fft.fft(clean_signal)
#     fft_noisy = np.fft.fft(noisy_signal)
#     fft_freq = np.fft.fftfreq(n=len(clean_signal), d=dt * 1e-9)  # Convert dt from ns to seconds

#     # Step 4: Convert frequency to MHz and amplitude to mV/MHz
#     fft_freq_mhz = fft_freq / 1e6  # Convert frequency from Hz to MHz
#     fft_amplitude_clean_mvmhz = np.abs(fft_clean)# / len(clean_signal)  # Normalize the amplitude
#     fft_amplitude_noisy_mvmhz = np.abs(fft_noisy)# / len(noisy_signal)  # Normalize the amplitude


#     # Step 5: Plot the FFT result
#     if plot:
#         plt.figure(figsize=(10, 5))
#         plt.plot(fft_freq_mhz, fft_amplitude_clean_mvmhz, label='Clean Signal')
#         plt.plot(fft_freq_mhz, fft_amplitude_noisy_mvmhz, label='Noisy Signal')
#         plt.title('FFT of the Signal')
#         plt.yscale('log')
#         plt.xlabel('Frequency (MHz)')
#         plt.ylabel('Amplitude (mV/MHz)')
#         plt.xlim(0, 1000)  # Only plot up to Nyquist frequency, converted to MHz
#         plt.show()
#     return fft_freq_mhz, fft_amplitude_clean_mvmhz, fft_amplitude_noisy_mvmhz


def psd(trace_array_clean, trace_array_noisy, sampling_frequency= 500):
    # sampling_frquency is in MHz
    # units of the psd are [trace_array]^2/ [sampling_rate]
    N_clean = trace_array_clean.shape[-1]
    N_noisy = trace_array_noisy.shape[-1]
    fft_freq = np.fft.rfftfreq(n=N_clean) * sampling_frequency

    psd_clean = np.abs(np.fft.rfft(trace_array_clean)) ** 2
    psd_noisy = np.abs(np.fft.rfft(trace_array_noisy)) ** 2

    psd_clean[..., 1:-1] *= 2
    psd_noisy[..., 1:-1] *= 2

    psd_clean = psd_clean / N_clean / sampling_frequency
    psd_noisy = psd_noisy / N_noisy / sampling_frequency
    return fft_freq, psd_clean, psd_noisy 

def bandwidth_filter(time, clean_signal, noisy_signal):
    fft_noisy = np.fft.fft(noisy_signal)
    fft_clean = np.fft.fft(clean_signal)
    sampling_rate = time[1] - time[0] # Time difference between consecutive samples in nanoseconds
    fft_freq = np.fft.fftfreq(n=len(clean_signal), d=sampling_rate * 1e-9)  # Convert dt from ns to seconds
    fft_freq_mhz = fft_freq / 1e6
    low_cutoff = 50 # Lower bound of the band-pass filter (45 MHz)
    high_cutoff = 200 # Upper bound of the band-pass filter (210 MHz)
    band_pass_filter = np.logical_and(fft_freq_mhz > low_cutoff, fft_freq_mhz < high_cutoff) | \
                    np.logical_and(fft_freq_mhz< -low_cutoff, fft_freq_mhz> -high_cutoff)
    
    fft_clean[~band_pass_filter] = 0
    fft_noisy[~band_pass_filter] = 0

    ifft_noisy_signal = np.fft.ifft(fft_noisy)
    ifft_clean_signal = np.fft.ifft(fft_clean)
    return np.real(ifft_clean_signal), np.real(ifft_noisy_signal)