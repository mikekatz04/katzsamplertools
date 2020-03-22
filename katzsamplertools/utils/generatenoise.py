import numpy as np


def generate_noise_frequencies(Tobs, fs):
    df = 1.0 / Tobs
    number_of_samples = int(np.round(Tobs * fs))
    number_of_frequencies = int(np.round(number_of_samples / 2) + 1)

    noise_freqs = np.linspace(
        start=0, stop=sampling_frequency / 2, num=number_of_frequencies
    )

    return noise_freqs


def generate_noise_single_channel(noise_func, noise_args, noise_kwargs, df, data_freqs):

    norm1 = 0.5 * (1.0 / df) ** 0.5
    re = np.random.normal(0, norm1, data_freqs.shape)
    im = np.random.normal(0, norm1, data_freqs.shape)
    htilde = re + 1j * im

    return np.sqrt(noise_func(data_freqs, *noise_args, **noise_kwargs)) * htilde
