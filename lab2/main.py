"""Bird-song frequency separation: STFT, FIR and IIR band-pass filters.

Usage:
    python main.py                     (uses DEFAULT_INPUT)
    python main.py audio/birds.mp3     (any format ffmpeg reads; non-WAV is converted to WAV)

Install dependencies:
    pip install numpy scipy matplotlib

Edit BANDS after inspecting output/00_original_spectrogram.png.
"""
from pathlib import Path
import argparse
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import firwin, butter, lfilter, sosfilt, freqz, sosfreqz, group_delay

# Resolved relative to this file, so it works from any working directory.
DEFAULT_INPUT = Path(__file__).resolve().parent / "audio" / "birds.mp3"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"

# Example frequencies ONLY; update after analyzing your audio spectrogram.
BANDS = {
    "bird_1": (2000, 3000),
    "bird_2": (4000, 5000),
}
FIR_TAPS = 401              # Odd -> symmetric linear-phase band-pass FIR
IIR_ORDER = 6               # Butterworth band-pass prototype order
WINDOW_SIZE = 2048          # STFT samples per frame
HOP_SIZE = 512              # STFT shift between frames


def convert_to_wav(path, out_dir):
    """Convert any ffmpeg-readable file (e.g. MP3) to WAV next to the outputs."""
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg is required to read non-WAV files (brew install ffmpeg)')
    wav_path = Path(out_dir) / (Path(path).stem + '.wav')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(path), str(wav_path)], check=True)
    return wav_path


def read_audio(path):
    fs, audio = wavfile.read(path)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = max(abs(int(info.min)), abs(int(info.max)))
        audio = audio.astype(np.float64) / scale
    else:
        audio = audio.astype(np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if not np.all(np.isfinite(audio)):
        raise ValueError("WAV contains non-finite samples")
    return int(fs), audio


def stft_custom(x, fs, window_size=2048, hop=512):
    """Manual STFT with Hann windows and real-valued FFT."""
    if not (0 < hop <= window_size):
        raise ValueError("Expected 0 < hop <= window_size")
    if len(x) == 0:
        raise ValueError("Empty audio")
    starts = np.arange(0, len(x), hop, dtype=int)
    window = np.hanning(window_size)
    spec = np.empty((window_size // 2 + 1, len(starts)), dtype=np.complex128)
    for j, start in enumerate(starts):
        frame = np.zeros(window_size, dtype=np.float64)
        chunk = x[start:start + window_size]
        frame[:len(chunk)] = chunk
        spec[:, j] = np.fft.rfft(frame * window)
    freqs = np.fft.rfftfreq(window_size, d=1 / fs)
    times = (starts + window_size / 2) / fs
    return freqs, times, spec


def plot_spectrogram(x, fs, filename, title):
    freqs, times, z = stft_custom(x, fs, WINDOW_SIZE, HOP_SIZE)
    # Relative dB reference shared with the original for comparable plots.
    power_db = 20 * np.log10(np.maximum(np.abs(z), 1e-12))
    fig, ax = plt.subplots(figsize=(12, 5))
    m = ax.pcolormesh(times, freqs / 1000, power_db, shading='auto', cmap='magma', vmin=-85, vmax=45)
    ax.set(ylim=(0, min(10, fs / 2000)), xlabel='Time, s', ylabel='Frequency, kHz', title=title)
    fig.colorbar(m, ax=ax, label='STFT magnitude, dB (fixed reference)')
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def plot_responses(fir_b, iir_sos, fs, filename, band_name):
    f, hf = freqz(fir_b, worN=4096, fs=fs)
    _, hi = sosfreqz(iir_sos, worN=4096, fs=fs)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for h, label in [(hf, 'FIR'), (hi, 'IIR')]:
        axes[0].plot(f / 1000, 20 * np.log10(np.maximum(np.abs(h), 1e-10)), label=label)
        # Unwrap phase separately; undefined phase in deep stopbands is not meaningful.
        axes[1].plot(f / 1000, np.unwrap(np.angle(h)), label=label)
    # FIR exactly linear phase except around deep zero crossings.
    _, gd_fir = group_delay((fir_b, [1.0]), w=4096, fs=fs)
    # IIR group delay computed from unwrapped phase derivative.
    phase_iir = np.unwrap(np.angle(hi))
    omega = 2 * np.pi * f / fs
    gd_iir = -np.gradient(phase_iir, omega)
    axes[2].plot(f / 1000, gd_fir / fs * 1000, label='FIR')
    axes[2].plot(f / 1000, gd_iir / fs * 1000, label='IIR')
    axes[0].set(ylabel='Amplitude, dB', ylim=(-100, 5))
    axes[1].set(ylabel='Unwrapped phase, rad')
    axes[2].set(ylabel='Group delay, ms', xlabel='Frequency, kHz')
    for ax in axes:
        ax.set_xlim(0, min(fs / 2000, 10))
        ax.grid(alpha=.2)
        ax.legend(loc='best')
    fig.suptitle(f'Frequency response: {band_name}')
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def wav_export(path, fs, audio):
    # Use float32 WAV to preserve relative amplitude without individual normalization.
    wavfile.write(path, fs, audio.astype(np.float32))


def band_energy(x, fs, low, high):
    """Approximate band energy (Parseval scaling) from real FFT."""
    if len(x) == 0:
        return 0.0
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    power = np.abs(np.fft.rfft(x)) ** 2
    weights = np.full(len(power), 2.0)
    weights[0] = 1.0
    if len(x) % 2 == 0:
        weights[-1] = 1.0
    mask = (freqs >= low) & (freqs <= high)
    return float(np.sum(power[mask] * weights[mask]) / len(x))


def main():
    parser = argparse.ArgumentParser(description='FIR/IIR bird song separation')
    parser.add_argument('wav', type=Path, nargs='?', default=DEFAULT_INPUT,
                        help=f'Input audio file (WAV, MP3, ...); default: {DEFAULT_INPUT}')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT, help='Output directory')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.wav.suffix.lower() != '.wav':
        args.wav = convert_to_wav(args.wav, args.out)
        print(f'Converted to {args.wav}')
    fs, x = read_audio(args.wav)
    print(f'Loaded {args.wav}: {fs} Hz, {len(x) / fs:.2f} sec, mono, {len(x)} samples')
    if not len(x):
        raise ValueError('Audio is empty')
    plot_spectrogram(x, fs, args.out / '00_original_spectrogram.png', 'Original recording')

    for name, (low, high) in BANDS.items():
        if not (0 < low < high < fs / 2):
            print(f'Skipping {name}: [{low}, {high}] Hz invalid for Nyquist {fs/2:g} Hz')
            continue
        fir_b = firwin(FIR_TAPS, [low, high], fs=fs, pass_zero=False, window='hamming')
        iir_sos = butter(IIR_ORDER, [low, high], btype='bandpass', fs=fs, output='sos')
        y_fir = lfilter(fir_b, [1.0], x)
        y_iir = sosfilt(iir_sos, x)
        wav_export(args.out / f'{name}_FIR.wav', fs, y_fir)
        wav_export(args.out / f'{name}_IIR.wav', fs, y_iir)
        plot_spectrogram(y_fir, fs, args.out / f'{name}_FIR_spectrogram.png', f'{name}: FIR')
        plot_spectrogram(y_iir, fs, args.out / f'{name}_IIR_spectrogram.png', f'{name}: IIR')
        plot_responses(fir_b, iir_sos, fs, args.out / f'{name}_filters.png', name)
        unwanted = (min(fs / 2 - 1, high + 1200), min(fs / 2, high + 2200))
        if unwanted[0] < unwanted[1]:
            e0 = band_energy(x, fs, *unwanted)
            for method, y in [('FIR', y_fir), ('IIR', y_iir)]:
                e1 = band_energy(y, fs, *unwanted)
                if e0 < 1e-12:
                    print(f'{name} {method}: no measurable input energy in unwanted band; attenuation N/A')
                else:
                    attenuation = 10 * np.log10(e0 / max(e1, 1e-24))
                    print(f'{name} {method}: unwanted {unwanted[0]:.0f}–{unwanted[1]:.0f} Hz attenuation {attenuation:.1f} dB')
        print(f'{name}: FIR nominal delay {(FIR_TAPS - 1) / 2 / fs * 1000:.2f} ms; IIR delay varies with frequency')

    print(f'Finished. See {args.out.resolve()}')


if __name__ == '__main__':
    main()
