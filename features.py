"""Time and frequency descriptors adapted from a voltage-trace fault experiment."""

import argparse
import json
from pathlib import Path

import numpy as np


def extract(samples, sample_rate: float) -> dict[str, float]:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("Provide one channel with at least two finite samples")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be finite and positive")

    mean = float(np.mean(x))
    variance = float(np.var(x, ddof=1))
    result = {
        "Mean": mean,
        "Std": variance ** 0.5,
        "Variance": variance,
        "RMS": float(np.sqrt(np.mean(x ** 2))),
        "Energy": float(np.sum(x ** 2)),
        "P2P": float(np.ptp(x)),
        "MAD": float(np.mean(np.abs(x - mean))),
        "ZCR": float(np.mean(np.diff(np.sign(x)) != 0)),
    }
    amplitude = np.abs(np.fft.rfft(x - mean)) / x.size
    # Double paired bins. An even-length signal has a standalone Nyquist bin.
    paired = amplitude[1:-1] if x.size % 2 == 0 else amplitude[1:]
    paired *= 2
    frequency = np.fft.rfftfreq(x.size, d=1 / sample_rate)
    weights = amplitude ** 2
    total = weights.sum()
    entropy = centroid = spread = 0.0
    if total > 0:
        weights /= total
        nonzero = weights[weights > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        centroid = float(np.sum(frequency * weights))
        spread = float(np.sum((frequency - centroid) ** 2 * weights))
    result.update(
        FFT_Peak=float(amplitude.max()),
        MeanFreq=centroid,
        FreqVariance=spread,
        SpectralEntropy=entropy,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Headerless CSV with one voltage per row")
    parser.add_argument("--sample-rate", type=float, help="Sampling frequency in Hz")
    args = parser.parse_args()
    if args.csv is None:
        if args.sample_rate is not None:
            parser.error("--sample-rate is only used with --csv")
        rate = 1000.0
        samples = 2 * np.sin(2 * np.pi * 50 * np.arange(1000) / rate)
    else:
        if args.sample_rate is None:
            parser.error("--csv requires --sample-rate")
        rate = args.sample_rate
        try:
            matrix = np.loadtxt(args.csv, delimiter=",", ndmin=2)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        if matrix.shape[1] != 1:
            parser.error("Expected one voltage per row")
        samples = matrix[:, 0]
    try:
        values = extract(samples, rate)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(values, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
