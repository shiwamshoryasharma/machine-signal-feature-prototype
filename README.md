# Machine signal feature prototype

An offline Python excerpt from a November 2025 fault-classification experiment. The original feature generator processed voltage traces collected at several velocity settings. This résumé edition isolates the time-domain and FFT calculations so a reader can inspect the numerical conventions and reproduce a result with synthetic data.

## Run the reference signal

From the repository root, using Python 3.10 or newer:

```sh
python -m pip install -r requirements.txt
python features.py
```

The example generates one second of a 50 Hz sine wave with amplitude 2 at 1,000 samples per second. Expect RMS near 1.4142, energy near 2,000, a one-sided FFT peak amplitude near 2, and a spectral centroid near 50 Hz. A single-tone check makes unit and normalization errors easy to spot.

## Inspect a trace

Save one voltage per row in a headerless file named `trace.csv`. Supply the actual sample rate in Hz. For a trace sampled at 1 kHz:

```sh
python features.py --csv trace.csv --sample-rate 1000
```

The calculation assumes uniform sampling. It rejects gaps represented by NaN or infinity because dropping these values would change the time base.

| Output | Convention |
| --- | --- |
| Mean, standard deviation, variance | Sample variance and standard deviation use an N−1 denominator. |
| RMS, energy, peak-to-peak, MAD | Energy is the sum of squared samples. MAD is the mean absolute deviation from the mean. |
| ZCR | Fraction of adjacent samples whose signs differ. A transition to or from exactly zero counts. |
| FFT_Peak | Largest one-sided amplitude after mean removal. Paired bins are doubled. |
| MeanFreq, FreqVariance | Centroid in Hz and variance in Hz², weighted by squared one-sided amplitudes. |
| SpectralEntropy | Shannon entropy in bits over the normalized spectral weights. |

## Prototype scope

The archived experiment also explored wavelet features and model training. This extract retains a subset of the original descriptors, corrects the odd-length FFT endpoint, and replaces spreadsheet and acquisition integrations with local CSV input. Constant signals have zero spectral descriptors after mean removal. No classifier accuracy or hardware fault-detection result is claimed from this example.
