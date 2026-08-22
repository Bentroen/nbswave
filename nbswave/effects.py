"""Post-render audio effects for exported tracks."""

import numpy as np

EFFECTS_EXTRA = "pip install nbswave[effects]"


def _pedalboard():
    try:
        from pedalboard import Compressor, Limiter
        from pedalboard._pedalboard import Pedalboard

        return Compressor, Limiter, Pedalboard
    except ImportError as e:
        raise ImportError(
            f"This effect requires pedalboard. Install with: {EFFECTS_EXTRA}"
        ) from e


def _pyloudnorm():
    try:
        import pyloudnorm as pyln

        return pyln
    except ImportError as e:
        raise ImportError(
            f"This effect requires pyloudnorm. Install with: {EFFECTS_EXTRA}"
        ) from e


def clip_guard(data: np.ndarray, *, target_db: float = 0.0) -> np.ndarray:
    """Scale down only if the mix exceeds `target_db` (does not boost quiet mixes)."""
    peak = np.abs(data).max()
    target_peak = 10 ** (target_db / 20)
    if peak <= target_peak:
        return data

    print(
        f"The output is clipping by {peak:.2f}x. "
        f"Limiting peak to {target_db:.1f} dBFS"
    )
    return data * (target_peak / peak)


def normalize(data: np.ndarray, *, target_db: float = -1.0) -> np.ndarray:
    """Peak-normalize to the given dBFS level."""
    peak = np.abs(data).max()
    if peak == 0:
        return data.copy()
    target_peak = 10 ** (target_db / 20)
    return data * (target_peak / peak)


def compress(
    data: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -24,
    ratio: float = 12,
    attack_ms: float = 3,
    release_ms: float = 250,
) -> np.ndarray:
    """Soft dynamic range compression (requires pedalboard)."""
    Compressor, _, Pedalboard = _pedalboard()
    return Pedalboard(
        [
            Compressor(
                threshold_db=threshold_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
            )
        ]
    )(data, sample_rate)


def limiter(
    data: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -3,
    release_ms: float = 100,
) -> np.ndarray:
    """Brick-wall peak limiting (requires pedalboard)."""
    _, Limiter, Pedalboard = _pedalboard()

    return Pedalboard([Limiter(threshold_db=threshold_db, release_ms=release_ms)])(
        data, sample_rate
    )


def loudness(
    data: np.ndarray, sample_rate: int, *, target_lufs: float = -14.0
) -> np.ndarray:
    """LUFS loudness normalization (requires pyloudnorm)."""
    pyln = _pyloudnorm()

    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(data)
    if np.isinf(loudness):
        return data.copy()

    return pyln.normalize.loudness(data, loudness, target_lufs)
