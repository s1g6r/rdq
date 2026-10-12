"""Per-episode quality scores drawn from the robot-learning curation literature.

The rule-based checks in checks.py catch *broken* episodes (frozen logs, desync,
glitches, schema errors). This module scores *how good* each remaining episode is,
using signals that need no trained policy and no rollouts, only the tables:

  deminf        DemInf-lite: each episode's average contribution to the mutual
                information between states and action chunks, estimated with the
                KSG k-nearest-neighbour estimator (Hejna et al., RSS 2025, without the
                VAE). Higher is better: actions are predictable from states.
  consistency   leave-one-episode-out kNN behaviour-cloning residual, normalised by
                the local action spread (inspired by Belkhale et al., NeurIPS 2023).
                Lower is better. Frames where neighbours split into two action
                clusters are counted as multimodal, so a "minority strategy" is
                reported instead of being called noise.
  hfr, ldlj     smoothness: share of velocity power above a cut-off frequency, and
                log dimensionless jerk. Lower HFR / higher LDLJ is smoother. Reported for
                review only: pauses look "smooth", so HFR can rank hesitant demos as good.
  psd_power     the training-free spectral metric of Sojib et al. (2026) replicated as
                published. By Parseval it is roughly length x mean squared position,
                so it is reported as a baseline, not used in the composite.
  dither        fraction of time spent moving without making progress (windowed path
                efficiency below 0.3), a cheap proxy for hesitation and retries.
  gripper       number of close->open cycles, when a gripper channel can be found.
  final_outlier robust z of how far the final state is from other episodes' finals:
                a failed or unfinished attempt often ends somewhere unusual.
  near_dup      the episode this one nearly duplicates (time-warped copy, re-upload).
  keep_ranges   frame ranges left after trimming idle runs, following the openpi
                DROID recipe (idle runs of 7+ steps removed, segments under 16 dropped).

The composite is the mean rank of deminf and consistency. None of these are verdicts.
They are evidence for ranking and review. Length is
reported separately and the composite's correlation with length is checked,
because many published curation metrics turn out to be length in disguise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import digamma

from .dataset import Dataset

DIGAMMA_KS = (5, 6, 7)


@dataclass
class ScoreConfig:
    chunk: int = 4  # action chunk length for DemInf-lite
    state_dims: int = 12
    action_dims: int = 8
    batch: int = 1024
    passes: int = 4
    knn: int = 16  # neighbours for the consistency score
    max_frames: int = 60_000  # subsample frames above this for the kNN scores
    phase_weight: float = 0.5
    cutoff_hz: float = 2.0
    dither_eta: float = 0.3
    dup_resample: int = 32
    seed: int = 0


@dataclass
class EpisodeScores:
    index: int
    length: int
    idle_frac: float
    deminf: float | None = None
    consistency: float | None = None
    multimodal_frac: float | None = None
    minority_frac: float | None = None
    hfr: float | None = None
    ldlj: float | None = None
    psd_power: float | None = None
    dither: float | None = None
    path_length: float | None = None
    gripper_cycles: int | None = None
    final_outlier: float | None = None
    near_dup_of: int | None = None
    composite: float | None = None
    keep_ranges: list[tuple[int, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- helpers


def _robust_standardise(x: np.ndarray, center=None, scale=None):
    if center is None:
        center = np.nanmedian(x, axis=0)
        q75, q25 = np.nanpercentile(x, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349
        sd = np.nanstd(x, axis=0)
        scale = np.where(scale > 1e-9, scale, np.where(sd > 1e-9, sd, 1.0))
    return (x - center) / scale, center, scale


def _pca(x: np.ndarray, k: int) -> np.ndarray:
    x = x - x.mean(0)
    if x.shape[1] <= k:
        return x
    _, _, vt = np.linalg.svd(x[: min(len(x), 20_000)], full_matrices=False)
    return x @ vt[:k].T


def robust_z(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-12:
        sd = np.nanstd(v)
        mad = sd if sd > 1e-12 else 1.0
    return (v - med) / mad


def _runs(mask: np.ndarray):
    """Yield (start, end_exclusive) of True runs."""
    if not len(mask):
        return
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        yield int(s), int(e)


def idle_mask(action: np.ndarray, scale: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """A frame is idle if no action channel changed by more than tol x its spread."""
    if len(action) < 2:
        return np.zeros(len(action), bool)
    d = np.abs(np.diff(action, axis=0)) / np.maximum(scale, 1e-9)
    m = np.all(d < tol * 1.349, axis=1)  # scale is IQR/1.349, so this is |da| < tol*IQR
    return np.concatenate([[m[0]], m])


def keep_ranges(idle: np.ndarray, fps: float | None) -> list[tuple[int, int]]:
    """openpi's DROID non-idle ranges, with frame counts scaled from 15 Hz to this fps."""
    f = (fps or 15.0) / 15.0
    min_idle, min_keep, trim = max(2, round(7 * f)), max(4, round(16 * f)), round(10 * f)
    drop = np.zeros(len(idle), bool)
    for s, e in _runs(idle):
        if e - s >= min_idle:
            drop[s:e] = True
    out = []
    for s, e in _runs(~drop):
        if e - s >= min_keep:
            end = e - trim if e < len(idle) else e  # trim the tail before an idle stretch
            if end - s >= min_keep:
                out.append((s, end))
    return out


# --------------------------------------------------------------------------- DemInf-lite


def ksg_frame_mi(zs: np.ndarray, za: np.ndarray, batch: int, passes: int, rng: np.random.Generator) -> np.ndarray:
    """Per-frame KSG (Kraskov algorithm 1) mutual-information contributions, max-norm over blocks."""
    n = len(zs)
    total = np.zeros(n)
    counts = np.zeros(n)
    zs = zs + 1e-8 * rng.standard_normal(zs.shape)
    za = za + 1e-8 * rng.standard_normal(za.shape)
    for _ in range(passes):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            idx = order[i : i + batch]
            if len(idx) < max(DIGAMMA_KS) + 2:
                idx = order[max(0, n - batch) :]
            b = len(idx)
            s, a = zs[idx], za[idx]
            ds = np.sqrt(np.maximum(((s[:, None, :] - s[None, :, :]) ** 2).sum(-1), 0))
            da = np.sqrt(np.maximum(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1), 0))
            np.fill_diagonal(ds, np.inf)
            np.fill_diagonal(da, np.inf)
            dj = np.maximum(ds, da)
            srt = np.sort(dj, axis=1)
            acc = np.zeros(b)
            for k in DIGAMMA_KS:
                rho = srt[:, k - 1][:, None]
                ns = (ds <= rho).sum(1)
                na = (da <= rho).sum(1)
                acc += digamma(k) + digamma(b) - digamma(ns + 1) - digamma(na + 1)
            total[idx] += acc / len(DIGAMMA_KS)
            counts[idx] += 1
    out = total / np.maximum(counts, 1)
    lo, hi = np.percentile(out, [1, 99])
    return np.clip(out, lo, hi)


def _same_space(a: np.ndarray, s: np.ndarray) -> bool:
    """Action and state measure the same thing if every channel pair is strongly correlated."""
    if len(a) < 10:
        return False
    for j in range(a.shape[1]):
        x, y = a[:, j], s[:, j]
        if x.std() < 1e-9 or y.std() < 1e-9:
            continue
        if np.corrcoef(x, y)[0, 1] < 0.9:
            return False
    return True


def _chunks(a: np.ndarray, h: int) -> np.ndarray:
    if h <= 1:
        return a
    pad = np.vstack([a, np.repeat(a[-1:], h - 1, axis=0)])
    return np.hstack([pad[i : i + len(a)] for i in range(h)])


# --------------------------------------------------------------------------- consistency


def _two_means(x: np.ndarray, iters: int = 10):
    """Vectorised 2-means over the last-but-one axis. x: (N, k, d). Returns (M, minority_frac)."""
    mean = x.mean(1, keepdims=True)
    ssw1 = ((x - mean) ** 2).sum((1, 2)) + 1e-12
    d0 = ((x - mean) ** 2).sum(2)
    c1 = np.take_along_axis(x, d0.argmax(1)[:, None, None], 1)
    d1 = ((x - c1) ** 2).sum(2)
    c2 = np.take_along_axis(x, d1.argmax(1)[:, None, None], 1)
    for _ in range(iters):
        assign = ((x - c1) ** 2).sum(2) > ((x - c2) ** 2).sum(2)  # True -> cluster 2
        w2 = assign[..., None]
        n2 = w2.sum(1, keepdims=True)
        n1 = x.shape[1] - n2
        c2 = np.where(n2 > 0, (x * w2).sum(1, keepdims=True) / np.maximum(n2, 1), c2)
        c1 = np.where(n1 > 0, (x * ~w2).sum(1, keepdims=True) / np.maximum(n1, 1), c1)
    assign = ((x - c1) ** 2).sum(2) > ((x - c2) ** 2).sum(2)
    ssw2 = np.where(assign, ((x - c2) ** 2).sum(2), ((x - c1) ** 2).sum(2)).sum(1)
    n2 = assign.sum(1)
    minority = np.minimum(n2, x.shape[1] - n2) / x.shape[1]
    return 1 - ssw2 / ssw1, minority


def consistency_scores(zs, za, phase, ep_of, cfg: ScoreConfig, rng):
    from scipy.spatial import cKDTree

    n = len(zs)
    sel = np.arange(n) if n <= cfg.max_frames else np.sort(rng.choice(n, cfg.max_frames, replace=False))
    X = np.hstack([zs[sel], cfg.phase_weight * phase[sel, None] * np.sqrt(zs.shape[1])])
    A = za[sel]
    E = ep_of[sel]
    tree = cKDTree(X)
    k = cfg.knn
    want = min(len(sel) - 1, k * 6)
    _, nbr = tree.query(X, k=want + 1)
    nbr = nbr[:, 1:]
    same = E[nbr] == E[:, None]
    # keep the first k neighbours from other episodes (fall back to whatever exists)
    order = np.argsort(same, axis=1, kind="stable")
    nbr = np.take_along_axis(nbr, order, 1)[:, :k]
    ok = ~np.take_along_axis(same, order, 1)[:, :k]
    NA = A[nbr]  # (N, k, d)
    med = np.median(NA, axis=1)
    r = np.linalg.norm(A - med, axis=1)
    spread = np.median(np.linalg.norm(NA - med[:, None, :], axis=2), axis=1)
    R = r / (spread + 1e-6)
    R[ok.sum(1) < k // 2] = np.nan  # too few foreign neighbours to judge
    M, minority = _two_means(NA)
    multimodal = (M > 0.6) & (minority >= 0.25)
    return sel, R, multimodal


# --------------------------------------------------------------------------- per-episode signals


def smoothness(x: np.ndarray, fps: float, cutoff: float) -> tuple[float, float, float]:
    """(high-frequency ratio of velocity power, log dimensionless jerk, undetrended PSD power)."""
    psd_power = float((np.abs(np.fft.rfft(x, axis=0)) ** 2).sum())
    if len(x) < 8:
        return float("nan"), float("nan"), psd_power
    v = np.diff(x, axis=0) * fps
    v = v - v.mean(0)
    w = np.hanning(len(v))[:, None]
    P = np.abs(np.fft.rfft(v * w, axis=0)) ** 2
    f = np.fft.rfftfreq(len(v), d=1 / fps)
    fc = min(cutoff, 0.4 * fps)
    denom = P[f > 0].sum()
    hfr = float(P[f > fc].sum() / denom) if denom > 1e-12 else 0.0
    vel = np.diff(x, axis=0) * fps
    jerk = np.diff(vel, n=2, axis=0) * fps * fps
    speed = np.linalg.norm(vel, axis=1)
    dur = len(x) / fps
    vpeak = speed.max() if len(speed) else 0
    if vpeak < 1e-9 or not len(jerk):
        return hfr, float("nan"), psd_power
    ldlj = -float(np.log((dur**3 / vpeak**2) * (np.linalg.norm(jerk, axis=1) ** 2).sum() / fps + 1e-12))
    return hfr, ldlj, psd_power


def dither_fraction(x: np.ndarray, fps: float, eta_max: float, moving_tol: float) -> float:
    w = max(2, int(round(2 * fps)))
    if len(x) <= w + 1:
        return 0.0
    step = np.linalg.norm(np.diff(x, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(step)])
    path = cum[w:] - cum[:-w]
    disp = np.linalg.norm(x[w:] - x[:-w], axis=1)
    moving = path > moving_tol * w
    eta = disp / np.maximum(path, 1e-12)
    bad = (eta < eta_max) & moving
    frac = 0
    for s, e in _runs(bad):
        if e - s >= w:
            frac += e - s
    return float(frac / len(x))


def find_gripper(ds: Dataset, key: str) -> int | None:
    names = (ds.features.get(key) or {}).get("names")
    flat = []

    def walk(n):
        if isinstance(n, dict):
            for v in n.values():
                walk(v)
        elif isinstance(n, (list, tuple)):
            for v in n:
                walk(v) if isinstance(v, (list, tuple, dict)) else flat.append(str(v))

    walk(names)
    for i, n in enumerate(flat):
        if "gripper" in n.lower():
            return i
    return None


def gripper_cycles(g: np.ndarray, mid: float, debounce: int = 3) -> int:
    """Full open-close-open cycles of a gripper signal, ignoring flicker shorter than `debounce` frames."""
    b = g > mid
    state, run, changes = bool(b[0]), 0, 0
    for v in b[1:]:
        run = run + 1 if bool(v) != state else 0
        if run >= debounce:
            state, run = bool(v), 0
            changes += 1
    return changes // 2


def episode_embedding(s: np.ndarray, a: np.ndarray, length: int) -> np.ndarray:
    x = np.hstack([s, a])
    t_src = np.linspace(0, 1, len(x))
    t = np.linspace(0, 1, length)
    return np.stack([np.interp(t, t_src, x[:, d]) for d in range(x.shape[1])], 1).ravel()


def near_duplicates(emb: np.ndarray, rng: np.random.Generator, length: int, dims: int) -> tuple[dict[int, int], float]:
    """Map episode position -> position it nearly duplicates, with a calibrated cosine threshold."""
    n = len(emb)
    if n < 3:
        return {}, float("nan")
    mu = emb.mean(0)
    X = emb - mu
    _, _, vt = np.linalg.svd(X[: min(n, 5000)], full_matrices=False)
    P = vt[: min(64, vt.shape[0])]

    def proj(e):
        z = (e - mu) @ P.T
        return z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)

    Z = proj(emb)
    # Calibrate: synthetic near-duplicates (small noise + time warp) of a sample of episodes.
    m = min(n, 64)
    pick = rng.choice(n, m, replace=False)
    T = emb.reshape(n, length, dims)
    copies = []
    for i in pick:
        warp = np.clip(np.linspace(0, 1, length) ** rng.uniform(0.97, 1.03), 0, 1)
        src = np.linspace(0, 1, length)
        c = np.stack([np.interp(warp, src, T[i][:, d]) for d in range(dims)], 1)
        c = c + rng.normal(0, 0.01, c.shape)
        copies.append(c.ravel())
    copies = np.array(copies)
    dup_cos = (proj(copies) * Z[pick]).sum(1)
    dup_rms = np.sqrt(((copies - emb[pick]) ** 2).mean(1))
    rms_thr = float(min(0.05, np.percentile(dup_rms, 99)))  # robust-std units: at most 5% of a channel's IQR
    if n <= 4000:
        C = Z @ Z.T
        np.fill_diagonal(C, -1)
        rand_cos = C[np.triu_indices(n, 1)]
    else:
        a, b = rng.integers(0, n, 200_000), rng.integers(0, n, 200_000)
        rand_cos = (Z[a] * Z[b]).sum(1)[a != b]
        C = None
    tau = float(np.percentile(dup_cos, 1))
    if tau <= np.percentile(rand_cos, 99.9):
        tau = float(max(tau, np.percentile(rand_cos, 99.9)))
    out = {}
    if C is not None:
        for i in range(n):
            j = int(C[i].argmax())
            if C[i, j] >= tau and j < i and np.sqrt(((emb[i] - emb[j]) ** 2).mean()) <= rms_thr:
                out[i] = j
    return out, tau


# --------------------------------------------------------------------------- main entry


def score_episodes(ds: Dataset, cfg: ScoreConfig | None = None) -> dict:
    cfg = cfg or ScoreConfig()
    rng = np.random.default_rng(cfg.seed)
    skey, akey = ds.state_key, ds.action_key
    fps = float(ds.fps or 30.0)
    eps = sorted(ds.episodes)
    if not eps or akey is None:
        return {"episodes": {}, "summary": {"error": "needs an action feature"}}
    pos_key = skey or akey

    # Context for the kNN scores: the state plus every other numeric observation (e.g. object
    # poses or goals in environment_state). Smoothness and path metrics use the state alone.
    ctx_keys = [pos_key] + [k for k in ds.numeric_keys if k.startswith("observation.") and k != pos_key]

    def ctx(ep):
        return np.hstack([ep.features[k] for k in ctx_keys])

    all_a = np.concatenate([ds.episodes[i].features[akey] for i in eps])
    all_s = np.concatenate([ds.episodes[i].features[pos_key] for i in eps])
    all_c = np.concatenate([ctx(ds.episodes[i]) for i in eps])
    finite = np.all(np.isfinite(all_a), 1) & np.all(np.isfinite(all_c), 1)
    _, a_c, a_sc = _robust_standardise(all_a[finite])
    # Position-controlled arms record the commanded next position as the action, so the raw
    # action is almost a copy of the state. What the operator decided is the displacement,
    # a - s, so score that instead when action and state live in the same space.
    relative = all_a.shape[1] == all_s.shape[1] and _same_space(all_a[finite], all_s[finite])
    rel_all = all_a[finite] - all_s[finite] if relative else all_a[finite]
    _, r_c, r_sc = _robust_standardise(rel_all)
    _, s_c, s_sc = _robust_standardise(all_s[finite])
    _, c_c, c_sc = _robust_standardise(all_c[finite])

    out: dict[int, EpisodeScores] = {}
    zs_list, za_list, ph_list, ep_list, fr_list = [], [], [], [], []
    finals, embs = [], []
    g_idx = find_gripper(ds, akey)
    if g_idx is not None and g_idx >= all_a.shape[1]:
        g_idx = None  # names don't match the data's shape; don't guess
    g_mid = None
    if g_idx is not None:
        g_all = all_a[finite][:, g_idx]
        g_mid = float((np.nanpercentile(g_all, 5) + np.nanpercentile(g_all, 95)) / 2)

    for i in eps:
        ep = ds.episodes[i]
        a = ep.features[akey]
        s = ep.features[pos_key]
        c = ctx(ep)
        ok = np.all(np.isfinite(a), 1) & np.all(np.isfinite(c), 1)
        a, s, c = a[ok], s[ok], c[ok]
        es = EpisodeScores(index=i, length=ep.length, idle_frac=0.0)
        out[i] = es
        if len(a) < 4:
            es.notes.append("too short to score")
            continue
        an, _, _ = _robust_standardise(a, a_c, a_sc)
        rn, _, _ = _robust_standardise(a - s if relative else a, r_c, r_sc)
        sn, _, _ = _robust_standardise(s, s_c, s_sc)
        cn, _, _ = _robust_standardise(c, c_c, c_sc)
        idle = idle_mask(a, a_sc)
        es.idle_frac = float(idle.mean())
        es.keep_ranges = keep_ranges(idle, fps)
        es.hfr, es.ldlj, es.psd_power = smoothness(sn, fps, cfg.cutoff_hz)
        es.dither = dither_fraction(sn, fps, cfg.dither_eta, moving_tol=0.01)
        es.path_length = float(np.linalg.norm(np.diff(sn, axis=0), axis=1).sum())
        if g_idx is not None:
            es.gripper_cycles = gripper_cycles(a[:, g_idx], g_mid)
        finals.append((i, sn[-1]))
        embs.append((i, episode_embedding(sn, an, cfg.dup_resample)))
        keep = ~idle
        if keep.sum() < 4:
            continue
        chunk = _chunks(rn, cfg.chunk)[keep]
        zs_list.append(cn[keep])
        za_list.append(chunk)
        ph_list.append((np.arange(len(a)) / max(len(a) - 1, 1))[keep])
        ep_list.append(np.full(keep.sum(), i))

    summary: dict = {"frames_scored": 0, "context_features": ctx_keys, "relative_actions": bool(relative)}
    if zs_list:
        zs = np.concatenate(zs_list)
        za = np.concatenate(za_list)
        phase = np.concatenate(ph_list)
        ep_of = np.concatenate(ep_list)
        zs_p = _pca(zs, cfg.state_dims)
        za_p = _pca(za, cfg.action_dims)
        summary["frames_scored"] = int(len(zs))
        # DemInf-lite on a subsample if huge (cost grows with batch^2 x n/batch)
        sel = np.arange(len(zs)) if len(zs) <= 200_000 else np.sort(rng.choice(len(zs), 200_000, replace=False))
        mi = ksg_frame_mi(zs_p[sel], za_p[sel], cfg.batch, cfg.passes, rng)
        for i in np.unique(ep_of[sel]):
            out[int(i)].deminf = float(mi[ep_of[sel] == i].mean())
        csel, R, multi = consistency_scores(zs_p, za_p, phase, ep_of, cfg, rng)
        E = ep_of[csel]
        for i in np.unique(E):
            m = E == i
            r = R[m]
            r = r[np.isfinite(r)]
            out[int(i)].consistency = float(np.median(r)) if len(r) else None
            out[int(i)].multimodal_frac = float(multi[m].mean())
        summary["multimodal_frame_frac"] = float(multi.mean())

    # final-state outliers
    if len(finals) >= 6:
        F = np.array([f for _, f in finals])
        D = np.sqrt(((F[:, None] - F[None]) ** 2).sum(-1))
        np.fill_diagonal(D, np.inf)
        kd = np.sort(D, axis=1)[:, : min(5, len(F) - 1)].mean(1)
        z = robust_z(np.log(kd + 1e-9))
        for (i, _), zz in zip(finals, z):
            out[i].final_outlier = float(zz)

    # near-duplicates
    if len(embs) >= 3:
        E_ids = [i for i, _ in embs]
        dims = embs[0][1].size // cfg.dup_resample
        dup, tau = near_duplicates(np.array([e for _, e in embs]), rng, cfg.dup_resample, dims)
        for pos, other in dup.items():
            out[E_ids[pos]].near_dup_of = E_ids[other]
        summary["near_dup_threshold"] = tau

    _composite(out, summary)
    _flag(out)
    return {
        "config": cfg.__dict__,
        "summary": summary,
        "episodes": {i: es.__dict__ for i, es in out.items()},
    }


def _rank01(v: np.ndarray, higher_is_better: bool) -> np.ndarray:
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    r = np.full(len(v), np.nan)
    if ok.sum() < 2:
        return r
    order = np.argsort(np.argsort(v[ok] if higher_is_better else -v[ok]))
    r[ok] = order / (ok.sum() - 1)
    return r


def _composite(out: dict[int, EpisodeScores], summary: dict) -> None:
    ids = sorted(out)
    if len(ids) < 3:
        return
    # Smoothness (HFR) and hesitation are reported but left out of the composite: in the
    # curation benchmark HFR ranked paused or failed demos as *better* (AUROC 0.04-0.19 on
    # those types) and dither carried no signal, so averaging them in diluted DemInf-lite and
    # consistency. This choice was made after seeing that benchmark; the RoboMimic MH run
    # (experiments/robomimic_mh.py) is the independent check.
    cols = [
        _rank01([out[i].deminf if out[i].deminf is not None else np.nan for i in ids], True),
        _rank01([out[i].consistency if out[i].consistency is not None else np.nan for i in ids], False),
    ]
    M = np.vstack(cols)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        comp = np.nanmean(M, axis=0)
    for i, c in zip(ids, comp):
        out[i].composite = float(c) if np.isfinite(c) else None
    lengths = np.array([out[i].length for i in ids], float)
    ok = np.isfinite(comp)
    if ok.sum() > 5 and lengths[ok].std() > 0:
        from scipy.stats import spearmanr

        rho = float(spearmanr(comp[ok], lengths[ok]).statistic)
        summary["composite_vs_length_spearman"] = rho
        if abs(rho) > 0.5:
            summary["length_confound_warning"] = (
                f"the composite score is strongly correlated with episode length (Spearman {rho:.2f}); "
                "compare against a keep-the-shortest baseline before trusting it"
            )


def _flag(out: dict[int, EpisodeScores]) -> None:
    ids = sorted(out)
    if len(ids) < 8:
        return

    def z(attr, sign=1):
        v = np.array([getattr(out[i], attr) if getattr(out[i], attr) is not None else np.nan for i in ids], float)
        return sign * robust_z(v)

    zc, zh, zd = z("consistency"), z("hfr"), z("dither")
    paths = np.array([out[i].path_length if out[i].path_length is not None else np.nan for i in ids], float)
    med_path = np.nanmedian(paths) if np.isfinite(paths).any() else np.nan
    cyc = np.array([out[i].gripper_cycles if out[i].gripper_cycles is not None else np.nan for i in ids], float)
    med_cyc = np.nanmedian(cyc) if np.isfinite(cyc).any() else np.nan
    for k, i in enumerate(ids):
        es = out[i]
        if np.isfinite(zc[k]) and zc[k] > 3.5:
            if (es.multimodal_frac or 0) > 0.3:
                es.notes.append("minority strategy: actions differ from most episodes in multimodal regions (review, don't auto-drop)")
            else:
                es.notes.append("inconsistent actions: poorly predicted from states compared with other episodes")
        moves = np.isfinite(paths[k]) and np.isfinite(med_path) and paths[k] >= 0.5 * med_path
        if np.isfinite(zh[k]) and zh[k] > 3.5 and moves:
            es.notes.append("jerky motion: unusually high share of high-frequency velocity")
        if np.isfinite(zd[k]) and zd[k] > 3.5 and (es.dither or 0) > 0.1:
            es.notes.append("hesitation: long stretches of motion without progress")
        if es.final_outlier is not None and es.final_outlier > 3.5:
            es.notes.append("unusual final state: may be a failed or unfinished attempt")
        if es.gripper_cycles is not None and np.isfinite(med_cyc):
            if es.gripper_cycles > med_cyc + 2:
                es.notes.append(f"{es.gripper_cycles} gripper cycles vs median {med_cyc:.0f}: regrasps or retries")
            elif es.gripper_cycles == 0 and med_cyc >= 1:
                es.notes.append("gripper never cycles although most episodes grasp once or more")
        if es.near_dup_of is not None:
            es.notes.append(f"near-duplicate of episode {es.near_dup_of}")
