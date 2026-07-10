#!/usr/bin/env python3
"""
Sanity plots for the completed calibrated simulations (read-only; separate from the
runner). Points at the default sim outputs and the combined per-pop mutation-rate map.

For the first N sims of each pop it computes, per 20 kb window and genome-wide
(chr1..22), the realized seg-sites/window (theta_hat), Tajima's D, and nucleotide
diversity pi, plus a folded site-frequency spectrum, and writes into --out-dir:
    {pop}_genomewide.png   3-panel: S/window (vs target theta), Tajima's D, pi
    {pop}_sfs.png          folded SFS vs neutral 1/i reference
    summary_by_pop.png     cross-pop Tajima's D + mean S/window vs target
    summary.tsv            per-pop numeric summary (incl. sample-size shortfall)

Sample sizes (--size v9, default): analyses use the v9 per-pop sizes (V9_SAMPLES).
Each sim is downsampled AT READ TIME by keeping the first 2*n haplotypes, i.e. dropping
the later/higher-index individuals ("drop the later sample IDs"). Where v9 EXCEEDS the
size a sim was generated with we cannot upsample: all available samples are used and the
shortfall is reported (summary.tsv + a stderr warning). Use --size sim to instead use
each sim's full sample set (useful to verify the theta calibration, realized ~ target).

Usage:
    python plot_sim_sanity.py                        # first 5 sims/pop, chr1-22, defaults
    python plot_sim_sanity.py --n-sims 5 --workers 4 --pops afr,eur --chroms 1-5
"""
import os, argparse, sys, time, traceback
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, h5py, tszip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WINDOW_SIZE = 20_000
ALL_POPS = ["afr", "eur", "sas", "mid", "eas", "amr"]
# v9 per-pop DIPLOID sample sizes to use in ALL downstream analyses (OTH not simulated).
V9_SAMPLES  = {"afr": 2641, "eur": 1858, "amr": 2070, "eas": 1312, "sas": 1196, "mid": 416}
# sims are (being) regenerated at the v9 sizes, so simulated == v9 (kept for shortfall logic).
SIM_SAMPLES = dict(V9_SAMPLES)

CFG = dict(sim_dir=Path("/scratch.global/soisa001/sims"),
           h5_path=Path("mvn/mutation_rate_map_perpop_all.h5"),
           out_dir=Path("sim_sanity_plots"))

# ─────────────────────────────── worker ───────────────────────────────
def _target_haps(pop, size_mode):
    return 2 * (SIM_SAMPLES[pop] if size_mode == "sim" else V9_SAMPLES[pop])

def analyze_unit(job):
    """One (pop, sim, chrom): windowed S / Tajima's D / pi + folded SFS at the chosen
    sample size. Downsamples by keeping the first 2*n haplotypes (drop later samples)."""
    pop, sim_idx, chrom, size_mode = job
    path = CFG["sim_dir"] / pop / f"sim_{sim_idx:05d}" / f"chr{chrom}.tsz"
    if not path.exists():
        return dict(pop=pop, sim=sim_idx, chrom=chrom, status="missing", msg=str(path))
    try:
        ts = tszip.decompress(str(path))
        samp = ts.samples()
        avail = int(samp.size)
        want = _target_haps(pop, size_mode)
        keep = samp[:min(want, avail)]                    # first 2n haps = drop later samples
        with h5py.File(CFG["h5_path"], "r") as f:
            we = np.asarray(f[f"chr{chrom}"]["window_ends"][...], float)
        seq_len = float(ts.sequence_length)
        edges = np.concatenate([[0.0], we])
        edges[-1] = min(edges[-1], seq_len)               # guard tiny float overrun
        # one sample set -> tskit returns shape (n_windows, 1); ravel to 1D per window
        S  = np.asarray(ts.segregating_sites(sample_sets=[keep], windows=edges, mode="site", span_normalise=False)).ravel()
        D  = np.asarray(ts.Tajimas_D(sample_sets=[keep], windows=edges, mode="site")).ravel()
        pi = np.asarray(ts.diversity(sample_sets=[keep], windows=edges, mode="site", span_normalise=True)).ravel()
        afs = np.asarray(ts.allele_frequency_spectrum(sample_sets=[keep], mode="site",
                                                      polarised=False, span_normalise=False)).ravel()
        return dict(pop=pop, sim=sim_idx, chrom=chrom, status="ok",
                    S=S.astype(np.float64), D=D.astype(np.float64),
                    pi=pi.astype(np.float64), afs=afs.astype(np.float64),
                    n_used=int(keep.size), n_avail=avail, want=int(want),
                    shortfall=int(max(0, want - avail)))
    except Exception:
        return dict(pop=pop, sim=sim_idx, chrom=chrom, status="error",
                    msg=traceback.format_exc().splitlines()[-1])

# ─────────────────────────────── helpers ───────────────────────────────
def parse_chroms(s):
    out = []
    for tok in s.split(","):
        if "-" in tok:
            a, b = tok.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out

def bin_mean(idx, v, n_bins):
    """nan-aware mean of v grouped into n_bins by precomputed bin index idx."""
    v = np.asarray(v, float); ok = np.isfinite(v)
    s = np.bincount(idx[ok], weights=v[ok], minlength=n_bins)
    c = np.bincount(idx[ok], minlength=n_bins)
    out = np.full(n_bins, np.nan); nz = c > 0; out[nz] = s[nz] / c[nz]
    return out

def load_genome_axis(chrom_order, pops):
    """From the h5: per-chrom window structure + target theta, and the genome-wide axis
    (window midpoints with chromosome offsets, chrom boundaries, per-pop target arrays)."""
    meta = {}          # chrom -> dict(we, nwin, seq_len, slice)
    mids = []          # genome-coordinate window midpoints
    offset = 0.0; start = 0; boundaries = [0.0]; labels = []
    with h5py.File(CFG["h5_path"], "r") as f:
        avail_chroms = {int(k[3:]) for k in f if k.startswith("chr") and k[3:].isdigit()}
        chrom_order = [c for c in chrom_order if c in avail_chroms]
        for c in chrom_order:
            we = np.asarray(f[f"chr{c}"]["window_ends"][...], float)
            nwin = len(we); seq_len = float(we[-1])
            edges = np.concatenate([[0.0], we])
            mids.append((edges[:-1] + edges[1:]) / 2.0 + offset)
            meta[c] = dict(we=we, nwin=nwin, seq_len=seq_len, slice=slice(start, start + nwin))
            start += nwin; offset += seq_len
            boundaries.append(offset); labels.append(f"chr{c}")
        genome_mid = np.concatenate(mids) if mids else np.zeros(0)
        n_win_genome = start
        target = {}    # pop -> genome-wide target theta (seg sites/window at full sim size)
        for pop in pops:
            t = np.full(n_win_genome, np.nan)
            for c in chrom_order:
                t[meta[c]["slice"]] = np.asarray(f[f"chr{c}"][pop]["theta"][...], float)
            target[pop] = t
    return chrom_order, meta, genome_mid, n_win_genome, np.array(boundaries), labels, target

def _draw_chrom_guides(ax, boundaries, labels, ymax_axis):
    for b in boundaries:
        ax.axvline(b / 1e6, color="grey", lw=0.4, alpha=0.5)
    mids = (boundaries[:-1] + boundaries[1:]) / 2.0
    for m, lab in zip(mids, labels):
        ax.text(m / 1e6, ymax_axis, lab.replace("chr", ""), ha="center", va="bottom",
                fontsize=6, color="grey", clip_on=False)

def plot_pop_genomewide(pop, idx, centers, boundaries, labels, Sg, Dg, Pg, target,
                        n_used, n_avail, want, out_dir, size_mode):
    n_sims = Sg.shape[0]
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    xc = centers / 1e6
    # panel 0: seg sites / window (realized) vs target theta
    for si in range(n_sims):
        axes[0].plot(xc, bin_mean(idx, Sg[si], len(centers)), lw=0.4, alpha=0.30, color="C0")
    axes[0].plot(xc, bin_mean(idx, np.nanmean(Sg, axis=0), len(centers)),
                 lw=1.3, color="C0", label=f"realized (mean of {n_sims} sims)")
    axes[0].plot(xc, bin_mean(idx, target, len(centers)),
                 lw=1.0, ls="--", color="k", label="target theta (full-size calibration)")
    axes[0].set_ylabel("seg sites / 20 kb"); axes[0].legend(loc="upper right", fontsize=8)
    # panel 1: Tajima's D
    for si in range(n_sims):
        axes[1].plot(xc, bin_mean(idx, Dg[si], len(centers)), lw=0.4, alpha=0.30, color="C1")
    axes[1].plot(xc, bin_mean(idx, np.nanmean(Dg, axis=0), len(centers)), lw=1.3, color="C1")
    axes[1].axhline(0.0, color="k", lw=0.6, ls=":")
    axes[1].set_ylabel("Tajima's D")
    # panel 2: pi
    for si in range(n_sims):
        axes[2].plot(xc, bin_mean(idx, Pg[si], len(centers)), lw=0.4, alpha=0.30, color="C2")
    axes[2].plot(xc, bin_mean(idx, np.nanmean(Pg, axis=0), len(centers)), lw=1.3, color="C2")
    axes[2].set_ylabel(r"$\pi$ / bp"); axes[2].set_xlabel("genome position (Mb, chr1..22 concatenated)")
    for ax in axes:
        _draw_chrom_guides(ax, boundaries, labels, ax.get_ylim()[1])
    short = "" if want <= n_avail else f"  ⚠ v9 wants {want//2} dip but sim has {n_avail//2} (using all)"
    fig.suptitle(f"{pop.upper()}  genome-wide sanity  (size={size_mode}: "
                 f"{n_used//2} diploids / {n_used} haplotypes){short}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    p = out_dir / f"{pop}_genomewide.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p

def plot_pop_sfs(pop, afs, n_used, out_dir):
    if afs is None or afs.size < 3:
        return None
    n = afs.size - 1                                   # number of haplotypes
    k = np.arange(1, n // 2 + 1)                        # folded minor-allele-count bins
    obs = afs[1:n // 2 + 1]
    ref = 1.0 / k; ref = ref / ref.sum() * np.nansum(obs)   # neutral 1/i, scaled to obs total
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(k, obs, ".", ms=3, label="observed (summed over sims/chroms)")
    ax.loglog(k, ref, "-", lw=1, color="k", alpha=0.7, label=r"neutral $\propto 1/i$")
    ax.set_xlabel("minor allele count"); ax.set_ylabel("number of sites")
    ax.set_title(f"{pop.upper()} folded SFS  ({n_used} haplotypes)")
    ax.legend(fontsize=8); fig.tight_layout()
    p = out_dir / f"{pop}_sfs.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p

def plot_summary(rows, Dpool, out_dir):
    pops = [r["pop"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # left: genome-wide Tajima's D distribution per pop
    data = [Dpool[p] for p in pops]
    axes[0].boxplot(data, tick_labels=[p.upper() for p in pops], showfliers=False)
    axes[0].axhline(0.0, color="k", lw=0.6, ls=":")
    axes[0].set_ylabel("Tajima's D (per-window, all sims)"); axes[0].set_title("Tajima's D by pop")
    # right: mean seg sites / window realized vs target
    x = np.arange(len(pops)); w = 0.38
    axes[1].bar(x - w / 2, [r["mean_S_realized"] for r in rows], w, label="realized", color="C0")
    axes[1].bar(x + w / 2, [r["mean_theta_target"] for r in rows], w, label="target", color="grey")
    axes[1].set_xticks(x); axes[1].set_xticklabels([p.upper() for p in pops])
    axes[1].set_ylabel("mean seg sites / 20 kb"); axes[1].set_title("realized vs target theta")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "summary_by_pop.png"; fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ─────────────────────────────── driver ───────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=5, help="first N sims per pop (sim_00000..)")
    ap.add_argument("--pops", type=str, default=",".join(ALL_POPS))
    ap.add_argument("--chroms", type=str, default="1-22")
    ap.add_argument("--size", choices=["v9", "sim"], default="v9",
                    help="v9: downsample to v9 sizes (drop later samples); sim: full sim size")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sim-dir", type=Path); ap.add_argument("--h5", type=Path, dest="h5_path")
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args()
    for k in ("sim_dir", "h5_path", "out_dir"):
        v = getattr(a, k, None)
        if v is not None:
            CFG[k] = v

    pops = [p for p in a.pops.split(",") if p]
    bad = [p for p in pops if p not in V9_SAMPLES]
    if bad:
        sys.exit(f"unknown pop(s) {bad}; known: {list(V9_SAMPLES)}")
    sims = list(range(a.n_sims))
    chrom_order = sorted(parse_chroms(a.chroms))
    CFG["out_dir"].mkdir(parents=True, exist_ok=True)
    print(f"pops={pops} sims=first {a.n_sims} chroms={chrom_order} size={a.size} workers={a.workers}")
    print(f"sim_dir={CFG['sim_dir']}  h5={CFG['h5_path']}  out={CFG['out_dir']}")

    # sample-size reconciliation up front (loud about the v9 > sim shortfall)
    print("  sample sizes (diploid) — v9 target vs simulated:")
    for p in pops:
        v9, sm = V9_SAMPLES[p], SIM_SAMPLES[p]
        note = "ok (drop %d)" % (sm - v9) if sm >= v9 else "SHORTFALL: v9 exceeds sim by %d (using all %d)" % (v9 - sm, sm)
        print(f"    {p}: v9={v9} sim={sm} -> {note}")

    chrom_order, meta, genome_mid, n_win_genome, boundaries, labels, target = \
        load_genome_axis(chrom_order, pops)
    if n_win_genome == 0:
        sys.exit("no chromosomes found in h5 for the requested --chroms")
    # display binning: up to ~2500 genome bins, but never more than we have windows
    # (else sparse data scatters into mostly-empty bins and the lines break on nan)
    n_bins = int(min(2500, max(1, n_win_genome)))
    lo, hi = genome_mid.min(), genome_mid.max()
    idx = np.clip(((genome_mid - lo) / (hi - lo + 1e-9) * n_bins).astype(int), 0, n_bins - 1)
    centers = (np.arange(n_bins) + 0.5) / n_bins * (hi - lo) + lo

    # run the per-unit analysis in parallel
    jobs = [(p, s, c, a.size) for p in pops for s in sims for c in chrom_order]
    print(f"  {len(jobs)} (pop,sim,chrom) units to analyze", flush=True)
    results = {}; t0 = time.time(); done = 0; miss = err = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(analyze_unit, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result(); done += 1
            results[(r["pop"], r["sim"], r["chrom"])] = r
            if r["status"] == "missing":
                miss += 1; print(f"  [{done}/{len(jobs)}] MISSING {r['msg']}", flush=True)
            elif r["status"] == "error":
                err += 1; print(f"  [{done}/{len(jobs)}] ERROR ({r['pop']},{r['sim']},{r['chrom']}): {r['msg']}", flush=True)
            elif done % 25 == 0 or done == len(jobs):
                print(f"  … {done}/{len(jobs)} analyzed in {time.time()-t0:.0f}s", flush=True)

    # assemble genome-wide arrays per pop and plot
    rows = []; Dpool = {}
    for pop in pops:
        Sg = np.full((len(sims), n_win_genome), np.nan)
        Dg = np.full_like(Sg, np.nan); Pg = np.full_like(Sg, np.nan)
        afs_pop = None; n_used = n_avail = want = 0
        for si, sim in enumerate(sims):
            for c in chrom_order:
                r = results.get((pop, sim, c))
                if not r or r["status"] != "ok":
                    continue
                sl = meta[c]["slice"]
                Sg[si, sl] = r["S"]; Dg[si, sl] = r["D"]; Pg[si, sl] = r["pi"]
                n_used, n_avail, want = r["n_used"], r["n_avail"], r["want"]
                if afs_pop is None:
                    afs_pop = np.zeros_like(r["afs"])
                if r["afs"].shape == afs_pop.shape:
                    afs_pop += r["afs"]
        if n_used == 0:
            print(f"  [warn] {pop}: no usable units, skipping plots", flush=True)
            continue
        p1 = plot_pop_genomewide(pop, idx, centers, boundaries, labels, Sg, Dg, Pg,
                                 target[pop], n_used, n_avail, want, CFG["out_dir"], a.size)
        p2 = plot_pop_sfs(pop, afs_pop, n_used, CFG["out_dir"])
        Dpool[pop] = Dg[np.isfinite(Dg)]
        rows.append(dict(pop=pop, n_used_hap=n_used, n_used_dip=n_used // 2,
                         v9_dip=V9_SAMPLES[pop], sim_dip=SIM_SAMPLES[pop],
                         shortfall_dip=max(0, V9_SAMPLES[pop] - SIM_SAMPLES[pop]),
                         mean_S_realized=float(np.nanmean(Sg)),
                         mean_theta_target=float(np.nanmean(target[pop])),
                         mean_TajD=float(np.nanmean(Dg)), mean_pi=float(np.nanmean(Pg))))
        print(f"  [{pop}] wrote {p1.name}" + (f", {p2.name}" if p2 else ""), flush=True)

    if rows:
        ps = plot_summary(rows, Dpool, CFG["out_dir"])
        tsv = CFG["out_dir"] / "summary.tsv"
        cols = ["pop", "n_used_hap", "n_used_dip", "v9_dip", "sim_dip", "shortfall_dip",
                "mean_S_realized", "mean_theta_target", "mean_TajD", "mean_pi"]
        with open(tsv, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])
                                   for c in cols) + "\n")
        print(f"\nDONE in {time.time()-t0:.0f}s. wrote {ps.name}, summary.tsv, and per-pop PNGs "
              f"to {CFG['out_dir']}  (missing={miss} error={err})")
        shortfalls = [r["pop"] for r in rows if r["shortfall_dip"] > 0]
        if a.size == "v9" and shortfalls:
            print(f"  ⚠ v9 sample size exceeds the simulated size for: {', '.join(shortfalls)} "
                  f"— plots for these use ALL simulated samples (see summary.tsv shortfall_dip).")
    else:
        print("\nno pops produced plots (all units missing?) — check --sim-dir")

if __name__ == "__main__":
    main()
