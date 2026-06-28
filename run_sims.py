#!/usr/bin/env python3
"""
Local calibrated-simulation runner (no Slurm). Parallelizes across (pop, sim, chrom)
work units with a process pool — each msprime sim is single-threaded, so N workers =
N sims in flight. Idempotent: existing .tsz are skipped; partial writes are atomic.

Usage:
    python run_sims_local.py                      # 16 workers, defaults below
    python run_sims_local.py --workers 8 --n-sims 1000 --pops eur,eas --chroms 1-5
"""
# ── pin per-process math threads BEFORE numpy import so forked workers inherit it ──
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, sys, time, traceback, functools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import h5py, demes, msprime, tszip

# ─────────────────────────────── CONSTANTS ───────────────────────────────
WINDOW_SIZE = 20_000
# Diploid individuals per pop — FIXED to the empirical cohort counts in the data
# (merged_bcf), so simulated 2N haplotypes match the 2N used to compute empirical theta.
POP_SAMPLES = {"afr": 2417, "eur": 1956, "sas": 1144, "mid": 351, "eas": 1228, "amr": 1664}
PLOIDY      = 2
RECOMB_RATE = 1e-8
BASE_SEED   = 42
MU_INIT     = 5e-8        # overshoot rate
MU_RETRY    = 1e-7        # base retry rate for under-target windows
GEN_TIME    = 29
TIME_GRID   = np.geomspace(100.0, 40_000.0, 1000)
ALL_POPS    = ["afr", "eur", "sas", "mid", "eas", "amr"]

# paths (overridable via CLI)
CFG = dict(
    mvn_dir=Path("mvn"), h5_path=Path("mvn/mutation_rate_map_perpop.h5"),
    hardmask=Path("hardmask.hg38.v4.over99.bed"),
    demog_dir=Path("demographies"), sim_dir=Path("/scratch.global/soisa001/sims"),
)

# ─────────────────────────────── helpers ───────────────────────────────
def ne_to_demography(ne, pop):
    ne = np.asarray(ne, float)
    dg = msprime.Demography(); dg.add_population(name=pop, initial_size=float(ne[0]))
    for j in range(1, len(ne)):
        dg.add_population_parameters_change(time=float(TIME_GRID[j]),
                                            initial_size=float(ne[j]), population=pop)
    return dg

def ensure_demographies(pop, n_draws):
    d = np.load(CFG["mvn_dir"] / f"{pop}_mvn.npz")
    mean, cov = np.asarray(d["mean"], float), np.asarray(d["cov"], float)
    out = CFG["demog_dir"] / pop; out.mkdir(parents=True, exist_ok=True)
    written = 0
    for i in range(n_draws):
        p = out / f"demo_{i:05d}.yaml"
        if p.exists():
            continue
        rng = np.random.default_rng([BASE_SEED, i])
        ne = np.exp(rng.multivariate_normal(mean, cov))
        demes.dump(ne_to_demography(ne, pop).to_demes(), str(p))
        written += 1
    return written

@functools.lru_cache(maxsize=64)               # per-worker cache: parse each chrom's BED once
def load_mask(chrom):
    bed = CFG["hardmask"]; iv = []
    if Path(bed).exists():
        with open(bed) as fh:
            for line in fh:
                if not line.strip() or line[0] == "#":
                    continue
                c, s, e = line.split()[:3]
                if c == f"chr{chrom}":
                    iv.append((int(s), int(e)))
    if not iv:
        return np.empty((0, 2), np.int64)
    a = np.array(iv, np.int64); return a[np.argsort(a[:, 0])]

def masked(pos, mask):
    if len(mask) == 0:
        return np.zeros(len(pos), bool)
    s, e = mask[:, 0], mask[:, 1]
    idx = np.searchsorted(s, pos, side="right") - 1
    out = np.zeros(len(pos), bool); ok = idx >= 0
    out[ok] = pos[ok] < e[idx[ok]]; return out

def calibrate_chrom(demography, pop, n_samples, seq_len, theta_target, mask, seed, max_retry_iters=6):
    """Overshoot-and-thin, vectorized, single table copy. Hits each window exactly."""
    n_win = len(theta_target)
    ts = msprime.sim_ancestry(samples={pop: n_samples}, demography=demography,
                              sequence_length=seq_len, recombination_rate=RECOMB_RATE,
                              ploidy=PLOIDY, random_seed=seed)
    mts = msprime.sim_mutations(ts, rate=MU_INIT, model=msprime.BinaryMutationModel(),
                                discrete_genome=True, random_seed=seed)
    pos = mts.tables.sites.position.astype(np.int64)
    is_m = masked(pos, mask); win = np.minimum(pos // WINDOW_SIZE, n_win - 1)
    cand = ~is_m; cw = win[cand]; ci = np.nonzero(cand)[0]
    order = np.argsort(cw, kind="stable"); cw = cw[order]; ci = ci[order]
    starts = np.searchsorted(cw, np.arange(n_win), side="left")
    ends   = np.searchsorted(cw, np.arange(n_win), side="right")
    rng = np.random.default_rng(seed); dm = is_m.copy(); under = {}
    for w in range(n_win):
        h = ends[w] - starts[w]; tgt = int(theta_target[w]); grp = ci[starts[w]:ends[w]]
        if h >= tgt:
            if h > tgt:
                dm[rng.choice(grp, size=h - tgt, replace=False)] = True
        else:
            dm[grp] = True
            if tgt > 0:
                under[w] = tgt
    tb = mts.dump_tables(); tb.delete_sites(np.where(dm)[0])
    if under:
        edges = np.minimum(np.arange(n_win + 1) * WINDOW_SIZE, seq_len).astype(float)
        needed = dict(under); cur = MU_RETRY
        for it in range(max_retry_iters):
            if not needed:
                break
            rate = np.zeros(n_win)
            for w in needed:
                rate[w] = cur
            mts2 = msprime.sim_mutations(ts, rate=msprime.RateMap(position=edges, rate=rate),
                                         model=msprime.BinaryMutationModel(),
                                         discrete_genome=True, random_seed=seed + 1000 + it)
            p2 = mts2.tables.sites.position.astype(np.int64)
            k2 = ~masked(p2, mask); w2 = np.minimum(p2 // WINDOW_SIZE, n_win - 1)
            still = {}
            for w, need in needed.items():
                grp = np.nonzero(k2 & (w2 == w))[0][:need]
                for s in grp:
                    st = mts2.site(int(s))
                    si = tb.sites.add_row(position=st.position, ancestral_state=st.ancestral_state)
                    for m in st.mutations:
                        tb.mutations.add_row(site=si, node=m.node, derived_state=m.derived_state,
                                             parent=-1, time=m.time)
                if len(grp) < need:
                    still[w] = need - len(grp)
            needed = still; cur *= 2.0
    tb.sort(); tb.build_index(); tb.compute_mutation_parents()
    return tb.tree_sequence(), (needed if under else {})

# ─────────────────────────────── worker ───────────────────────────────
def simulate_unit(unit):
    """One (pop, sim_idx, chrom) job. Returns a status dict. Top-level for picklability."""
    pop, sim_idx, chrom = unit
    out = CFG["sim_dir"] / pop / f"sim_{sim_idx:05d}" / f"chr{chrom}.tsz"
    if out.exists():
        return dict(unit=unit, status="skip")
    try:
        demo_path = CFG["demog_dir"] / pop / f"demo_{sim_idx:05d}.yaml"
        if not demo_path.exists():
            return dict(unit=unit, status="no_demog", msg=str(demo_path))
        with h5py.File(CFG["h5_path"], "r") as f:
            g = f[f"chr{chrom}"]
            theta = np.asarray(g[pop]["theta"][...]).astype(np.int64)
            seq_len = int(g["window_ends"][-1])      # source of truth = the theta map
        if int(theta.sum()) == 0:
            return dict(unit=unit, status="zero_theta")
        demography = msprime.Demography.from_demes(demes.load(str(demo_path)))
        n_samples = POP_SAMPLES[pop]
        seed = BASE_SEED + 1_000_000 * (ALL_POPS.index(pop) + 1) + 1000 * sim_idx + chrom
        cal, short = calibrate_chrom(demography, pop, n_samples, seq_len, theta, load_mask(chrom), seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tsz.tmp")
        tszip.compress(cal, str(tmp)); os.replace(tmp, out)   # atomic
        return dict(unit=unit, status="ok", n_sites=int(cal.num_sites),
                    target=int(theta.sum()), under=len(short), n=n_samples)
    except Exception:
        return dict(unit=unit, status="error", msg=traceback.format_exc().splitlines()[-1])

# ─────────────────────────────── driver ───────────────────────────────
def parse_chroms(s):
    out = []
    for tok in s.split(","):
        if "-" in tok:
            a, b = tok.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-sims", type=int, default=10)
    ap.add_argument("--n-draws", type=int, default=1000, help="demographies to ensure per pop")
    ap.add_argument("--pops", type=str, default=",".join(ALL_POPS))
    ap.add_argument("--chroms", type=str, default="1-5")
    ap.add_argument("--mvn-dir", type=Path); ap.add_argument("--h5", type=Path, dest="h5_path")
    ap.add_argument("--hardmask", type=Path); ap.add_argument("--demog-dir", type=Path)
    ap.add_argument("--sim-dir", type=Path)
    a = ap.parse_args()
    for k in ("mvn_dir", "h5_path", "hardmask", "demog_dir", "sim_dir"):
        v = getattr(a, k, None)
        if v is not None:
            CFG[k] = v

    pops = [p for p in a.pops.split(",") if p]
    bad = [p for p in pops if p not in POP_SAMPLES]
    if bad:
        sys.exit(f"unknown pop(s) {bad}; known: {list(POP_SAMPLES)}")
    chroms = parse_chroms(a.chroms)
    print(f"pops={pops} chroms={chroms} n_sims={a.n_sims} workers={a.workers}")
    print(f"sim_dir={CFG['sim_dir']}  h5={CFG['h5_path']}")
    print("  per-pop diploid samples: " + ", ".join(f"{p}={POP_SAMPLES[p]}" for p in pops))
    print(f"  mem guidance: chr1 ~3 GB/worker at these sizes -> ~{3*a.workers} GB if all "
          f"workers hit chr1; set --workers <= RAM_GB/3.5")

    # 1) demographies (serial, cheap, idempotent)
    for pop in pops:
        w = ensure_demographies(pop, max(a.n_draws, a.n_sims))
        print(f"  [demog] {pop}: ensured {max(a.n_draws,a.n_sims)} ({w} newly written)")

    # 2) build work units, drop already-done up front
    units = [(p, i, c) for p in pops for i in range(a.n_sims) for c in chroms]
    pending = [u for u in units
               if not (CFG["sim_dir"] / u[0] / f"sim_{u[1]:05d}" / f"chr{u[2]}.tsz").exists()]
    print(f"  {len(units)} total units, {len(units)-len(pending)} already done, {len(pending)} to run")

    # 3) parallel run
    t0 = time.time(); done = err = 0
    tally = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(simulate_unit, u): u for u in pending}
        for fut in as_completed(futs):
            r = fut.result(); done += 1
            tally[r["status"]] = tally.get(r["status"], 0) + 1
            if r["status"] == "ok":
                p, i, c = r["unit"]
                print(f"  [{done}/{len(pending)}] {p} sim{i} chr{c}: {r['n_sites']:,} sites "
                      f"(target {r['target']:,})" + (f" retry={r['under']}" if r['under'] else ""))
            elif r["status"] == "error":
                err += 1; print(f"  [{done}/{len(pending)}] ERROR {r['unit']}: {r['msg']}")
            elif r["status"] in ("no_demog", "zero_theta"):
                print(f"  [{done}/{len(pending)}] SKIP {r['unit']} ({r['status']} {r.get('msg','')})")
    dt = time.time() - t0
    print(f"\nDONE in {dt:.0f}s  ({dt/max(len(pending),1):.1f}s/unit wall)  tally={tally}")
    if err:
        sys.exit(1)

if __name__ == "__main__":
    main()