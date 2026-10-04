"""Part A (Track 2): classical schedulers vs. learned selector, with a workload shift.

Run:  python src/run_partA.py          (all outputs -> results/partA/)
Everything is seeded, so the numbers are exactly reproducible.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from schedulers import SCHEDULERS, NAMES, MAX_CYL
from workloads import gen_dataset, gen_shift_timeline, TYPES
from features import window_features, FEATURES
from labels import evaluate_window
from selector import SchedulerSelector, calibration_stats
from explain import explain

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "partA")
os.makedirs(OUT, exist_ok=True)
SEED_TRAIN, SEED_TEST = 1, 2
SHIFT_SEEDS = list(range(100, 110))            # 10 independent shift timelines
PHASES = (("sequential", 50), ("random", 50), ("bursty", 50))
N_TRAIN, N_TEST = 1500, 500                     # windows per workload type


def build(rows):
    X, opt, seeks = [], [], []
    for kind, reqs, h, d in rows:
        out, _, o, _ = evaluate_window(reqs, h, d)
        X.append(window_features(reqs, h, d)); opt.append(o)
        seeks.append({k: v[0] for k, v in out.items()})
    return np.array(X), opt, seeks


def main():
    # ---------------- data ----------------
    train_rows = gen_dataset(N_TRAIN, SEED_TRAIN)
    test_rows = gen_dataset(N_TEST, SEED_TEST)
    Xtr, Otr, _ = build(train_rows)
    Xte, Ote, Ste = build(test_rows)
    kinds_tr = np.array([r[0] for r in train_rows])
    kinds_te = np.array([r[0] for r in test_rows])

    selectors = {
        "Selector-DT": SchedulerSelector("tree").fit(Xtr, Otr),
        "Selector-LR": SchedulerSelector("logreg").fit(Xtr, Otr),
        # trained ONLY on the pre-shift (sequential) workload -> tests learned-model drift
        "Selector-DT-seqonly": SchedulerSelector("tree").fit(
            Xtr[kinds_tr == "sequential"], [o for o, k in zip(Otr, kinds_tr) if k == "sequential"]),
    }

    # label distribution (for the report)
    lab_counts = {s: int(sum(s in o for o in Otr)) for s in NAMES}

    # ---------------- 1. static test set, per workload type ----------------
    rec = []
    for i, (kind, reqs, h, d) in enumerate(test_rows):
        best = min(Ste[i].values())
        row = {"kind": kind, "Oracle": best}
        for s in NAMES:
            row[s] = Ste[i][s]; row[f"{s}_opt"] = s in Ote[i]
        for name, sel in selectors.items():
            pick, conf, _ = sel.select(Xte[i])
            row[name] = Ste[i][pick]; row[f"{name}_pick"] = pick
            row[f"{name}_conf"] = conf; row[f"{name}_opt"] = pick in Ote[i]
        f = dict(zip(FEATURES, Xte[i]))
        _, econf, pattern = explain(f, row["Selector-DT_pick"])
        row["explainer_conf"] = econf; row["explainer_pattern"] = pattern
        rec.append(row)
    te = pd.DataFrame(rec)
    te.to_csv(os.path.join(OUT, "test_windows.csv"), index=False)

    policies = NAMES + list(selectors) + ["Oracle"]
    by_type = []
    for kind in TYPES + ["all"]:
        sub = te if kind == "all" else te[te.kind == kind]
        for p in policies:
            opt_col = f"{p}_opt"
            by_type.append({"workload": kind, "policy": p,
                            "mean_seek": sub[p].mean(),
                            "mean_regret": (sub[p] - sub["Oracle"]).mean(),
                            "optimal_rate": sub[opt_col].mean() if opt_col in sub else 1.0})
    by_type = pd.DataFrame(by_type)
    by_type.to_csv(os.path.join(OUT, "test_by_type.csv"), index=False)

    # ---------------- 2. calibration ----------------
    calib, rel_tables = [], {}
    for name in list(selectors) + ["explainer"]:
        conf_col = "explainer_conf" if name == "explainer" else f"{name}_conf"
        corr_col = "Selector-DT_opt" if name == "explainer" else f"{name}_opt"
        for kind in TYPES + ["all"]:
            sub = te if kind == "all" else te[te.kind == kind]
            st, table = calibration_stats(sub[conf_col], sub[corr_col])
            calib.append({"model": name, "workload": kind, **st})
            if kind == "all":
                rel_tables[name] = table
    calib = pd.DataFrame(calib)
    calib.to_csv(os.path.join(OUT, "calibration.csv"), index=False)

    # ---------------- 3. shift timeline (head state carried across windows) ----------------
    def run_timeline(seed):
        timeline = gen_shift_timeline(seed, PHASES)
        state = {p: (MAX_CYL // 2, 1) for p in NAMES + list(selectors) + ["Oracle"]}
        trows, expl_rows = [], []
        for t, (kind, reqs) in enumerate(timeline):
            row = {"t": t, "kind": kind}
            for p in state:
                h, d = state[p]
                out, _, o, best = evaluate_window(reqs, h, d)
                if p in NAMES:
                    pick, conf = p, np.nan
                elif p == "Oracle":   # per-window best, given its own head position
                    pick, conf = min(NAMES, key=lambda s: out[s][0]), np.nan
                else:
                    f = window_features(reqs, h, d)
                    pick, conf, _ = selectors[p].select(f)
                    if p == "Selector-DT":
                        text, econf, pattern = explain(dict(zip(FEATURES, f)), pick)
                        expl_rows.append({"t": t, "kind": kind, "pick": pick, "model_conf": conf,
                                          "explainer_conf": econf, "pattern_seen": pattern,
                                          "pick_optimal": pick in o, "explanation": text})
                seek, nh, nd = out[pick]
                row[p] = seek; row[f"{p}_regret"] = seek - best; row[f"{p}_pick"] = pick
                row[f"{p}_conf"] = conf; row[f"{p}_opt"] = pick in o
                state[p] = (nh, nd)
            trows.append(row)
        return pd.DataFrame(trows), pd.DataFrame(expl_rows)

    phase_rows, regret_curves = [], []
    for k, seed in enumerate(SHIFT_SEEDS):
        tl_s, ex_s = run_timeline(seed)
        regret_curves.append(tl_s[[c for c in tl_s.columns if c.endswith("_regret")]].cumsum())
        if k == 0:                       # first seed: saved in full + used for Fig. 2
            tl = tl_s
            tl.to_csv(os.path.join(OUT, "shift_timeline_seed%d.csv" % seed), index=False)
            ex_s.to_csv(os.path.join(OUT, "bonus_explanations.csv"), index=False)
        for p in NAMES + list(selectors) + ["Oracle"]:
            for kind in [ph for ph, _ in PHASES] + ["whole timeline"]:
                sub = tl_s if kind == "whole timeline" else tl_s[tl_s.kind == kind]
                phase_rows.append({"seed": seed, "policy": p, "phase": kind,
                                   "total_seek": int(sub[p].sum()),
                                   "total_regret": int(sub[f"{p}_regret"].sum()),
                                   "optimal_rate": sub[f"{p}_opt"].mean()})
    phase_all = pd.DataFrame(phase_rows)
    phase_all.to_csv(os.path.join(OUT, "shift_phase_all_seeds.csv"), index=False)
    phase = (phase_all.groupby(["policy", "phase"])
             .agg(seek_mean=("total_seek", "mean"), seek_std=("total_seek", "std"),
                  regret_mean=("total_regret", "mean"), regret_std=("total_regret", "std"),
                  optimal_rate=("optimal_rate", "mean")).reset_index())
    phase.to_csv(os.path.join(OUT, "shift_phase_summary.csv"), index=False)

    # ---------------- figures ----------------
    plt.rcParams.update({"font.size": 9})
    C = {"FCFS": "tab:gray", "SSTF": "tab:blue", "SCAN": "tab:orange", "CSCAN": "tab:green",
         "Selector-DT": "tab:red", "Selector-LR": "tab:purple", "Selector-DT-seqonly": "tab:brown",
         "Oracle": "black"}
    show = ["FCFS", "SSTF", "SCAN", "CSCAN", "Selector-DT", "Selector-LR", "Oracle"]
    # Fig 1: mean seek per workload type
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    w = 0.11
    for j, p in enumerate(show):
        vals = [by_type[(by_type.workload == k) & (by_type.policy == p)].mean_seek.iloc[0] for k in TYPES]
        ax.bar(np.arange(3) + (j - 3) * w, vals, w, label=p, color=C[p])
    ax.set_xticks(range(3)); ax.set_xticklabels(TYPES); ax.set_yscale("log")
    ax.set_ylabel("Mean seek per window (cyl.)"); ax.legend(ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_seek_by_type.png"), dpi=200); plt.close(fig)

    # Fig 2: shift timeline (rolling seek + cumulative regret)
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True)
    for p in ["FCFS", "SCAN", "CSCAN", "SSTF", "Selector-DT", "Oracle"]:
        axs[0].plot(tl.t, tl[p].rolling(5, min_periods=1).mean(), label=p, lw=1.1,
                    ls=":" if p == "Oracle" else "-", color=C[p])
    axs[0].set_yscale("log")
    mean_curve = sum(regret_curves) / len(regret_curves)
    for p in ["SSTF", "Selector-DT", "Selector-LR", "Selector-DT-seqonly"]:
        axs[1].plot(tl.t, mean_curve[f"{p}_regret"], label=p, lw=1.2, color=C[p])
    axs[1].legend(fontsize=7, frameon=False)
    for ax in axs:
        for x in (50, 100):
            ax.axvline(x - 0.5, color="k", ls="--", lw=0.8)
    axs[0].set_ylabel("Seek / window\n(5-window mean, seed 100)")
    axs[1].set_ylabel("Cumulative regret\n(mean of 10 seeds)")
    axs[1].set_xlabel("Window index  (sequential | random | bursty)")
    axs[0].legend(ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_shift_timeline.png"), dpi=200); plt.close(fig)

    # Fig 3: reliability diagram
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    for name, mk, col in [("Selector-DT", "o", C["Selector-DT"]), ("Selector-LR", "s", C["Selector-LR"]), ("explainer", "^", "tab:green")]:
        tb = rel_tables[name]
        ax.plot([r[3] for r in tb], [r[4] for r in tb], marker=mk, lw=1, ms=4, color=col,
                label=name if name != "explainer" else "Explainer (bonus)")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Fraction of picks that were optimal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_reliability.png"), dpi=200); plt.close(fig)

    # ---------------- console + json summary ----------------
    summary = {"label_optimal_counts_train": lab_counts,
               "n_train_windows": int(len(Xtr)), "n_test_windows": int(len(Xte)),
               "shift_points": [50, 100], "timeline_windows": int(len(tl)), "shift_seeds": SHIFT_SEEDS}
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    pd.set_option("display.width", 160)
    print(by_type.pivot(index="policy", columns="workload", values="mean_seek").round(1))
    print(by_type[by_type.workload == "all"].round(3).to_string(index=False))
    print(calib[calib.workload == "all"].round(3).to_string(index=False))
    print(phase.pivot(index="policy", columns="phase", values="seek_mean").round(0))
    print(phase.pivot(index="policy", columns="phase", values="seek_std").round(0))
    print(phase.pivot(index="policy", columns="phase", values="regret_mean").round(0))
    print(phase.pivot(index="policy", columns="phase", values="optimal_rate").round(3))
    print(summary)


if __name__ == "__main__":
    main()
