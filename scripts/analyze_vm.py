"""Part B analysis: parse raw logs in results/vm/ -> CSV tables + figures.

Usage (from repo root):  python scripts/analyze_vm.py
Works with whatever configurations / fio files are present.
"""
from __future__ import annotations
import glob, json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
VM = os.path.join(ROOT, "results", "vm")
STATUS = {0: "OK", 124: "TIMEOUT", 137: "KILLED"}


def parse_time(path):
    txt = open(path).read()
    def grab(label, cast=float):
        m = re.search(re.escape(label) + r".*?:\s*(.+)", txt)
        return cast(m.group(1).strip()) if m else np.nan
    el = grab("Elapsed (wall clock) time (h:mm:ss or m:ss)", str)
    secs = np.nan
    if isinstance(el, str):
        parts = [float(p) for p in el.split(":")]
        secs = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    return {"elapsed_s": secs,
            "max_rss_mib": grab("Maximum resident set size (kbytes)") / 1024,
            "major_faults": grab("Major (requiring I/O) page faults"),
            "minor_faults": grab("Minor (reclaiming a frame) page faults"),
            "vol_ctx_switches": grab("Voluntary context switches"),
            "invol_ctx_switches": grab("Involuntary context switches"),
            "cpu_percent": float(str(grab("Percent of CPU this job got", str)).rstrip("%") or "nan")}


def parse_procvmstat(before, after):
    rd = lambda p: {l.split()[0]: int(l.split()[1]) for l in open(p) if l.strip()}
    b, a = rd(before), rd(after)
    return {f"sys_{k}": a.get(k, 0) - b.get(k, 0) for k in ("pswpin", "pswpout", "pgmajfault")}


def parse_vmstat(path):
    lines = [l.split() for l in open(path) if l.strip()]
    hdr = next((l for l in lines if "si" in l and "so" in l), None)
    if hdr is None:
        return {}
    rows = [l for l in lines if l[0].isdigit() and len(l) == len(hdr)][1:]  # drop since-boot row
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=hdr).astype(float)
    out = {"vmstat_free_min_mib": df["free"].min() / 1024,
           "vmstat_si_total_kib": df["si"].sum(), "vmstat_so_total_kib": df["so"].sum(),
           "vmstat_wa_mean": df["wa"].mean(), "vmstat_us_mean": df["us"].mean(),
           "vmstat_sy_mean": df["sy"].mean()}
    if "st" in df:
        out["vmstat_st_mean"] = df["st"].mean()
    return out


def parse_sysinfo(path):
    txt = open(path).read()
    m_n = re.search(r"== nproc\n(\d+)", txt)
    m_m = re.search(r"Mem:\s+(\d+)", txt)
    return {"vcpus": int(m_n.group(1)) if m_n else np.nan,
            "ram_mib": int(m_m.group(1)) if m_m else np.nan}


def analyze_configs():
    runs = []
    for cfg_dir in sorted(glob.glob(os.path.join(VM, "C*"))):
        cfg = os.path.basename(cfg_dir)
        si = parse_sysinfo(os.path.join(cfg_dir, "sysinfo.txt")) if os.path.exists(
            os.path.join(cfg_dir, "sysinfo.txt")) else {}
        for tpath in sorted(glob.glob(os.path.join(cfg_dir, "time_r*.txt"))):
            r = re.search(r"time_r(\d+)", tpath).group(1)
            row = {"config": cfg, "rep": int(r), **si, **parse_time(tpath)}
            ec_p = os.path.join(cfg_dir, f"exitcode_r{r}.txt")
            ec = int(open(ec_p).read().strip()) if os.path.exists(ec_p) else -1
            row["exit_code"], row["status"] = ec, STATUS.get(ec, f"ERROR {ec}")
            bp, ap = (os.path.join(cfg_dir, f"procvmstat_{w}_r{r}.txt") for w in ("before", "after"))
            if os.path.exists(bp) and os.path.exists(ap):
                row.update(parse_procvmstat(bp, ap))
            vp = os.path.join(cfg_dir, f"vmstat_r{r}.log")
            if os.path.exists(vp):
                row.update(parse_vmstat(vp))
            wp = os.path.join(cfg_dir, f"workload_r{r}.json")
            if ec == 0 and os.path.exists(wp) and os.path.getsize(wp) > 0:
                w = json.loads(open(wp).read().strip().splitlines()[-1])
                row.update({"mem_phase_s": w["memory_phase"]["total_s"],
                            "mem_fill_s": w["memory_phase"]["fill_s"],
                            "mem_seq_s": w["memory_phase"]["seq_passes_s"],
                            "mem_rand_s": w["memory_phase"]["random_pass_s"],
                            "cpu_phase_s": w["cpu_phase"]["total_s"],
                            "workers": w["cpu_phase"]["workers"], "checksum": w["checksum"]})
            runs.append(row)
    if not runs:
        print("No configuration results found in results/vm/C*/")
        return None
    df = pd.DataFrame(runs)
    df.to_csv(os.path.join(VM, "vm_runs.csv"), index=False)
    if "checksum" in df and df["checksum"].notna().any():
        n_unique = df["checksum"].dropna().nunique()
        print(f"Workload checksum identical across all successful runs: {n_unique == 1}")
    num = df.select_dtypes("number").columns.drop(["rep", "exit_code"], errors="ignore")
    ok = df[df.status == "OK"]
    summ = ok.groupby("config")[list(num)].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    cfgs = sorted(df.config.unique())
    summ = summ.reindex(cfgs)
    summ["runs_ok"] = ok.groupby("config").size().reindex(cfgs).fillna(0).astype(int)
    summ["runs_total"] = df.groupby("config").size().reindex(cfgs)
    summ["statuses"] = df.groupby("config").status.agg(lambda x: ",".join(x)).reindex(cfgs)
    for c in ("ram_mib", "vcpus"):                  # known even when every run failed
        if c in df:
            summ[f"{c}_mean"] = df.groupby("config")[c].first().reindex(cfgs)
    summ = summ.reset_index()
    summ.to_csv(os.path.join(VM, "vm_summary.csv"), index=False)
    cols = ["config", "ram_mib_mean", "vcpus_mean", "runs_ok", "runs_total", "statuses", "elapsed_s_mean",
            "elapsed_s_std", "mem_phase_s_mean", "cpu_phase_s_mean", "max_rss_mib_mean",
            "major_faults_mean", "sys_pswpin_mean", "sys_pswpout_mean", "vmstat_wa_mean_mean",
            "vmstat_st_mean_mean", "invol_ctx_switches_mean"]
    print(summ[[c for c in cols if c in summ]].round(2).to_string(index=False))
    plot_configs(summ)
    return summ


def plot_configs(s):
    plt.rcParams.update({"font.size": 9})
    labels = [f"{c}\n{r/1024:.1f} GB/{int(v)} vCPU" if not np.isnan(r) else c
              for c, r, v in zip(s.config, s.get("ram_mib_mean", np.nan), s.get("vcpus_mean", np.nan))]
    x = np.arange(len(s))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for k, (col, lab) in enumerate([("mem_phase_s", "Memory phase"), ("cpu_phase_s", "CPU phase")]):
        if f"{col}_mean" in s:
            axs[0].bar(x + (k - 0.5) * 0.38, s[f"{col}_mean"], 0.38, yerr=s.get(f"{col}_std"),
                       capsize=2, label=lab)
    axs[0].set_xticks(x); axs[0].set_xticklabels(labels, fontsize=7)
    axs[0].set_ylabel("Time (s), mean ± sd"); axs[0].legend(frameon=False, fontsize=7)
    if "major_faults_mean" in s:
        axs[1].bar(x - 0.19, s["major_faults_mean"].clip(lower=1), 0.38, label="Major page faults")
    if "sys_pswpin_mean" in s:
        axs[1].bar(x + 0.19, (s["sys_pswpin_mean"] + s["sys_pswpout_mean"]).clip(lower=1), 0.38,
                   label="Pages swapped in+out")
    axs[1].set_yscale("log"); axs[1].set_xticks(x); axs[1].set_xticklabels(labels, fontsize=7)
    axs[1].set_ylabel("Count per run (log)"); axs[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(VM, "fig4_vm_configs.png"), dpi=200); plt.close(fig)


def analyze_fio():
    files = sorted(glob.glob(os.path.join(VM, "fio", "*_r*.json")))
    if not files:
        print("No fio results found in results/vm/fio/")
        return None
    rows = []
    for f in files:
        sched, pattern, rep = re.match(r"(.+)_(read|randread)_r(\d+)\.json", os.path.basename(f)).groups()
        txt = open(f).read()
        d = json.loads(txt[txt.index("{"):])          # skip any warning lines before the JSON
        rd = d["jobs"][0]["read"]
        lat = rd.get("clat_ns", rd.get("lat_ns", {}))
        rows.append({"scheduler": sched, "pattern": pattern, "rep": int(rep), "iops": rd["iops"],
                     "bw_mib_s": rd["bw"] / 1024, "mean_lat_us": lat.get("mean", np.nan) / 1000,
                     "p99_lat_us": lat.get("percentile", {}).get("99.000000", np.nan) / 1000})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(VM, "fio_runs.csv"), index=False)
    summ = df.groupby(["pattern", "scheduler"]).agg(
        iops_mean=("iops", "mean"), iops_std=("iops", "std"), bw_mib_s=("bw_mib_s", "mean"),
        mean_lat_us=("mean_lat_us", "mean"), p99_lat_us=("p99_lat_us", "mean")).reset_index()
    summ.to_csv(os.path.join(VM, "fio_summary.csv"), index=False)
    print(summ.round(1).to_string(index=False))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for ax, pat, title in zip(axs, ["read", "randread"], ["Sequential 4 KiB reads", "Random 4 KiB reads"]):
        sub = summ[summ.pattern == pat]
        ax.bar(sub.scheduler, sub.iops_mean, yerr=sub.iops_std, capsize=3,
               color=["tab:blue", "tab:orange", "tab:green"][:len(sub)])
        ax.set_title(title, fontsize=9); ax.set_ylabel("IOPS (mean ± sd)")
    fig.tight_layout(); fig.savefig(os.path.join(VM, "fig5_fio.png"), dpi=200); plt.close(fig)
    return summ


if __name__ == "__main__":
    analyze_configs()
    analyze_fio()
    import shutil                                       # copy figures for the LaTeX report
    fig_dir = os.path.join(ROOT, "report", "figures")
    for f in ("fig4_vm_configs.png", "fig5_fio.png"):
        if os.path.exists(os.path.join(VM, f)):
            shutil.copy(os.path.join(VM, f), fig_dir)
            print(f"copied {f} -> report/figures/")
