#!/usr/bin/env python3
"""QXU6028 L2: authoritative 4000 x 400 grid, paired-repeat bootstrap.

Python >=3.10; numpy>=2.0, pandas>=2.0, matplotlib>=3.7.
Run: python3 reproduce.py --data GROUP_DATA_DIR --output NEW_OUTPUT
The group dataset is obtained separately through an authorized course channel.
The output directory must be new and empty. A Git-based checkpoint also
requires a checkout that tracks the original input data and full provenance.
This script performs L2 only; it does not invent L1/L3 results.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KB = 8.617333e-5  # eV/K, exactly as in the Methods Guide
NA = 3.816e15    # atoms/cm^2
EMAX = 2.5      # eV
N_ENERGY = 4000
N_EF = 400
N_BOOT = 200
DEFAULT_SEED = 20260923
BLUE, ORANGE, GREY = "#1764A3", "#D27720", "#525963"
SCRIPT = Path(__file__).resolve()
DEFAULT_OUTPUT = Path("/Users/dhs225/Downloads/L2提交")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def fermi(E, EF, T):
    return 1.0 / (np.exp(np.clip((E - EF) / (KB*T), -60, 60)) + 1.0)


def dos(E, Eg=0.0, c=1.0):
    # Eg=0 gives the cone. c is a fixed positive scale and cancels in normalisation.
    return c * np.maximum(np.abs(E) - Eg/2.0, 0.0)


def carrier_np(EF, T, Eg=0.0, c=1.0, n_energy=N_ENERGY):
    """Literal reference implementation from Methods Guide p7.

    Do NOT replace the masked full-grid trapezoid by sliced-grid quadrature,
    integration starting exactly at EF, band-edge counts, or an analytic count.
    Those are different numerical models from the one used for this dataset.
    """
    E = np.linspace(-EMAX, EMAX, n_energy)
    f, g = fermi(E, EF, T), dos(E, Eg, c)
    n = NA*np.trapezoid(np.where(E >= EF, g*f, 0.0), E)
    p = NA*np.trapezoid(np.where(E <= EF, g*(1.0-f), 0.0), E)
    return float(n), float(p)


def normalise(pairs):
    """Two independent positive scales; never take abs of the Hall signal."""
    pairs = np.asarray(pairs, dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.isfinite(pairs).all():
        raise ValueError("Expected finite (N,2) Hall/sheet pairs")
    if np.any(pairs[:, 1] <= 0):
        raise ValueError("Sheet resistance must be positive")
    scales = np.array([np.max(np.abs(pairs[:, 0])), np.max(pairs[:, 1])])
    if np.any(scales <= 0):
        raise ValueError("Cannot normalise an all-zero channel")
    return pairs/scales, scales


def build_library(T, n_energy=N_ENERGY, n_ef=N_EF):
    """Vectorise EF only; keep identical trapezoidal masks and fixed grids."""
    E = np.linspace(-EMAX, EMAX, n_energy)
    ef = np.linspace(-8*KB*T, 9*KB*T, n_ef)
    gaps = np.r_[0.0, np.arange(0.50, 8.001, 0.25)]
    f = fermi(E[None, :], ef[:, None], T)
    electron = np.where(E[None, :] >= ef[:, None], f, 0.0)
    hole = np.where(E[None, :] <= ef[:, None], 1.0-f, 0.0)
    curves, populations, scales = [], [], []
    for gap in gaps:
        g = dos(E, gap*KB*T)
        n = NA*np.trapezoid(electron*g, E, axis=1)
        p = NA*np.trapezoid(hole*g, E, axis=1)
        total = n+p
        if not (np.isfinite(total).all() and np.all(total > 0)):
            raise ValueError("Invalid carrier population")
        raw = np.column_stack([(p-n)/total**2, 1.0/total])  # e=mu=1
        xy, scale = normalise(raw)  # FULL curve, not the matched subset
        curves.append(xy)
        populations.append(np.column_stack([n, p]))
        scales.append(scale)
    return {"T": T, "ef": ef, "gaps": gaps, "xy": np.asarray(curves),
            "np": np.asarray(populations), "scales": np.asarray(scales),
            "n_energy": n_energy, "n_ef": n_ef}


def choose_model(rms, gaps):
    best_gap = 1 + int(np.argmin(rms[1:]))
    # Strict inequality is deliberate: exactly 15% improvement is not enough.
    selected = best_gap if (rms[best_gap] < 0.85*rms[0]
                            and gaps[best_gap] >= 1.0) else 0
    return selected, best_gap


def fit(pairs, lib):
    xy, scales = normalise(pairs)
    # models x measurement indices x 400 EF candidates, no index-to-index fit
    d2 = ((xy[None, :, None, :] - lib["xy"][:, None, :, :])**2).sum(axis=3)
    nearest = np.argmin(d2, axis=2)
    mins = np.take_along_axis(d2, nearest[:, :, None], axis=2)[..., 0]
    rms = np.sqrt(np.mean(mins, axis=1))
    selected, best_gap = choose_model(rms, lib["gaps"])
    return {"selected": selected, "best_gap": best_gap, "rms": rms,
            "nearest": nearest, "ef": lib["ef"][nearest[selected]],
            "xy": xy, "scales": scales, "distance": np.sqrt(mins[selected])}


def zero_crossings(index, hall):
    found = []
    for i in range(len(index)):
        if hall[i] == 0:
            found.append({"left_index": int(index[i]), "right_index": int(index[i]),
                          "linear_interpolated_index": float(index[i])})
        if i+1 < len(index) and hall[i]*hall[i+1] < 0:
            x = index[i] - hall[i]*(index[i+1]-index[i])/(hall[i+1]-hall[i])
            found.append({"left_index": int(index[i]), "right_index": int(index[i+1]),
                          "linear_interpolated_index": float(x)})
    return found


def load_data(data_dir, group):
    meta = json.loads((data_dir/"L2_meta.json").read_text(encoding="utf-8"))
    template = json.loads((data_dir/"results_template.json").read_text(encoding="utf-8"))
    if meta.get("group") != group or template.get("group") != group:
        raise ValueError("Group mismatch between --group, metadata and template")
    if meta.get("dataset_set") != template.get("dataset_set"):
        raise ValueError("Metadata and template dataset_set mismatch")
    T = float(meta["T_K"])
    if not np.isfinite(T) or T <= 0:
        raise ValueError("Invalid T_K")
    df = pd.read_csv(data_dir/"L2_data.csv")
    expected = ["measurement_index", "repeat", "R_H_raw", "R_xx_raw"]
    if list(df.columns) != expected or meta.get("columns") != expected:
        raise ValueError("Unexpected CSV columns or metadata columns")
    if df.empty or not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Empty data, missing values or non-finite numbers")
    for col in ["measurement_index", "repeat"]:
        if not np.all(df[col] == df[col].astype(int)):
            raise ValueError(f"{col} must contain integer identifiers")
    if df.duplicated(["measurement_index", "repeat"]).any():
        raise ValueError("Duplicate (measurement_index, repeat) rows")
    # First-appearance order is the acquisition order. Never sort by inferred EF.
    index = df["measurement_index"].drop_duplicates().to_numpy(dtype=int)
    if not np.all(np.diff(index) > 0):
        raise ValueError("Measurement indices are not increasing in first-appearance order. "
                         "Check acquisition order; refusing to reorder silently.")
    repeat_ids = sorted(df["repeat"].unique().astype(int).tolist())
    if len(repeat_ids) != 3:
        raise ValueError("The assessment requires exactly three repeat IDs")
    groups = []
    for idx in index:
        rows = df.loc[df.measurement_index == idx].set_index("repeat")
        if len(rows) != 3 or set(rows.index) != set(repeat_ids):
            raise ValueError(f"Index {idx} does not have three distinct repeats")
        groups.append(rows.loc[repeat_ids, ["R_H_raw", "R_xx_raw"]].to_numpy())
    reps = np.asarray(groups)  # indices x three PAIRED rows x channels
    if np.any(reps[..., 1] <= 0):
        raise ValueError("Non-positive sheet-resistance observation")
    files = [data_dir/n for n in ["L2_data.csv", "L2_meta.json", "results_template.json"]]
    hashes = {p.name: sha256(p) for p in files}
    manifest_path = data_dir/"manifest.json"
    manifest_checked = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["group"] != group:
            raise ValueError("Manifest group mismatch")
        by_name = {x["name"]: x for x in manifest["files"]}
        for path in files:
            record = by_name[path.name]
            if hashes[path.name] != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise ValueError(f"Raw file differs from supplied manifest: {path.name}")
        manifest_checked = True
    return {"meta": meta, "template": template, "T": T, "index": index,
            "reps": reps, "repeat_ids": repeat_ids, "mean": reps.mean(axis=1),
            "sd": reps.std(axis=1, ddof=1), "hashes": hashes,
            "manifest_verified": manifest_checked}


def bootstrap(reps, lib, seed, count=N_BOOT, verbose=True):
    rng = np.random.default_rng(seed)
    selected, ef, rows, r_cone, r_gap, best_gap = [], [], [], [], [], []
    for b in range(count):
        # One whole paired repeat row per index, NOT independent channel draws,
        # and NOT the conventional sample-three-and-average bootstrap.
        picks = rng.integers(0, 3, size=len(reps))
        outcome = fit(reps[np.arange(len(reps)), picks, :], lib)
        selected.append(outcome["selected"])
        ef.append(outcome["ef"])
        rows.append(picks)
        r_cone.append(outcome["rms"][0])
        r_gap.append(outcome["rms"][outcome["best_gap"]])
        best_gap.append(lib["gaps"][outcome["best_gap"]])
        if verbose and (b+1) % 50 == 0:
            print(f"Bootstrap {b+1}/{count}", flush=True)
    return {"selected": np.asarray(selected), "ef": np.asarray(ef),
            "draws": np.asarray(rows), "gap": lib["gaps"][selected],
            "r_cone": np.asarray(r_cone), "r_gap": np.asarray(r_gap),
            "best_gap": np.asarray(best_gap)}


def verify_git(required_files, strict=False):
    """A real HEAD is insufficient: each computational input must match HEAD."""
    info = {"commit": None, "verified": False, "reason": ""}
    try:
        root = Path(subprocess.check_output(["git", "-C", str(SCRIPT.parent),
                       "rev-parse", "--show-toplevel"], stderr=subprocess.PIPE,
                       text=True).strip())
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                         stderr=subprocess.PIPE, text=True).strip()
        if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            raise ValueError("Invalid Git commit")
        for file in required_files:
            file = Path(file).resolve()
            relative = file.relative_to(root).as_posix()
            committed = subprocess.check_output(["git", "-C", str(root), "show",
                                                f"HEAD:{relative}"], stderr=subprocess.PIPE)
            if committed != file.read_bytes():
                raise ValueError(f"Uncommitted change in {relative}")
        info.update(commit=commit, verified=True, reason="Source and inputs match HEAD")
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        info["reason"] = f"Cannot verify committed source and inputs: {exc}"
        if strict:
            raise ValueError(info["reason"] + "\nCommit the code, requirements and inputs, "
                             "then rerun with --checkpoint. No upload ZIP was created.") from exc
    return info


def gates(data, lib, central):
    n0, p0 = carrier_np(0.0, data["T"])
    if not np.isclose(n0, p0, rtol=1e-12):
        raise AssertionError("Neutrality symmetry failed")
    for model in [0, 1, len(lib["gaps"])-1]:
        for j in [0, 177, 399]:
            ref = carrier_np(lib["ef"][j], data["T"], lib["gaps"][model]*KB*data["T"])
            np.testing.assert_allclose(lib["np"][model, j], ref, rtol=1e-13)
    if not np.array_equal(np.sign(central["xy"][:, 0]), np.sign(data["mean"][:, 0])):
        raise AssertionError("Hall sign was altered")
    crossings = zero_crossings(data["index"], data["mean"][:, 0])
    if not crossings or not (np.min(data["mean"][:, 0]) < 0 < np.max(data["mean"][:, 0])):
        raise ValueError("No measured Hall zero crossing: check input before interpreting L2")
    if not np.all(np.min(lib["xy"][..., 0], axis=1) < 0):
        raise AssertionError("Negative model Hall wing missing")
    if not np.all(np.max(lib["xy"][..., 0], axis=1) > 0):
        raise AssertionError("Positive model Hall wing missing")
    wings = np.abs(lib["ef"]/(KB*data["T"])) >= 5
    s2 = lib["xy"][0, wings, 1]**2
    h = np.abs(lib["xy"][0, wings, 0])  # magnitude ONLY in this wing diagnostic
    ratio = h/s2
    slope = np.polyfit(np.log(lib["xy"][0, wings, 1]), np.log(h), 1)[0]
    return {"neutrality_n_equals_p": True, "literal_quadrature_spotchecks": 9,
            "independent_normalisation": True, "Hall_sign_preserved": True,
            "zero_crossings": crossings,
            "cone_wings": {"definition": "abs(EF/kBT)>=5; diagnostic only",
                "log_abs_RH_vs_log_Rs_slope": float(slope),
                "abs_RH_over_Rs_squared_mean": float(ratio.mean()),
                "ratio_coefficient_of_variation": float(ratio.std()/ratio.mean())}}


def set_plot_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11, "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "figure.dpi": 120, "savefig.dpi": 300,
        "axes.prop_cycle": matplotlib.cycler(color=[BLUE, ORANGE, GREY])})


def make_figures(out, data, lib, central, boot, check, diagnostics):
    set_plot_style()
    index, mean, sd = data["index"], data["mean"], data["sd"]
    selected, bg = central["selected"], central["best_gap"]
    gap, kT = lib["gaps"][selected], KB*data["T"]
    saved = []

    def save(fig, name):
        for ext in ["png", "svg"]:
            path = out/f"{name}.{ext}"
            fig.savefig(path, bbox_inches="tight")
            saved.append(path.name)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, layout="constrained")
    for ch, ax in enumerate(axes):
        for r, label in enumerate(data["repeat_ids"]):
            ax.plot(index, data["reps"][:, r, ch], "o-", ms=2.4, lw=0.75,
                    alpha=0.55, label=f"Repeat {label}")
        ax.errorbar(index, mean[:, ch], yerr=sd[:, ch], fmt="k.-", ms=3, lw=1,
                    elinewidth=0.6, label="Mean +/- repeat SD")
        for crossing in check["zero_crossings"]:
            ax.axvspan(crossing["left_index"], crossing["right_index"], color=ORANGE, alpha=0.14)
            ax.axvline(crossing["linear_interpolated_index"], color=GREY, ls=":", lw=0.8)
        ax.set_ylabel([r"Raw $R_H$ (supplied scale)", r"Raw $R_{xx}$ (supplied scale)"][ch])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].axhline(0, color="black", lw=0.65)
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].set_title(f"{data['meta']['group']} | raw repeats and Hall zero crossing")
    axes[1].set_xlabel("Measurement index")
    save(fig, "01_raw_repeats_zero_crossing")

    fig, ax = plt.subplots(figsize=(7.2, 5.7), layout="constrained")
    ax.plot(lib["xy"][0, :, 0], lib["xy"][0, :, 1],
            color=BLUE if selected == 0 else GREY, ls="-" if selected == 0 else "--", lw=1.6,
            label=f"{'Selected cone' if selected == 0 else 'Cone'}: RMS {central['rms'][0]:.4f}")
    if selected != 0:
        ax.plot(lib["xy"][selected, :, 0], lib["xy"][selected, :, 1], color=BLUE, lw=1.6,
                label=rf"Selected hard gap: $E_g/k_BT={gap:.2f}$")
    else:
        ax.plot(lib["xy"][bg, :, 0], lib["xy"][bg, :, 1], color=ORANGE, ls="--", lw=1.1,
                label=rf"Best gap (rejected): $E_g/k_BT={lib['gaps'][bg]:.2f}$")
    point = central["xy"]
    # SD bars use central scaling; they are repeat spread, not a normalisation-aware CI.
    ax.errorbar(point[:, 0], point[:, 1], xerr=sd[:, 0]/central["scales"][0],
                yerr=sd[:, 1]/central["scales"][1], fmt="none", ecolor="#999999",
                elinewidth=0.6, alpha=0.55, zorder=1)
    ax.scatter(point[:, 0], point[:, 1], s=20, facecolors="white", edgecolors="black",
               linewidths=0.8, label="Mean pairs +/- repeat SD", zorder=4)
    for i, label in [(0, "start"), (len(index)-1, "end")]:
        ax.annotate(f"{label} ({index[i]})", point[i],
                    xytext=(12, -16) if i == 0 else (-65, 14),
                    textcoords="offset points", fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.6})
    ax.axvline(0, color="black", lw=0.6, alpha=0.5)
    ax.set(xlabel=r"Normalised $\widetilde{R}_H$ (signed)",
           ylabel=r"Normalised $\widetilde{R}_s$", title="L2 model comparison")
    ax.legend(fontsize=9, loc="center", bbox_to_anchor=(0.5, 0.53),
              frameon=True, facecolor="white", edgecolor="white", framealpha=1)
    save(fig, "02_model_fit")

    lo, mid, hi = np.percentile(boot["ef"], [16, 50, 84], axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    ax.fill_between(index, lo, hi, color=BLUE, alpha=0.2,
                    label="200 paired-repeat draws: 16th-84th percentiles")
    ax.plot(index, central["ef"], "o-", color=BLUE, ms=3.2, lw=1.2, label="Fit to repeat means")
    ax.plot(index, mid, color=GREY, ls=":", lw=1, label="Bootstrap median")
    ax.axhline(0, color="black", lw=0.7)
    if gap > 0:
        for edge in [-gap*kT/2, gap*kT/2]:
            ax.axhline(edge, color=ORANGE, lw=0.8, ls="--")
        ax.plot([], [], color=ORANGE, ls="--", label=r"Band edges $\pm E_g/2$")
    for crossing in check["zero_crossings"]:
        ax.axvspan(crossing["left_index"], crossing["right_index"], color=ORANGE, alpha=0.1)
    ax.set(xlabel="Measurement index (original order)", ylabel=r"Assigned $E_F$ (eV)",
           title="L2 Fermi-level trajectory")
    ax.set_xticks(np.unique(np.r_[index[0], ax.get_xticks()[(ax.get_xticks() > index[0]) &
                                                          (ax.get_xticks() < index[-1])], index[-1]]))
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "03_EF_trajectory")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    axes[0].plot(lib["gaps"][1:], central["rms"][1:], "o-", color=BLUE, ms=3)
    axes[0].axhline(central["rms"][0], color=GREY, ls="--", label="Cone RMS")
    axes[0].axhline(0.85*central["rms"][0], color=ORANGE, ls=":", label="0.85 x cone RMS")
    axes[0].axvline(1, color=GREY, ls=":", lw=0.7)
    axes[0].scatter([lib["gaps"][bg]], [central["rms"][bg]], marker="*", s=90, color=ORANGE, zorder=4)
    axes[0].set(xlabel=r"Candidate $E_g/k_BT$", ylabel="RMS pair-space distance", title="Gap scan and selection rule")
    axes[0].legend(fontsize=8)
    values, counts = np.unique(boot["gap"], return_counts=True)
    axes[1].bar(values, counts, width=0.19, color=BLUE)
    axes[1].axvline(gap, color=ORANGE, ls="--", lw=1, label="Central selected gap")
    axes[1].set(xlabel=r"Selected bootstrap $E_g/k_BT$", ylabel="Count (out of 200)",
                title=f"Hard-gap selection: {diagnostics['hard_gap_fraction']:.1%}")
    axes[1].set_xlim(-0.3, max(1.0, float(values.max())+0.3))
    axes[1].set_ylim(0, N_BOOT*1.16)
    axes[1].legend(fontsize=8, loc="upper right")
    save(fig, "04_gap_selection_bootstrap")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), layout="constrained")
    residual = central["xy"]-lib["xy"][selected, central["nearest"][selected]]
    axes[0].plot(index, residual[:, 0], "o-", ms=2.5, lw=0.8, label=r"$\Delta\widetilde{R}_H$")
    axes[0].plot(index, residual[:, 1], "s-", ms=2.5, lw=0.8, label=r"$\Delta\widetilde{R}_s$")
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set(xlabel="Measurement index", ylabel="Data - nearest model point", title="Residuals in measurement order")
    axes[0].legend(fontsize=9)
    wings = np.abs(lib["ef"]/kT) >= 5
    x = lib["xy"][0, wings, 1]**2
    y = np.abs(lib["xy"][0, wings, 0])
    beta = np.dot(x, y)/np.dot(x, x)
    axes[1].scatter(x, y, s=9, facecolors="none", edgecolors=BLUE, label="Cone model wings")
    axes[1].plot([0, x.max()], [0, beta*x.max()], color=GREY, ls="--", label="Origin-constrained linear fit")
    axes[1].set(xlabel=r"$\widetilde{R}_s^{\,2}$", ylabel=r"$|\widetilde{R}_H|$",
                title=r"Cone gate: $|E_F/k_BT|\geq5$")
    axes[1].legend(fontsize=8)
    save(fig, "05_residuals_cone_wing_gate")

    return saved


def write_notes(out, data, lib, central, boot, diagnostics):
    gap = diagnostics["Eg_kT"]
    lo, med, hi = diagnostics["gap_percentiles_kT"]
    fraction = diagnostics["hard_gap_fraction"]
    model = diagnostics["model"]
    rc, rg = diagnostics["r_cone"], diagnostics["r_gap"]
    best = diagnostics["best_candidate_gap_kT"]
    zero = diagnostics["gates"]["zero_crossings"]
    kT = KB*data["T"]
    confidence = ("The model decision is unstable under repeat resampling."
                  if 0.25 < fraction < 0.75 else
                  "The model decision is consistently selected in the supplied repeat-resampling test."
                  if (fraction >= 0.75 and gap > 0) or (fraction <= 0.25 and gap == 0)
                  else "The central decision and repeat-resampling majority disagree; treat the class as uncertain.")
    if gap == 0:
        recommendation = ("Prefer sensor/interconnect exploration over thermally robust switching. "
            "The data do not resolve a hard gap under the prescribed decision rule, so an OFF state "
            "cannot be justified from these measurements. The Hall sign change supports ambipolar "
            "response, but normalised curves alone cannot establish absolute sensor sensitivity, "
            "mobility, contact resistance or interconnect performance.")
    else:
        recommendation = (f"The resolved hard gap ({gap:.2f} kBT) supports investigating switching "
            "more strongly than an ideal cone does. This is a conditional device recommendation, "
            "not proof of a useful transistor OFF state or on/off ratio. Finite-temperature carriers, "
            "contacts, traps and bias-dependent transport are not tested. Sensor use remains plausible "
            "because the Hall response changes across neutrality; absolute performance is not "
            "recoverable from separately normalised curves alone.")
    note = f"""# {data['meta']['group']} L2 checkpoint note

## Question and central result
Using {len(data['index'])} measurement indices with three paired repeats at {data['T']:g} K,
the selected class is **{model}**, with **Eg/kBT = {gap:.2f}**
(Eg = {gap*kT:.6f} eV). The central estimate is the fit to repeat means, not the
bootstrap median. Hall zero-crossing bracket(s): {', '.join(str(z['left_index'])+' to '+str(z['right_index']) for z in zero)}.
Linear interpolation locates the crossing at {', '.join(f"{z['linear_interpolated_index']:.3f}" for z in zero)}
in measurement-index units; this is not a calibrated gate-voltage or time coordinate.

## Method and model decision
Fermi-window populations use n above EF and p below EF, evaluated by masked
trapezoidal integration on the specified 4,000-point [-2.5,2.5] eV grid.
The two channels are independently normalised. The library uses 400 EF candidates
from -8 kBT to +9 kBT and hard gaps 0.50 to 8.00 kBT in steps of 0.25.
Each data pair is assigned its nearest discrete library point in Euclidean
normalised Hall/sheet space. No smoothing, continuous EF interpolation or
trajectory monotonicity constraint is imposed.

Cone RMS = {rc:.7f}; best hard-gap RMS = {rg:.7f} at Eg/kBT = {best:.2f}.
Improvement = {100*(1-rg/rc):.2f}%. A hard gap is selected only when
r_gap < 0.85 r_cone AND the best gap is at least 1 kBT. Both conditions are
applied to the central fit and to every bootstrap draw.

## Uncertainty and resolution limits
The prescribed 200 replicates use seed {diagnostics['seed']}. At each index,
one entire paired repeat row is sampled, followed by fresh normalisation,
fitting and model selection. The selected-gap bootstrap median is {med:.2f} kBT,
with 16th-84th percentiles [{lo:.2f}, {hi:.2f}] kBT.
Hard-gap selection fraction = {fraction:.3f} ({int(np.count_nonzero(boot['selected']))}/200).
{confidence}
This fraction is resampling stability, not a Bayesian probability that the
material is gapped. The plotted EF interval is pointwise, not a simultaneous
confidence band. This single-repeat-per-index procedure measures repeat-level
sensitivity and is not the conventional bootstrap standard error of a mean.

Eg_uncertainty_kT is stored as (P84-P16)/2 = {(hi-lo)/2:.6f}; full asymmetric
percentiles and the median are in L2_diagnostics.json. The supplied template
does not specify the uncertainty field type; this scalar convention is documented
explicitly rather than silently equating it to a 95% confidence interval.
Even a zero percentile width does not establish an exactly known gap. The gap
grid spacing is 0.25 kBT and EF spacing is {17/399:.6f} kBT
({17*kT/399:.7f} eV). Sub-thermal gaps cannot be reliably distinguished from a
cone by the prescribed rule. Three repeats do not capture model mismatch,
temperature uncertainty or systematic measurement error.

The independent-nearest-point trajectory retains acquisition order. Occasional
local reversals can reflect measurement noise and finite-grid assignment; they
are not removed to make the trajectory smoother. Endpoint assignments:
{diagnostics['EF_endpoint_hits']}; maximum adjacent EF step:
{diagnostics['maximum_adjacent_EF_step_eV']:.6f} eV.
Numerical resolution sensitivity checks (diagnostic only, not substituted for
the required 4,000 x 400 central fit) are in L2_diagnostics.json.

## Device implication
{recommendation}

## Evidence and reproducibility
02_model_fit shows the mean pairs and candidate curves; 03_EF_trajectory shows
the assigned EF and pointwise resampling interval. Supporting figures show raw
repeats, the gap scan, residuals and the ideal-cone wing gate. Raw channel scales
are left as supplied because L2_meta.json does not state physical units.
See processed_pairs.csv, EF_trajectory.csv and bootstrap_replicates.csv for
inspectable numerical evidence. Raw-file SHA-256 values and software versions
are recorded in L2_diagnostics.json.

Source: QXU6028 Virtual Lab Practical Methods Guide 2026-27, pp. 3, 6-9 and 14;
Group Coursework Assessment Brief, section 5.2 and L2 gate.
The methods guide is the controlling numerical procedure.
This is an L2 checkpoint, not the full L1-L3 or blind-test submission.
L1/L3 template fields remain unchanged unless an existing results file is passed.
The script never invents missing phase results or a repository commit.
"""
    (out/"brief_note.md").write_text(note, encoding="utf-8")
    return recommendation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None,
                        help="Path to authorized, untouched group dataset directory (required)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group", default="G29")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--base-results", type=Path, help="Preserve existing L1/L3 results")
    parser.add_argument("--checkpoint", action="store_true", help="Require real Git provenance and build upload ZIP")
    args = parser.parse_args(argv)
    if args.data is None:
        raise ValueError("Provide --data pointing to the authorized group dataset directory.")
    data_dir, out = args.data.expanduser().resolve(), args.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output folder is not empty. Choose a NEW output folder; previous files are not overwritten.")
    data = load_data(data_dir, args.group)
    required = [SCRIPT, data_dir/"L2_data.csv", data_dir/"L2_meta.json", data_dir/"results_template.json"]
    for optional in [SCRIPT.parent/"requirements.txt", data_dir/"manifest.json"]:
        if optional.exists():
            required.append(optional)
    if args.base_results:
        required.append(args.base_results)
    provenance = verify_git(required, args.checkpoint)
    print(f"{args.group}: {len(data['index'])} indices x 3 paired repeats; T={data['T']} K", flush=True)
    print("Building prescribed 32-model library (4000 energy x 400 EF points)...", flush=True)
    lib = build_library(data["T"])
    central = fit(data["mean"], lib)
    check = gates(data, lib, central)
    boot = bootstrap(data["reps"], lib, args.seed)
    gap_percentiles = np.percentile(boot["gap"], [16, 50, 84])
    selected, bg = central["selected"], central["best_gap"]
    model = "cone" if selected == 0 else "hard_gap"
    ef_lo, ef_med, ef_hi = np.percentile(boot["ef"], [16, 50, 84], axis=0)
    diag = {"group": args.group, "dataset_set": data["meta"]["dataset_set"],
        "model": model, "model_definition": "cone c|E|; hard_gap c*max(|E|-Eg/2,0)",
        "Eg_kT": float(lib["gaps"][selected]), "T_K": data["T"], "kBT_eV": KB*data["T"],
        "r_cone": float(central["rms"][0]), "r_gap": float(central["rms"][bg]),
        "selected_rms": float(central["rms"][selected]),
        "best_candidate_gap_kT": float(lib["gaps"][bg]),
        "improvement_fraction": float(1-central["rms"][bg]/central["rms"][0]),
        "seed": args.seed, "bootstrap_count": N_BOOT,
        "bootstrap_method": "One paired repeat row per index, renormalise, refit, reselect",
        "gap_percentiles_kT": gap_percentiles.tolist(),
        "gap_percentile_order": [16, 50, 84],
        "Eg_uncertainty_kT_definition": "(P84-P16)/2; full interval retained separately",
        "hard_gap_fraction": float(np.mean(boot["selected"] != 0)),
        "measurement_index": data["index"].tolist(),
        "EF_percentile16_eV": ef_lo.tolist(), "EF_median_eV": ef_med.tolist(),
        "EF_percentile84_eV": ef_hi.tolist(),
        "EF_endpoint_hits": int(np.sum((central["nearest"][selected] == 0) |
                                       (central["nearest"][selected] == N_EF-1))),
        "maximum_adjacent_EF_step_eV": float(np.max(np.abs(np.diff(central["ef"])))),
        "EF_local_decreases": int(np.sum(np.diff(central["ef"]) < 0)),
        "normalisation_scales": central["scales"].tolist(),
        "grid": {"energy_eV": [-2.5, 2.5, N_ENERGY], "EF_kT": [-8, 9, N_EF],
                 "hard_gap_kT": [0.5, 8.0, 0.25]},
        "source_sha256": data["hashes"], "manifest_verified": data["manifest_verified"],
        "gates": check, "provenance": provenance,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "scope": "L2 only; L1/L3 untouched; not a complete blind-test pipeline"}
    print("Checking energy/EF resolution sensitivity (central fit only)...", flush=True)
    sensitivity = []
    for ne, nf in [(8000, 400), (4000, 800)]:
        other = build_library(data["T"], ne, nf)
        result = fit(data["mean"], other)
        sensitivity.append({"energy_points": ne, "EF_points": nf,
            "selected_model": "cone" if result["selected"] == 0 else "hard_gap",
            "Eg_kT": float(other["gaps"][result["selected"]]),
            "selected_rms": float(result["rms"][result["selected"]]),
            "max_EF_change_eV": float(np.max(np.abs(result["ef"]-central["ef"])))})
    diag["grid_sensitivity_diagnostic_only"] = sensitivity
    result = copy.deepcopy(data["template"])
    if args.base_results:
        result = json.loads(args.base_results.read_text(encoding="utf-8"))
        for key in ["schema_version", "group", "dataset_set"]:
            if result.get(key) != data["template"][key]:
                raise ValueError(f"Existing results mismatch in {key}")
        if set(result) != set(data["template"]) or any(set(result[k]) != set(data["template"][k]) for k in ["L1", "L2", "L3"]):
            raise ValueError("Existing results must preserve supplied keys and nesting")
    result["pipeline_commit"] = provenance["commit"]
    result["L2"].update(model=model, Eg_kT=diag["Eg_kT"],
        Eg_uncertainty_kT=float((gap_percentiles[2]-gap_percentiles[0])/2),
        EF_trajectory_eV=central["ef"].tolist())
    if len(result["L2"]["EF_trajectory_eV"]) != len(data["index"]):
        raise AssertionError("Trajectory length mismatch")
    json.dumps(result, allow_nan=False)  # fail before writing on NaN/Inf
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"results.json", result)
    write_json(out/"L2_diagnostics.json", diag)
    write_json(out/"run_config.json", {"group": args.group, "seed": args.seed,
        "data_folder": data_dir.name, "checkpoint": args.checkpoint,
        "base_results_used": args.base_results is not None,
        "command_pattern": "python3 reproduce.py --data DATA --output NEW_OUTPUT --group GROUP --seed SEED --checkpoint"})
    table = {"measurement_index": data["index"], "repeat_count": np.full(len(data["index"]), 3)}
    for ch, name in enumerate(["RH", "Rs"]):
        table[name+"_mean_raw"] = data["mean"][:, ch]
        table[name+"_sd_raw"] = data["sd"][:, ch]
        table[name+"_normalised"] = central["xy"][:, ch]
    pd.DataFrame(table).to_csv(out/"processed_pairs.csv", index=False)
    pd.DataFrame({"measurement_index": data["index"], "EF_eV": central["ef"],
        "EF_kT": central["ef"]/(KB*data["T"]), "EF_p16_eV": ef_lo,
        "EF_median_eV": ef_med, "EF_p84_eV": ef_hi,
        "nearest_EF_grid_index": central["nearest"][selected],
        "pair_distance": central["distance"]}).to_csv(out/"EF_trajectory.csv", index=False)
    pd.DataFrame({"Eg_kT": lib["gaps"], "model": ["cone"]+["hard_gap"]*31,
                  "rms": central["rms"]}).to_csv(out/"gap_scan.csv", index=False)
    pd.DataFrame({"replicate": np.arange(1, N_BOOT+1), "selected_Eg_kT": boot["gap"],
        "r_cone": boot["r_cone"], "r_gap": boot["r_gap"],
        "best_candidate_gap_kT": boot["best_gap"]}).to_csv(out/"bootstrap_replicates.csv", index=False)
    np.savez_compressed(out/"bootstrap_arrays.npz", EF_eV=boot["ef"],
        repeat_draw_zero_based=boot["draws"], selected_Eg_kT=boot["gap"],
        measurement_index=data["index"])
    np.savez_compressed(out/"model_library.npz", EF_eV=lib["ef"], Eg_kT=lib["gaps"],
                        RH_Rs_normalised=lib["xy"], populations=lib["np"])
    files = make_figures(out, data, lib, central, boot, check, diag)
    recommendation = write_notes(out, data, lib, central, boot, diag)
    (out/"commit_hash.txt").write_text((provenance["commit"] or "UNVERIFIED - not a submission commit")+"\n", encoding="utf-8")
    summary = (f"Group: {args.group}\nModel: {model}\nEg/kBT: {diag['Eg_kT']:.2f}\n"
        f"Eg (eV): {diag['Eg_kT']*KB*data['T']:.6f}\n"
        f"Bootstrap gap [p16, median, p84]: {gap_percentiles.tolist()}\n"
        f"Hard-gap selection fraction: {diag['hard_gap_fraction']:.3f}\n"
        f"RMS cone / best gap: {diag['r_cone']:.7f} / {diag['r_gap']:.7f}\n"
        f"Verified commit: {provenance['commit']}\n"
        f"{recommendation}\n")
    (out/"run_log.txt").write_text(summary, encoding="utf-8")
    package = None
    if args.checkpoint:
        name = f"{args.group}_L2_Checkpoint.zip"
        package = out/name
        members = ["results.json", "02_model_fit.png", "03_EF_trajectory.png",
            "brief_note.md", "commit_hash.txt", "L2_diagnostics.json", "EF_trajectory.csv",
            "01_raw_repeats_zero_crossing.png", "04_gap_selection_bootstrap.png",
            "05_residuals_cone_wing_gate.png", "run_config.json"]
        with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for member in members:
                z.write(out/member, member)
        with zipfile.ZipFile(package) as z:
            if z.testzip() is not None or set(z.namelist()) != set(members):
                raise AssertionError("Checkpoint ZIP validation failed")
            if json.loads(z.read("results.json"))["pipeline_commit"] != provenance["commit"]:
                raise AssertionError("ZIP commit mismatch")
    status = {"L2_fields_finite": True, "trajectory_length": len(central["ef"]),
        "raw_hashes_unchanged": all(sha256(data_dir/n)==v for n,v in data["hashes"].items()),
        "required_grid": [N_ENERGY, N_EF], "bootstrap_count": N_BOOT,
        "figures": files, "verified_git_commit": provenance["verified"],
        "checkpoint_zip": None if package is None else package.name,
        "note": "Automated checks do not establish hidden-truth accuracy or full-coursework completion."}
    write_json(out/"validation.json", status)
    print(summary, flush=True)
    print(f"Outputs: {out}", flush=True)
    if package:
        print(f"L2 checkpoint ZIP: {package}", flush=True)
    else:
        print("No official checkpoint ZIP: rerun from committed source with --checkpoint.", flush=True)
    return result, diag


if __name__ == "__main__":
    try:
        main()
    except (ValueError, AssertionError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
