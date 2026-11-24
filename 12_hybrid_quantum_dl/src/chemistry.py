"""Pure-Python SMILES 24-descriptor cheminformatics extractor & MoleculeNet BBBP loader."""
from __future__ import annotations

import csv
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BBBP_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"

DESCRIPTOR_NAMES: list[str] = [
    "count_C",
    "count_N",
    "count_O",
    "count_S",
    "count_F",
    "count_Cl",
    "count_Br",
    "count_P",
    "aromatic_atom_count",
    "aromatic_atom_ratio",
    "ring_closure_count",
    "double_bond_count",
    "triple_bond_count",
    "max_branch_depth",
    "chiral_center_count",
    "formal_charge_abs_sum",
    "estimated_mw",
    "hbd_proxy",
    "hba_proxy",
    "tpsa_proxy",
    "c_to_heteroatom_ratio",
    "lipinski_violations",
    "smiles_length",
    "token_shannon_entropy",
]

# SMILES grammar regex tokenizing bracket atoms, two-letter halogens/elements, ring indices, bonds, branches.
_SMILES_TOKEN_RE = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|Na|Ca|Fe|Li|Mg|Al|Zn|Cu|Mn|As|%\d{2}|[BCNOPSFIbcnops]|=|#|-|\(|\)|\.|/|\\|\d)"
)
_BRACKET_ELEM_RE = re.compile(r"^\[\d*([A-Z][a-z]?|[bcnops])")

_ATOMIC_WEIGHTS: dict[str, float] = {
    "H": 1.008,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.904,
    "Si": 28.085,
    "Se": 78.971,
    "Na": 22.990,
    "K": 39.098,
    "Ca": 40.078,
    "Mg": 24.305,
    "Fe": 55.845,
    "Zn": 65.38,
    "Li": 6.94,
    "Al": 26.982,
    "Cu": 63.546,
    "As": 74.922,
}

# Approximate implicit H count contribution per heavy atom for fast RDKit-free MW estimation.
_IMPLICIT_H_VALENCE: dict[str, float] = {
    "C": 1.6,
    "c": 0.65,
    "N": 1.0,
    "n": 0.25,
    "O": 0.5,
    "o": 0.0,
    "S": 0.2,
    "s": 0.0,
    "P": 0.2,
}


def _parse_bracket_charge(token: str) -> int:
    """Extract absolute formal charge magnitude from a SMILES bracket atom such as [NH3+] or [O-2]."""
    inner = token[1:-1]
    # Look for explicit +n / -n first, then repeated ++ / -- signs.
    m = re.search(r"([+-])(\d+)", inner)
    if m:
        return int(m.group(2))
    pos = inner.count("+")
    neg = inner.count("-")
    return abs(pos - neg)


def _parse_bracket_explicit_h(token: str) -> int:
    inner = token[1:-1]
    m = re.search(r"H(\d*)", inner)
    if not m:
        return 0
    return int(m.group(1)) if m.group(1) else 1


def extract_smiles_descriptors(smiles: str) -> np.ndarray:
    """Extract 24 real physicochemical descriptors from a SMILES string (zero RDKit dependency)."""
    s = (smiles or "").strip()
    if not s:
        return np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32)

    tokens = _SMILES_TOKEN_RE.findall(s)
    if not tokens:
        return np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32)

    counts = Counter()
    aromatic_count = 0
    heavy_atoms = 0
    ring_markers = 0
    double_bonds = 0
    triple_bonds = 0
    cur_depth = 0
    max_depth = 0
    chiral_centers = 0
    formal_charge_abs = 0
    est_mw = 0.0
    hbd_proxy = 0.0

    for tok in tokens:
        if tok == "(":
            cur_depth += 1
            if cur_depth > max_depth:
                max_depth = cur_depth
        elif tok == ")":
            cur_depth = max(0, cur_depth - 1)
        elif tok == "=":
            double_bonds += 1
        elif tok == "#":
            triple_bonds += 1
        elif tok.isdigit() or (tok.startswith("%") and len(tok) == 3):
            ring_markers += 1
        elif tok.startswith("[") and tok.endswith("]"):
            if "@" in tok:
                chiral_centers += 1
            formal_charge_abs += _parse_bracket_charge(tok)
            m_elem = _BRACKET_ELEM_RE.match(tok)
            if m_elem:
                raw_el = m_elem.group(1)
                is_arom = raw_el.islower()
                el = raw_el.upper() if len(raw_el) == 1 else raw_el
                if el != "H":
                    heavy_atoms += 1
                    counts[el] += 1
                    if is_arom:
                        aromatic_count += 1
                    est_mw += _ATOMIC_WEIGHTS.get(el, 14.0)
                exp_h = _parse_bracket_explicit_h(tok)
                est_mw += exp_h * _ATOMIC_WEIGHTS["H"]
                if el in ("N", "O") and exp_h > 0:
                    hbd_proxy += float(exp_h)
        elif tok in (".", "-", "/", "\\"):
            continue
        else:
            # Organic subset or two-letter element token
            is_arom = len(tok) == 1 and tok.islower()
            el = tok.upper() if len(tok) == 1 else tok
            heavy_atoms += 1
            counts[el] += 1
            if is_arom:
                aromatic_count += 1
            est_mw += _ATOMIC_WEIGHTS.get(el, 14.0)
            est_mw += _IMPLICIT_H_VALENCE.get(tok, 0.0) * _ATOMIC_WEIGHTS["H"]

    # Adjust estimated implicit hydrogens downward for double/triple bonds and rings.
    est_mw = max(12.0, est_mw - (double_bonds * 2.0 + triple_bonds * 4.0) * _ATOMIC_WEIGHTS["H"] * 0.4)

    count_c = float(counts["C"])
    count_n = float(counts["N"])
    count_o = float(counts["O"])
    count_s = float(counts["S"])
    count_f = float(counts["F"])
    count_cl = float(counts["Cl"])
    count_br = float(counts["Br"])
    count_p = float(counts["P"])

    ring_closures = ring_markers / 2.0
    arom_ratio = aromatic_count / max(1.0, float(heavy_atoms))

    # Estimate HBD from explicit bracket N/O-H plus uncharged aliphatic hydroxyl/amine proxies.
    aliph_o = max(0.0, count_o - float(double_bonds) * 0.45)
    aliph_n = max(0.0, count_n - float(triple_bonds) * 0.5)
    hbd_total = hbd_proxy + 0.35 * aliph_o + 0.30 * aliph_n
    hba_total = count_n + count_o

    # Ertl-inspired TPSA proxy (polar surface area in Angstrom^2) strongly governs BBB permeability!
    tpsa_proxy = 20.23 * count_o + 23.79 * count_n + 25.30 * count_s + 13.59 * count_p

    heteroatoms = count_n + count_o + count_s + count_f + count_cl + count_br + count_p
    c_to_het = count_c / max(1.0, heteroatoms)

    # Lipophilic proxy cLogP (Wildman-Crippen inspired linear approximation)
    clogp_proxy = (
        0.35 * count_c
        + 0.45 * count_cl
        + 0.65 * count_br
        + 0.15 * count_f
        - 0.65 * count_o
        - 0.70 * count_n
        - 0.30 * formal_charge_abs
    )
    lipinski = (
        float(est_mw > 500.0)
        + float(hbd_total > 5.0)
        + float(hba_total > 10.0)
        + float(clogp_proxy > 5.0)
    )

    # Shannon entropy over SMILES token distribution (captures structural complexity).
    tok_freq = Counter(tokens)
    total_toks = float(len(tokens))
    shannon_entropy = -sum(
        (c / total_toks) * math.log2(c / total_toks) for c in tok_freq.values() if c > 0
    )

    vec = np.array(
        [
            count_c,
            count_n,
            count_o,
            count_s,
            count_f,
            count_cl,
            count_br,
            count_p,
            float(aromatic_count),
            float(arom_ratio),
            float(ring_closures),
            float(double_bonds),
            float(triple_bonds),
            float(max_depth),
            float(chiral_centers),
            float(formal_charge_abs),
            float(est_mw),
            float(hbd_total),
            float(hba_total),
            float(tpsa_proxy),
            float(c_to_het),
            float(lipinski),
            float(len(s)),
            float(shannon_entropy),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def parse_bbbp_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Parse MoleculeNet BBBP.csv into (X [N, 24], y [N], smiles_list)."""
    features: list[np.ndarray] = []
    labels: list[int] = []
    smiles_list: list[str] = []

    with Path(csv_path).open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            raw_lbl = (row.get("p_np") or "").strip()
            if not smi or raw_lbl not in {"0", "1"}:
                continue
            features.append(extract_smiles_descriptors(smi))
            labels.append(int(raw_lbl))
            smiles_list.append(smi)

    if not features:
        raise ValueError(f"No valid BBBP records parsed from {csv_path}")

    return np.stack(features, axis=0), np.asarray(labels, dtype=np.int64), smiles_list


def download_bbbp_csv(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "BBBP.csv"
    if not csv_path.exists() or csv_path.stat().st_size < 1024:
        urllib.request.urlretrieve(BBBP_URL, csv_path)
    return csv_path


def load_bbbp_dataset(data_dir: Path, seed: int = 42) -> dict[str, Any]:
    """Download/load real MoleculeNet BBBP dataset with leakage-free stratified 70/15/15 split."""
    csv_path = download_bbbp_csv(data_dir)
    X, y, smiles_list = parse_bbbp_csv(csv_path)

    indices = np.arange(len(y))
    idx_train, idx_tmp, y_train, y_tmp = train_test_split(
        indices, y, test_size=0.30, stratify=y, random_state=seed
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[idx_train]).astype(np.float32)
    X_val = scaler.transform(X[idx_val]).astype(np.float32)
    X_test = scaler.transform(X[idx_test]).astype(np.float32)

    return {
        "name": "MoleculeNet BBBP (Blood-Brain Barrier Permeability)",
        "source_url": BBBP_URL,
        "feature_names": DESCRIPTOR_NAMES,
        "X_train": X_train,
        "y_train": y_train.astype(np.int64),
        "X_val": X_val,
        "y_val": y_val.astype(np.int64),
        "X_test": X_test,
        "y_test": y_test.astype(np.int64),
        "test_smiles": [smiles_list[i] for i in idx_test],
        "n_total": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
    }
