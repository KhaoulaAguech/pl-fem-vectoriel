#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHOTONIC LANTERN V.1 - VERSION OPTIMISÉE POUR DATASET 2000+ VALIDÉS
Avec logs détaillés n_eff bruts + modes trouvés / guidés
Filtre n_eff assoupli + beta ajouté + debug amélioré
"""

import argparse
import numpy as np
import sys
import time
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

from scipy.sparse.linalg import eigsh
from scipy.sparse import eye
from scipy.spatial import Delaunay

try:
    from skfem import Basis
    from skfem.helpers import grad, dot
    from skfem.mesh import MeshTri
    from skfem.assembly import BilinearForm, asm
    from skfem.element import ElementTriP2
except ImportError:
    print("❌ Installez scikit-fem : pip install scikit-fem")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from losses import LossCalculator
except ImportError:
    LossCalculator = None
    print("⚠️ losses.py absent → pertes non calculées")


# ============================================================================
# CONFIG GLOBALE
# ============================================================================
POLYMER_N = 1.53
AIR_N     = 1.0
V_MIN     = 2.4
V_MAX     = 10.0

logger = logging.getLogger('pl_dataset_gen')


def setup_logger(level=logging.INFO, log_file=None):
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)


# ============================================================================
# GÉOMÉTRIE
# ============================================================================
class SimplePLGeometry:
    def __init__(self, sample, use_pml=False):
        self.n_cores = sample['n_cores']
        self.r_core  = sample['core_radius_um']
        self.pitch   = sample['pitch_um']
        self.n_core  = POLYMER_N + (sample['delta_n_percent'] - 1.0) / 100
        self.wl      = sample['wavelength_nm'] / 1000.0
        self.k0      = 2 * np.pi / self.wl

        # FIX: import géométries complètes
        try:
            from geometry_mcf import generate_core_positions
            self.positions, *_ = generate_core_positions(self.n_cores, self.pitch)
        except (ImportError, ValueError):
            # Fallback: circulaire simple
            angles = 2 * np.pi * np.arange(self.n_cores) / self.n_cores
            self.positions = self.pitch * np.column_stack([np.cos(angles), np.sin(angles)])

        max_dist = np.max(np.linalg.norm(self.positions, axis=1)) if self.n_cores > 1 else 0
        self.domain_radius = max(max_dist + 60, 120.0)

        self.use_pml = use_pml
        # ÉTAPE 5 FIX: épaisseur PML adaptive (15% du domaine, min 15µm max 40µm)
        if use_pml:
            raw_thick = self.domain_radius * 0.15
            self.pml_thickness = float(min(max(raw_thick, 15.0), 40.0))
        else:
            self.pml_thickness = 0.0
        self.pml_strength  = 0.3   # Force atténuation (0.3 = bon compromis absorption/spurious)

        self.V_number = self.k0 * self.r_core * np.sqrt(self.n_core**2 - AIR_N**2)

    def epsilon(self, x, y):
        eps = np.full_like(x, AIR_N**2, dtype=complex)
        for cx, cy in self.positions:
            mask = (x - cx)**2 + (y - cy)**2 <= self.r_core**2
            eps[mask] = self.n_core**2 + 0j

        if self.use_pml:
            r = np.sqrt(x**2 + y**2)
            pml_start = self.domain_radius - self.pml_thickness
            mask_pml = r > pml_start
            if np.any(mask_pml):
                rho = (r[mask_pml] - pml_start) / self.pml_thickness
                sigma = self.pml_strength * rho**3
                eps[mask_pml] *= (1 + 1j * sigma)
        return eps


# ============================================================================
# MESH
# ============================================================================
def build_mesh(geom):
    R = geom.domain_radius
    n_base = 36

    x = np.linspace(-R, R, n_base)
    y = np.linspace(-R, R, n_base)
    pts = np.vstack([x.ravel(), y.ravel()]).T

    theta = np.linspace(0, 2*np.pi, 28, endpoint=False)
    for cx, cy in geom.positions:
        for rr in np.linspace(0, geom.r_core * 2.0, 14):
            pts = np.vstack([pts, np.column_stack([cx + rr*np.cos(theta), cy + rr*np.sin(theta)])])

    pts = np.unique(np.round(pts, 6), axis=0)
    pts = pts[np.linalg.norm(pts, axis=1) <= R * 0.99]

    tri = Delaunay(pts)
    mesh = MeshTri(tri.points.T, tri.simplices.T)

    max_points = 18000
    for _ in range(2):
        if mesh.p.shape[1] > max_points:
            break
        mesh = mesh.refined()

    basis = Basis(mesh, ElementTriP2())
    logger.info(f"Mesh → {mesh.p.shape[1]:,} pts | {basis.N:,} DOFs")
    return mesh, basis


# ============================================================================
# SOLVEUR
# ============================================================================
def solve_modes(geom, basis, n_modes_target=12):
    @BilinearForm
    def stiff(u, v, _): return dot(grad(u), grad(v))

    @BilinearForm
    def mass(u, v, _): return u * v

    @BilinearForm
    def epsmass(u, v, w): return geom.epsilon(w.x[0], w.x[1]) * u * v

    K = asm(stiff, basis)
    M = asm(mass, basis)
    Me = asm(epsmass, basis)

    A = K - (geom.k0**2) * Me
    B = M + 1e-13 * eye(M.shape[0], format='csr')

    n_eff_shift = geom.n_core - 0.008
    sigma = - (geom.k0 * n_eff_shift)**2

    try:
        evals, evecs = eigsh(A, k=min(n_modes_target + 12, B.shape[0]-8),
                             M=B, sigma=sigma, which='LM', tol=1e-6, maxiter=8000)
    except Exception as e:
        logger.error(f"eigsh failed: {e}")
        return []

    # FIX V18.7 : filtrer evals < 0 avant sqrt (évite NaN)
    valid = evals < -1e-6
    n_eff_raw = np.full(len(evals), np.nan)
    n_eff_raw[valid] = np.real(np.sqrt(-evals[valid].astype(complex)) / geom.k0)

    if np.any(valid):
        neff_valid = n_eff_raw[valid]
        sorted_neff = np.sort(neff_valid)[::-1]
        logger.info(f"   n_eff bruts triés (top 8) : {sorted_neff[:8].tolist()}")
        logger.info(f"   n_eff min/max/moy : {np.nanmin(n_eff_raw):.4f} / "
                    f"{np.nanmax(n_eff_raw):.4f} / {np.nanmean(n_eff_raw):.4f}")
    else:
        logger.warning("   Aucun eigenvalue négatif → vérifier formulation")

    mask = (~np.isnan(n_eff_raw)) & (n_eff_raw > 0.9) & (n_eff_raw < geom.n_core + 0.5)

    # FIX V18.7 : confinement via produit scalaire masse FEM (exact, sans biais maillage)
    # M_core_l (loose = r*1.10) → confinement    M_core_s (strict = r) → core_overlap
    r_core   = geom.r_core
    positions = geom.positions

    @BilinearForm
    def mass_core_loose(u, v, w):
        x_q, y_q = w.x[0], w.x[1]
        m = np.zeros_like(x_q, dtype=float)
        for cx, cy in positions:
            m = np.where((x_q-cx)**2 + (y_q-cy)**2 <= (r_core*1.10)**2, 1.0, m)
        return m * u * v

    @BilinearForm
    def mass_core_strict(u, v, w):
        x_q, y_q = w.x[0], w.x[1]
        m = np.zeros_like(x_q, dtype=float)
        for cx, cy in positions:
            m = np.where((x_q-cx)**2 + (y_q-cy)**2 <= r_core**2, 1.0, m)
        return m * u * v

    M_core_l = asm(mass_core_loose,  basis)
    M_core_s = asm(mass_core_strict, basis)

    modes = []
    for i in np.where(mask)[0]:
        v = evecs[:, i].copy()
        norm = np.sqrt(np.real(v.conj() @ M @ v))
        if norm < 1e-9:
            continue
        v /= norm

        denom    = float(v @ M @ v) + 1e-20
        conf     = float(np.clip(v @ M_core_l @ v / denom, 0.0, 1.0))
        overlap  = float(np.clip(v @ M_core_s @ v / denom, 0.0, 1.0))
        beta     = geom.k0 * n_eff_raw[i]

        modes.append({
            'n_eff':        n_eff_raw[i],
            'beta':         beta,
            'field_vector': v,
            'confinement':  conf,
            'core_overlap': overlap,
        })

    modes.sort(key=lambda m: m['n_eff'], reverse=True)

    logger.info(f"   → Nombre total de modes trouvés (bruts) : {len(evals)}")
    logger.info(f"   → Modes candidats après filtre n_eff : {len(modes)}")

    if modes:
        logger.info(f"   → Clés du premier mode : {list(modes[0].keys())}")
        logger.info(f"   → beta fondamental : {modes[0]['beta']:.6f}")

    # FIX V18.7 : seuils cohérents avec pl_v17_minimal et solveur vectoriel
    N = geom.n_cores
    max_modes     = 3 * N
    CONF_TARGET   = 0.85
    CONF_FALLBACK = [(0.70, "assoupli"), (0.50, "permissif"), (0.30, "minimal")]
    OVERLAP_MIN   = 0.80

    def _core_ov(m):
        return m.get('core_overlap', m.get('confinement', 0.0))

    def _ok(m, c_thr):
        return m['confinement'] >= c_thr and _core_ov(m) >= OVERLAP_MIN

    kept     = [m for m in modes if _ok(m, CONF_TARGET)]
    min_conf = CONF_TARGET
    if len(kept) < N:
        for threshold, label in CONF_FALLBACK:
            alt = [m for m in modes if _ok(m, threshold)]
            if len(alt) >= N:
                kept = alt
                min_conf = threshold
                logger.warning(f"   Seuil conf abaissé à {threshold:.2f} ({label}) : {len(alt)} modes")
                break
        else:
            kept = sorted(modes, key=lambda m: m['confinement'], reverse=True)
            min_conf = 0.0
            logger.warning("   Filtre overlap désactivé (dernier recours)")

    kept.sort(key=lambda m: m['confinement'], reverse=True)
    kept = kept[:max_modes]
    logger.info(f"   → Modes guidés gardés (conf≥{min_conf:.2f}, overlap≥{OVERLAP_MIN:.2f}, top-{max_modes}) : {len(kept)}")

    if kept:
        logger.info(f"     → n_eff max       : {kept[0]['n_eff']:.5f}")
        logger.info(f"     → confinement max : {kept[0]['confinement']:.3f}")
        logger.info(f"     → core_overlap    : {_core_ov(kept[0]):.3f}")
    else:
        if modes:
            confs = [m['confinement'] for m in modes]
            logger.warning(f"   Aucun mode guidé. Meilleurs conf : {sorted(confs, reverse=True)[:5]}")
        else:
            logger.warning("   Aucun mode n'a passé le filtre n_eff")

    return kept


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Génère dataset photonic lantern avec modes et pertes")
    parser.add_argument('--n', type=int, default=20, help="Nombre d'échantillons")
    parser.add_argument('--out', type=str, default='./dataset_pl_2000')
    parser.add_argument('--no-pml', action='store_true', default=False)  # ÉTAPE 5 FIX: PML activé par défaut
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(log_level, out_dir / 'run.log')

    logger.info("=== PHOTONIC LANTERN V17.1 - GÉNÉRATION DATASET ===")
    logger.info(f"Échantillons ciblés : {args.n}")
    logger.info(f"PML : {'désactivé' if args.no_pml else 'activé'}")
    logger.info(f"Output : {out_dir.absolute()}")

    samples = []
    for i in range(args.n):
        s = {
            'sample_id': f"S{i:04d}",
            # FIX: toutes configurations MCF (pondérées)
            'n_cores': int(np.random.choice(
                [2, 3, 4, 5, 6, 7, 9, 12, 19],
                p=[0.05, 0.15, 0.05, 0.05, 0.15, 0.25, 0.10, 0.10, 0.10]
            )),
            'core_radius_um': np.random.uniform(0.9, 1.6),
            'pitch_um': np.random.uniform(4.5, 12.0),
            'delta_n_percent': np.random.uniform(0.8, 2.5),
            'wavelength_nm': float(np.random.choice([1530, 1550, 1570, 1590, 1610])),
        }
        samples.append(s)

    records = []
    for idx, s in enumerate(samples, 1):
        logger.info(f"[{idx}/{len(samples)}] {s['sample_id']} - {s['n_cores']} cœurs @ {s['wavelength_nm']} nm")

        try:
            geom = SimplePLGeometry(s, use_pml=not args.no_pml)
            mesh, basis = build_mesh(geom)
            modes = solve_modes(geom, basis, n_modes_target=int(s['n_cores'] * 2.8))

            losses = {}
            if LossCalculator and modes:
                try:
                    losses = LossCalculator.calculate_physical_losses(
                        modes, geom, 'mux', wavelength_nm=s['wavelength_nm']
                    )
                    if losses.get('success', False):
                        logger.info(f"   → IL={losses.get('IL_dB', 'N/A'):.3f} dB | "
                                   f"MDL={losses.get('MDL_dB', 'N/A'):.3f} dB | "
                                   f"PDL={losses.get('PDL_dB', 'N/A'):.3f} dB | "
                                   f"XT={losses.get('crosstalk_dB', 'N/A'):.2f} dB | "
                                   f"α_rad={losses.get('radiation_loss_dB_per_m', 'N/A'):.3f} dB/m")
                except Exception as e:
                    logger.warning(f"Pertes non calculées : {e}")

            logger.debug(f"Insertion record → wavelength_nm = {s['wavelength_nm']}")

            rec = {
                'sample_id': s['sample_id'],
                'n_cores': s['n_cores'],
                'wavelength_nm': s['wavelength_nm'],
                'core_radius_um': s['core_radius_um'],
                'pitch_um': s['pitch_um'],
                'delta_n_percent': s['delta_n_percent'],
                'V_number': round(geom.V_number, 3),
                'n_modes_found': len(modes),
                'success': len(modes) > 0,
                'n_eff_max': modes[0]['n_eff'] if modes else np.nan,
                'confinement_max': modes[0]['confinement'] if modes else np.nan,
                **losses
            }
            records.append(rec)

        except Exception as e:
            logger.error(f"Erreur sample {s['sample_id']} : {e}")
            records.append({'sample_id': s['sample_id'], 'success': False, 'error': str(e)})

    if pd and records:
        df = pd.DataFrame(records)
        raw_path = out_dir / 'dataset_raw.csv'
        df.to_csv(raw_path, index=False)
        logger.info(f"Dataset brut sauvegardé : {raw_path}")

        logger.info("Vérification wavelengths dans dataset :")
        unique_wl = sorted(df['wavelength_nm'].unique())
        logger.info(f" → Valeurs uniques : {unique_wl}")

        df_valid = df[df['success'] & (df['n_modes_found'] > 0)]
        if not df_valid.empty:
            # Correction crash + utilisation des noms réels de colonnes
            logger.info(f"Colonnes disponibles pour filtrage : {list(df_valid.columns)}")

            if 'IL_dB' in df_valid.columns and 'MDL_dB' in df_valid.columns:
                df_valid = df_valid[
                    df_valid['IL_dB'].between(0.3, 10.0) &
                    (df_valid['MDL_dB'].abs() < 8.0)
                ]
            else:
                logger.warning("Colonnes IL_dB ou MDL_dB absentes → aucun filtrage appliqué")

            valid_path = out_dir / 'dataset_valid_phys.csv'
            df_valid.to_csv(valid_path, index=False)
            logger.info(f"Dataset validé physiquement ({len(df_valid)} lignes) : {valid_path}")
            
            if len(df_valid) > 0:
                print("\nStatistiques designs valides :")
                print(df_valid.describe().round(3))
            else:
                logger.warning("Aucun design n'a passé le filtre de validation")
        else:
            logger.warning("Aucun sample avec modes trouvés")

    logger.info("=== FIN ===")


if __name__ == '__main__':
    main()
