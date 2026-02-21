#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEOMETRY_UNIFIED.PY — Géométrie unifiée MCF / Taper / MMF
===========================================================
Version : V18.8
Date    : 2026-02-17

Intègre en un seul module cohérent :
  1.  MCFGeometry      — positions cœurs, ε(x,y), attributs unifiés
  2.  TaperSection     — 3 sections (Source → MUX → Output)
  3.  MMFGeometry      — fibre multimode sortie
  4.  PhotonicLantern  — assemblage complet

ATTRIBUTS GARANTIS (utilisés par solver_fem, mesh, pl_v17_minimal) :
─────────────────────────────────────────────────────────────────────
  geometry.positions         → np.ndarray (N,2)  positions cœurs [µm]
  geometry.core_positions    → alias de positions (compat mesh.py)
  geometry.core_radii        → np.ndarray (N,)   rayons cœurs [µm]
  geometry.r_core            → float  rayon moyen (compat main.py)
  geometry.n_core            → float  indice cœur
  geometry.n_clad            → float  indice gaine
  geometry.n_cores           → int    nombre de cœurs
  geometry.k0                → float  vecteur d'onde [µm⁻¹]
  geometry.wavelength        → float  longueur d'onde [µm]
  geometry.domain_radius     → float  rayon domaine FEM [µm]
  geometry.cladding_radius   → float  rayon gaine [µm]
  geometry.pml_thickness     → float  épaisseur PML [µm]
  geometry.use_complex_pml   → bool
  geometry.V_number          → float  paramètre V
  geometry.epsilon(x, y)     → np.ndarray complexe
  geometry.hash              → str    empreinte unique

SOURCES MCF (configurations expérimentales) :
────────────────────────────────────────────
  N=2  Kokubun & Koshiba, IEICE 2009
  N=3  Fontaine OE 2012 ; Zhu OL 2011
  N=4  Hayashi OE 2011 (Furukawa) ; NEC/KDDI submarine 2022
  N=5  Jinno OFC 2020 (pentagone régulier)
  N=6  Zhu OL 2011 (ring) ; Stern Optica 2021 (5+1)
  N=7  Carpenter Nat.Photon. 2015 ; Dana Light Sci.Appl. 2024
  N=8  Hayashi OFC 2015 (Sumitomo 1+7)
  N=9  Igarashi OE 2014 (KDDI 3×3)
  N=12 Takenaga/Ishida OFC 2014 (Fujikura)
  N=13 Takenaga OFC 2011 (Fujikura 1+6+6)
  N=19 Mizuno Nat.Photon. 2016 ; van Weerdenburg 2024
"""

import numpy as np
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Union
import logging

logger = logging.getLogger('pl_v18.geometry_unified')


# ============================================================================
# CONSTANTES
# ============================================================================
class PhysConst:
    N_SILICA      = 1.4440    # silice à 1550 nm
    N_POLYMER_BASE= 1.5200    # IP-Dip (NANOSCRIBE) base
    N_AIR         = 1.0000
    PML_STRENGTH  = 3.0
    PML_ORDER     = 2
    PML_THICKNESS_UM = 10.0   # µm


# ============================================================================
# 1. POSITIONS MCF — TOUTES CONFIGURATIONS EXPÉRIMENTALES
# ============================================================================

def mcf_positions(
    n_cores: int,
    pitch: float,
    variant: Optional[str] = None
) -> Tuple[np.ndarray, str, bool, int, float]:
    """
    Génère les positions de cœurs pour toutes les configurations MCF publiées.

    Args:
        n_cores  : nombre de cœurs (1,2,3,4,5,6,7,8,9,12,13,19)
        pitch    : pas nearest-neighbour [µm]
        variant  : pour N=6 : 'ring' (défaut) ou 'pentagon_center'

    Returns:
        (positions, config_type, has_central_core, n_peripheral, R_ring)
        positions   : np.ndarray (N,2)
        config_type : str identifiant unique
        has_central : bool cœur central ?
        n_peripheral: int nombre cœurs périphériques
        R_ring      : float rayon du ring extérieur [µm]
    """
    p = float(pitch)

    if n_cores == 1:
        return np.array([[0., 0.]]), 'single_1', True, 0, 0.

    elif n_cores == 2:
        # Dual-core linéaire — Kokubun & Koshiba IEICE 2009
        return (np.array([[-p/2, 0.], [p/2, 0.]]),
                'linear_2', False, 2, p/2)

    elif n_cores == 3:
        # Triangulaire équilatéral — Fontaine OE 2012 ; Zhu OL 2011
        a = np.radians([90, 210, 330])
        return (p * np.column_stack([np.cos(a), np.sin(a)]),
                'triangular_3', False, 3, p)

    elif n_cores == 4:
        # Carré 2×2 — Hayashi OE 2011 (Furukawa) ; NEC/KDDI 2022
        h = p / 2
        return (np.array([[-h,-h],[h,-h],[-h,h],[h,h]]),
                'square_2x2_4', False, 4, h*np.sqrt(2))

    elif n_cores == 5:
        # Pentagone régulier — Jinno OFC 2020 (5-core CSS, Fujikura)
        # NB : PAS un carré+centre
        a = np.radians(90 + np.arange(5)*72)
        return (p * np.column_stack([np.cos(a), np.sin(a)]),
                'pentagonal_ring_5', False, 5, p)

    elif n_cores == 6:
        if variant == 'pentagon_center':
            # 1+5 — Stern Optica 2021 (PL SDM 6-core)
            a = np.radians(90 + np.arange(5)*72)
            ring = p * np.column_stack([np.cos(a), np.sin(a)])
            return (np.vstack([[0.,0.], ring]),
                    'pentagon_center_6', True, 5, p)
        else:
            # 6 sur hexagone sans centre — Zhu OL 2011 ; Ryf ECOC 2012
            a = np.radians(np.arange(6)*60)
            return (p * np.column_stack([np.cos(a), np.sin(a)]),
                    'hexagonal_ring_6', False, 6, p)

    elif n_cores == 7:
        # Hexagonal 1+6 STANDARD — Carpenter Nat.Photon. 2015 ; Dana LSA 2024
        a = np.radians(np.arange(6)*60)
        ring = p * np.column_stack([np.cos(a), np.sin(a)])
        return (np.vstack([[0.,0.], ring]),
                'hexagonal_1plus6_7', True, 6, p)

    elif n_cores == 8:
        # Hexagonal 1+7 — Hayashi OFC 2015 Th5C.6 (Sumitomo 125µm O-band)
        a = np.radians(np.arange(7)*(360/7))
        ring = p * np.column_stack([np.cos(a), np.sin(a)])
        return (np.vstack([[0.,0.], ring]),
                'heptagonal_center_8', True, 7, p)

    elif n_cores == 9:
        # Grille 3×3 — Igarashi OE 2014 (KDDI 9-core MCF)
        coords = [-p, 0., p]
        pos = np.array([[x,y] for y in coords for x in coords])
        return (pos, 'square_3x3_9', True, 8, p*np.sqrt(2))

    elif n_cores == 12:
        # Double ring sans centre — Takenaga/Ishida OFC 2014 (Fujikura)
        a1 = np.radians(np.arange(6)*60)
        r1 = p   * np.column_stack([np.cos(a1), np.sin(a1)])
        a2 = np.radians(np.arange(6)*60 + 30)
        r2 = p*np.sqrt(3) * np.column_stack([np.cos(a2), np.sin(a2)])
        return (np.vstack([r1, r2]),
                'hex_double_ring_12', False, 12, p*np.sqrt(3))

    elif n_cores == 13:
        # 1+6+6 — Takenaga OFC 2011 (Fujikura 13-core)
        a1 = np.radians(np.arange(6)*60)
        r1 = p   * np.column_stack([np.cos(a1), np.sin(a1)])
        a2 = np.radians(np.arange(6)*60 + 30)
        r2 = p*np.sqrt(3) * np.column_stack([np.cos(a2), np.sin(a2)])
        return (np.vstack([[0.,0.], r1, r2]),
                'hex_1plus6plus6_13', True, 12, p*np.sqrt(3))

    elif n_cores == 19:
        # 1+6+12 STANDARD TÉLÉCOM — Mizuno Nat.Photon. 2016 ; van Weerdenburg 2024
        pos = [[0., 0.]]
        a1  = np.radians(np.arange(6)*60)
        for a in a1: pos.append([p*np.cos(a),         p*np.sin(a)])
        for a in a1: pos.append([2*p*np.cos(a),       2*p*np.sin(a)])
        a2  = np.radians(np.arange(6)*60 + 30)
        for a in a2: pos.append([p*np.sqrt(3)*np.cos(a), p*np.sqrt(3)*np.sin(a)])
        return (np.array(pos),
                'hex_1plus6plus12_19', True, 18, 2*p)

    else:
        valid = [1,2,3,4,5,6,7,8,9,12,13,19]
        raise ValueError(f"n_cores={n_cores} non supporté. Valides : {valid}")


# ============================================================================
# 2. GÉOMÉTRIE MCF UNIFIÉE
# ============================================================================

class MCFGeometry:
    """
    Géométrie Multi-Core Fiber avec attributs unifiés.

    Expose TOUS les attributs attendus par :
      - solver_fem.py    (.positions, .core_radii, .n_core, .n_clad,
                          .k0, .epsilon(), .domain_radius, .pml_thickness)
      - mesh.py          (.core_positions, .core_radii, .r_core,
                          .domain_radius, .pml_thickness, .use_complex_pml)
      - pl_v17_minimal   (.n_cores, .V_number, .hash)
      - losses.py        (.positions, .core_radii, .n_core, .n_clad,
                          .taper_length, .k0)
    """

    SUPPORTED_N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 19]

    def __init__(
        self,
        n_cores          : int,
        pitch_um         : float,
        core_radius_um   : float,
        n_core           : float,
        n_clad           : float           = PhysConst.N_AIR,
        wavelength_um    : float           = 1.55,
        cladding_radius  : Optional[float] = None,
        pml_thickness    : float           = PhysConst.PML_THICKNESS_UM,
        pml_strength     : float           = PhysConst.PML_STRENGTH,
        pml_order        : int             = PhysConst.PML_ORDER,
        use_complex_pml  : bool            = True,
        taper_length_um  : Optional[float] = None,
        variant          : Optional[str]   = None,
    ):
        self.n_cores   = int(n_cores)
        self.n_core    = float(n_core)
        self.n_clad    = float(n_clad)
        self.delta_n   = self.n_core - self.n_clad
        self.wavelength= float(wavelength_um)
        self.k0        = 2 * np.pi / self.wavelength

        if self.delta_n < 1e-6:
            raise ValueError(f"Δn={self.delta_n:.2e} trop faible")

        # Positions MCF
        (self.positions, self.config_type,
         self.has_central_core, self.n_peripheral,
         self.R_ring) = mcf_positions(n_cores, pitch_um, variant)

        # Rayons cœurs (uniformes)
        self.core_radii = np.full(self.n_cores, float(core_radius_um))

        # ── Alias pour compatibilité ─────────────────────────────────
        self.core_positions = self.positions          # mesh.py
        self.r_core         = float(core_radius_um)  # main.py

        # Paramètre V
        self.V_number = self.k0 * self.r_core * np.sqrt(
            max(self.n_core**2 - self.n_clad**2, 0.)
        )

        # Pitch et packing
        if n_cores > 1:
            dists = [np.linalg.norm(self.positions[i]-self.positions[j])
                     for i in range(n_cores)
                     for j in range(i+1, n_cores)]
            self.pitch     = float(np.min(dists))
            self.pitch_min = self.pitch
            max_r          = float(np.max(np.linalg.norm(self.positions, axis=1)))
        else:
            self.pitch = self.pitch_min = 0.
            max_r = 0.

        self.pitch_ratio = self.pitch / (2*self.r_core) if self.r_core > 0 else 0.

        # Rayon gaine
        self.cladding_radius = (
            cladding_radius if cladding_radius is not None
            else max(max_r * 1.8 + self.r_core * 2, 20.)
        )

        # Domaine FEM
        self._domain_radius = max(
            max_r + self.r_core * 4,
            self.cladding_radius + pml_thickness * 1.2
        )

        # PML
        self.pml_thickness  = float(pml_thickness)
        self.pml_strength   = float(pml_strength)
        self.pml_order      = int(pml_order)
        self.use_complex_pml= bool(use_complex_pml)

        # Taper
        self.taper_length = taper_length_um

        # Packing
        area_c = n_cores * np.pi * self.r_core**2
        area_t = np.pi * (max_r + self.r_core)**2 if n_cores > 1 else area_c
        self.packing_efficiency = float(area_c / max(area_t, 1e-9))

        # Hash
        self._hash = self._compute_hash()

        logger.debug(
            f"MCFGeometry créé: N={n_cores} config={self.config_type} "
            f"pitch={self.pitch:.2f}µm r={self.r_core:.2f}µm "
            f"V={self.V_number:.2f} domain_R={self._domain_radius:.1f}µm"
        )

    # ── Propriétés ───────────────────────────────────────────────────

    @property
    def domain_radius(self) -> float:
        return self._domain_radius

    @property
    def hash(self) -> str:
        return self._hash

    def _compute_hash(self) -> str:
        h = hashlib.sha256()
        h.update(str(self.n_cores).encode())
        h.update(self.positions.tobytes())
        h.update(self.core_radii.tobytes())
        h.update(f"{self.n_core:.6f}{self.n_clad:.6f}{self.wavelength:.6f}".encode())
        h.update(f"{self.cladding_radius:.4f}{self.pml_thickness:.2f}".encode())
        h.update(str(self.use_complex_pml).encode())
        return h.hexdigest()[:20]

    # ── ε(x,y) ───────────────────────────────────────────────────────

    def epsilon(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Permittivité relative complexe au point (x,y).
        Inclut PML annulaire si use_complex_pml=True.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        eps = np.full_like(x, self.n_clad**2, dtype=np.complex128)

        for (cx, cy), r in zip(self.positions, self.core_radii):
            mask = (x-cx)**2 + (y-cy)**2 <= r**2
            eps[mask] = self.n_core**2

        if self.use_complex_pml:
            r_dist   = np.sqrt(x**2 + y**2)
            pml_start= self._domain_radius - self.pml_thickness
            pml_mask = r_dist > pml_start
            if np.any(pml_mask):
                rn  = np.clip((r_dist[pml_mask]-pml_start)/self.pml_thickness, 0., 1.)
                sig = self.pml_strength * rn**self.pml_order
                eps[pml_mask] *= (1. + 1j*sig)

        return eps

    # ── Validation ───────────────────────────────────────────────────

    def validate(self) -> Tuple[bool, str]:
        if self.delta_n < 5e-4:
            return False, f"Δn trop faible ({self.delta_n:.2e})"
        if self.V_number < 0.5:
            return False, f"V-number trop faible ({self.V_number:.2f})"
        if self.V_number > 20.:
            return False, f"V-number très élevé ({self.V_number:.2f}) → multimode"
        for i in range(self.n_cores):
            for j in range(i+1, self.n_cores):
                d = np.linalg.norm(self.positions[i]-self.positions[j])
                if d < (self.core_radii[i]+self.core_radii[j])*0.85:
                    return False, f"Chevauchement cœurs {i}↔{j}: d={d:.2f}µm"
        return True, "OK"

    # ── Taper ────────────────────────────────────────────────────────

    def get_tapered(self, z: float) -> 'MCFGeometry':
        """Retourne géométrie scalée à la position z du taper."""
        if self.taper_length is None or self.taper_length <= 0.:
            return self
        s = float(np.clip(z / self.taper_length, 0., 1.))
        g = MCFGeometry(
            n_cores         = self.n_cores,
            pitch_um        = self.pitch * s if self.n_cores > 1 else self.pitch,
            core_radius_um  = self.r_core * s,
            n_core          = self.n_core,
            n_clad          = self.n_clad,
            wavelength_um   = self.wavelength,
            cladding_radius = self.cladding_radius,
            pml_thickness   = self.pml_thickness,
            pml_strength    = self.pml_strength,
            pml_order       = self.pml_order,
            use_complex_pml = self.use_complex_pml,
            taper_length_um = self.taper_length,
        )
        return g

    # ── Info dict ────────────────────────────────────────────────────

    def get_info(self) -> Dict:
        return {
            'n_cores'           : self.n_cores,
            'config_type'       : self.config_type,
            'has_central_core'  : self.has_central_core,
            'n_peripheral'      : self.n_peripheral,
            'R_ring_um'         : float(self.R_ring),
            'pitch_um'          : float(self.pitch),
            'pitch_ratio'       : float(self.pitch_ratio),
            'core_radius_um'    : float(self.r_core),
            'n_core'            : self.n_core,
            'n_clad'            : self.n_clad,
            'delta_n'           : float(self.delta_n),
            'V_number'          : float(self.V_number),
            'wavelength_um'     : self.wavelength,
            'cladding_radius_um': float(self.cladding_radius),
            'domain_radius_um'  : float(self._domain_radius),
            'pml_thickness_um'  : float(self.pml_thickness),
            'packing_efficiency': float(self.packing_efficiency),
            'taper_length_um'   : self.taper_length,
            'hash'              : self.hash,
        }

    def __repr__(self) -> str:
        return (f"MCFGeometry(N={self.n_cores}, {self.config_type}, "
                f"pitch={self.pitch:.1f}µm, r={self.r_core:.2f}µm, "
                f"V={self.V_number:.2f}, n={self.n_core:.4f}/{self.n_clad:.4f})")


# ============================================================================
# 3. SECTION TAPER
# ============================================================================

@dataclass
class TaperSection:
    """
    Structure taper 3 sections : Source → MUX → Sortie
    Réf: Dana et al., Light: Science & Applications 13:116 (2024)
    """
    # Section 1 — Source (down-taper MCF → bundle)
    source_length_um    : float
    source_diam_in_um   : float     # diamètre entrée (MCF pitch×2)
    source_diam_out_um  : float     # diamètre sortie section source

    # Section 2 — MUX (regroupement)
    mux_length_um       : float
    mux_diam_in_um      : float
    mux_diam_out_um     : float

    # Section 3 — Sortie (up-taper vers MMF)
    output_length_um    : float
    output_diam_in_um   : float
    output_diam_out_um  : float     # diamètre MMF sortie

    # Profils taper
    profile             : str   = 'exponential'   # 'linear','power','sinusoidal','exponential'
    exponent            : float = 1.0

    @property
    def total_length_um(self) -> float:
        return self.source_length_um + self.mux_length_um + self.output_length_um

    @property
    def total_length_mm(self) -> float:
        return self.total_length_um / 1000.

    def validate(self) -> Tuple[bool, str]:
        tol = 0.1
        if abs(self.source_diam_out_um - self.mux_diam_in_um) > tol:
            return False, (f"Discontinuité Source→MUX: "
                           f"{self.source_diam_out_um:.3f} ≠ {self.mux_diam_in_um:.3f} µm")
        if abs(self.mux_diam_out_um - self.output_diam_in_um) > tol:
            return False, (f"Discontinuité MUX→Output: "
                           f"{self.mux_diam_out_um:.3f} ≠ {self.output_diam_in_um:.3f} µm")
        if self.total_length_um <= 0:
            return False, "Longueur totale nulle"
        return True, "TaperSection valide"

    def scale_at(self, z_um: float) -> float:
        """Facteur d'échelle géométrique à la position z."""
        L = self.total_length_um
        if L <= 0:
            return 1.
        t = float(np.clip(z_um / L, 0., 1.))
        profiles = {
            'linear'      : lambda t: t,
            'power'       : lambda t: t**self.exponent,
            'sinusoidal'  : lambda t: 0.5*(1 - np.cos(np.pi*t)),
            'exponential' : lambda t: (np.exp(t) - 1) / (np.e - 1),
        }
        return float(profiles.get(self.profile, profiles['linear'])(t))

    @classmethod
    def from_mcf(cls, mcf: MCFGeometry, total_length_mm: float,
                 output_diam_um: float = 125.) -> 'TaperSection':
        """Construit un taper standard depuis une géométrie MCF."""
        L = total_length_mm * 1000.
        L1, L2, L3 = L*0.15, L*0.60, L*0.25
        d_src  = 2*(mcf.R_ring + mcf.r_core)
        d_mid  = d_src * 0.3
        return cls(
            source_length_um   = L1,
            source_diam_in_um  = d_src,
            source_diam_out_um = d_mid,
            mux_length_um      = L2,
            mux_diam_in_um     = d_mid,
            mux_diam_out_um    = output_diam_um * 0.15,
            output_length_um   = L3,
            output_diam_in_um  = output_diam_um * 0.15,
            output_diam_out_um = output_diam_um,
        )


# ============================================================================
# 4. GÉOMÉTRIE MMF (sortie du taper)
# ============================================================================

class MMFGeometry:
    """
    Fibre multimode sortie du photonic lantern.
    Gaine silice standard : ⌀125µm, NA=0.22.
    """
    def __init__(
        self,
        core_radius_um  : float = 25.,
        clad_radius_um  : float = 62.5,
        n_core          : float = PhysConst.N_SILICA * 1.005,
        n_clad          : float = PhysConst.N_SILICA,
        wavelength_um   : float = 1.55,
        length_um       : float = 100.,
    ):
        self.r_core      = float(core_radius_um)
        self.r_clad      = float(clad_radius_um)
        self.n_core      = float(n_core)
        self.n_clad      = float(n_clad)
        self.wavelength  = float(wavelength_um)
        self.length_um   = float(length_um)
        self.k0          = 2*np.pi / self.wavelength
        self.NA          = float(np.sqrt(max(n_core**2 - n_clad**2, 0.)))
        self.V_number    = self.k0 * self.r_core * self.NA
        self.M_modes     = max(1, int(self.V_number**2 / 2))

    @property
    def n_modes_estimate(self) -> int:
        return self.M_modes

    def get_info(self) -> Dict:
        return {
            'r_core_um'    : self.r_core,
            'r_clad_um'    : self.r_clad,
            'n_core'       : self.n_core,
            'n_clad'       : self.n_clad,
            'NA'           : self.NA,
            'V_number'     : self.V_number,
            'M_modes'      : self.M_modes,
            'length_um'    : self.length_um,
            'wavelength_um': self.wavelength,
        }

    def __repr__(self) -> str:
        return (f"MMFGeometry(r={self.r_core:.1f}µm, NA={self.NA:.3f}, "
                f"V={self.V_number:.1f}, M≈{self.M_modes})")


# ============================================================================
# 5. ASSEMBLAGE PHOTONIC LANTERN COMPLET
# ============================================================================

class PhotonicLantern:
    """
    Assemblage complet : MCF + Taper + MMF

    Usage:
        pl = PhotonicLantern.build(n_cores=7, pitch_um=8., core_radius_um=1.2,
                                   n_core=1.53, n_clad=1.0)
        geom = pl.mcf          # géométrie FEM
        taper= pl.taper        # structure taper
        mmf  = pl.mmf          # fibre sortie
        print(pl.summary())
    """

    def __init__(self, mcf: MCFGeometry, taper: TaperSection, mmf: MMFGeometry):
        self.mcf   = mcf
        self.taper = taper
        self.mmf   = mmf

    @classmethod
    def build(
        cls,
        n_cores          : int,
        pitch_um         : float,
        core_radius_um   : float,
        n_core           : float,
        n_clad           : float           = 1.0,
        wavelength_um    : float           = 1.55,
        taper_length_mm  : float           = 0.375,
        mmf_core_radius  : float           = 25.,
        mmf_clad_radius  : float           = 62.5,
        cladding_radius  : Optional[float] = None,
        pml_thickness    : float           = 10.,
        use_complex_pml  : bool            = True,
        variant          : Optional[str]   = None,
    ) -> 'PhotonicLantern':
        mcf = MCFGeometry(
            n_cores          = n_cores,
            pitch_um         = pitch_um,
            core_radius_um   = core_radius_um,
            n_core           = n_core,
            n_clad           = n_clad,
            wavelength_um    = wavelength_um,
            cladding_radius  = cladding_radius,
            pml_thickness    = pml_thickness,
            use_complex_pml  = use_complex_pml,
            taper_length_um  = taper_length_mm * 1000.,
            variant          = variant,
        )
        taper = TaperSection.from_mcf(mcf, taper_length_mm,
                                       output_diam_um=2*mmf_core_radius)
        mmf   = MMFGeometry(
            core_radius_um = mmf_core_radius,
            clad_radius_um = mmf_clad_radius,
            n_core         = n_core * 0.998,     # MMF ≈ même matériau, NA≠0
            n_clad         = n_clad * 1.002 if n_clad > 1.01 else n_clad,
            wavelength_um  = wavelength_um,
        )
        return cls(mcf, taper, mmf)

    def summary(self) -> str:
        lines = [
            "══════════════════════════════════════════════════",
            "   PHOTONIC LANTERN — PARAMÈTRES COMPLETS",
            "══════════════════════════════════════════════════",
            f"  MCF  : {self.mcf}",
            f"         V={self.mcf.V_number:.2f}  pitch={self.mcf.pitch:.2f}µm  r={self.mcf.r_core:.2f}µm",
            f"         Δn={self.mcf.delta_n:.4f}  packing={self.mcf.packing_efficiency*100:.1f}%",
            f"  Taper: L={self.taper.total_length_mm:.3f}mm  profil={self.taper.profile}",
            f"         d_in={self.taper.source_diam_in_um:.1f}µm → d_out={self.taper.output_diam_out_um:.1f}µm",
            f"  MMF  : {self.mmf}",
            "══════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


# ============================================================================
# 6. COMPATIBILITÉ ARRIÈRE — PhotonicLanternGeometry alias
# ============================================================================

class PhotonicLanternGeometry(MCFGeometry):
    """
    Alias de MCFGeometry pour compatibilité avec le code existant
    qui utilise PhotonicLanternGeometry (geometry.py).
    """
    def __init__(self,
                 n_cores, arrangement, core_positions, core_radii,
                 n_core, n_clad=1.0, cladding_radius=None,
                 wavelength=1.55, taper_length=None,
                 pml_thickness=10., pml_strength=3., pml_order=2,
                 use_complex_pml=True, **kwargs):
        # Déduire pitch depuis les positions
        positions = np.atleast_2d(np.asarray(core_positions, dtype=np.float64))
        if len(positions) > 1:
            dists = [np.linalg.norm(positions[i]-positions[j])
                     for i in range(len(positions))
                     for j in range(i+1, len(positions))]
            pitch = float(np.min(dists))
        else:
            pitch = float(np.max(core_radii)) * 4

        r_core = float(np.mean(core_radii))

        super().__init__(
            n_cores         = n_cores,
            pitch_um        = pitch,
            core_radius_um  = r_core,
            n_core          = n_core,
            n_clad          = n_clad,
            wavelength_um   = wavelength,
            cladding_radius = cladding_radius,
            pml_thickness   = pml_thickness,
            pml_strength    = pml_strength,
            pml_order       = pml_order,
            use_complex_pml = use_complex_pml,
            taper_length_um = taper_length,
        )
        # Écraser les positions avec celles fournies exactement
        self.positions      = positions
        self.core_positions = positions
        self.core_radii     = np.asarray(core_radii, dtype=np.float64)
        self.arrangement    = str(arrangement)


# ============================================================================
# MÉTADONNÉES DE RÉFÉRENCE
# ============================================================================

MCF_CONFIGS: Dict[int, Dict] = {
    1 : {'label': 'Single-core',          'standard': False, 'ref': 'baseline'},
    2 : {'label': 'Dual-core linéaire',   'standard': True,  'ref': 'Kokubun IEICE 2009'},
    3 : {'label': '3-core triangulaire',  'standard': True,  'ref': 'Fontaine OE 2012'},
    4 : {'label': '4-core carré 2×2',     'standard': True,  'ref': 'Hayashi OE 2011 → submarine NEC/KDDI 2022'},
    5 : {'label': '5-core pentagone',     'standard': True,  'ref': 'Jinno OFC 2020 (CSS, Fujikura)'},
    6 : {'label': '6-core ring / 5+1',    'standard': True,
         'ref': 'Zhu OL 2011 (ring) ; Stern Optica 2021 (5+1)',
         'variants': {'ring': '6 sur hexagone', 'pentagon_center': '1+5 pentagone'}},
    7 : {'label': '7-core hex 1+6',       'standard': True,  'ref': 'Carpenter Nat.Photon. 2015 ; Dana LSA 2024'},
    8 : {'label': '8-core hex 1+7',       'standard': True,  'ref': 'Hayashi OFC 2015 Th5C.6 (Sumitomo)'},
    9 : {'label': '9-core carré 3×3',     'standard': True,  'ref': 'Igarashi OE 2014 (KDDI)'},
    12: {'label': '12-core hex 6+6',      'standard': True,  'ref': 'Takenaga/Ishida OFC 2014 (Fujikura)'},
    13: {'label': '13-core hex 1+6+6',    'standard': True,  'ref': 'Takenaga OFC 2011 (Fujikura)'},
    19: {'label': '19-core hex 1+6+12',   'standard': True,  'ref': 'Mizuno Nat.Photon. 2016 ; van Weerdenburg 2024'},
}

SAMPLING_WEIGHTS: Dict[int, float] = {
    2:0.04, 3:0.11, 4:0.13, 5:0.05, 6:0.10,
    7:0.30, 8:0.05, 9:0.08, 12:0.07, 13:0.07, 19:0.10,
}


# ============================================================================
# VALIDATION & TEST
# ============================================================================

if __name__ == '__main__':
    import sys
    print("=" * 65)
    print("  GEOMETRY_UNIFIED.PY V18.8 — VALIDATION COMPLÈTE")
    print("=" * 65)

    # Test toutes configurations MCF
    print("\n─── Configurations MCF ───")
    print(f"{'N':>4}  {'Type':<28} {'Ctr'} {'V':>5}  {'Δn':>6}  Valide")
    print("-" * 65)
    ok = True
    for n in MCFGeometry.SUPPORTED_N:
        try:
            g = MCFGeometry(n, 8., 1.2, 1.53, 1.0)
            valid, msg = g.validate()
            status = "✅" if valid else f"⚠  {msg}"
            print(f"{n:>4}  {g.config_type:<28} {'✓' if g.has_central_core else '·'} "
                  f"{g.V_number:>5.2f}  {g.delta_n:>6.4f}  {status}")
        except Exception as e:
            print(f"{n:>4}  ERREUR: {e}")
            ok = False

    # Test attributs compat
    print("\n─── Attributs de compatibilité ───")
    g7 = MCFGeometry(7, 8., 1.2, 1.53, 1.0)
    checks = [
        ('positions',       hasattr(g7, 'positions')       and g7.positions.shape == (7,2)),
        ('core_positions',  hasattr(g7, 'core_positions')  and g7.core_positions.shape == (7,2)),
        ('core_radii',      hasattr(g7, 'core_radii')      and len(g7.core_radii) == 7),
        ('r_core',          hasattr(g7, 'r_core')          and g7.r_core == 1.2),
        ('n_core',          hasattr(g7, 'n_core')          and g7.n_core == 1.53),
        ('n_clad',          hasattr(g7, 'n_clad')          and g7.n_clad == 1.0),
        ('k0',              hasattr(g7, 'k0')              and g7.k0 > 0),
        ('domain_radius',   g7.domain_radius > 0),
        ('pml_thickness',   g7.pml_thickness > 0),
        ('use_complex_pml', isinstance(g7.use_complex_pml, bool)),
        ('epsilon()',       g7.epsilon(np.array([0.]),np.array([0.])).shape == (1,)),
        ('hash',            len(g7.hash) == 20),
        ('V_number',        g7.V_number > 0),
        ('taper_length',    g7.taper_length is None),
    ]
    for name, result in checks:
        print(f"  {'✅' if result else '❌'} {name}")
        if not result: ok = False

    # Test PhotonicLantern complet
    print("\n─── PhotonicLantern assemblage ───")
    pl = PhotonicLantern.build(n_cores=7, pitch_um=8., core_radius_um=1.2,
                               n_core=1.53, n_clad=1.0, taper_length_mm=0.375)
    print(pl.summary())
    tv, tm = pl.taper.validate()
    print(f"  Taper valid: {'✅' if tv else '❌'} {tm}")
    print(f"  MMF: {pl.mmf}")

    # Test ε
    print("\n─── Test ε(x,y) ───")
    g = MCFGeometry(7, 8., 1.2, 1.53, 1.0)
    eps_center = np.real(g.epsilon(np.array([0.]), np.array([0.])))[0]
    eps_far    = np.real(g.epsilon(np.array([100.]), np.array([0.])))[0]
    print(f"  ε(0,0)    = {eps_center:.4f}  (attendu {1.53**2:.4f} = n_core²)")
    print(f"  ε(100,0)  = {eps_far:.4f}   (gaine/PML, attendu ≤ {1.0**2:.4f})")

    print(f"\n{'✅ Tous les tests passés' if ok else '❌ Certains tests ont échoué'}")
    print("=" * 65)
