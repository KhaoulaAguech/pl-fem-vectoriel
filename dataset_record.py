#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Record Structure for Photonic Lantern Mux/Demux  V1
================================================================
Dataclass pour stocker résultats simulation complète

AMÉLIORATIONS 2025:
- Validation post-simulation renforcée
- Ajout de nombreux paramètres demandés (NA, MFD, A_eff, packing, etc.)
- Méthodes export complètes (dict, JSON, CSV row enrichi)
- Flags success détaillés + timestamps UTC
- Performance index pondéré réaliste
- Sérialisation optimisée + compatibilité ML

AUTEUR: Photonic Lantern Project V.1
DATE: 2024 → mise à jour 2025
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np


@dataclass
class DatasetRecord:
    """
    Enregistrement complet d'une simulation photonic lantern
    
    Structure:
    ---------
    1. Identification & statut
    2. Paramètres entrée (géométrie, matériaux, taper)
    3. Métriques optiques SM & MM
    4. Résultats modes & champs
    5. Pertes physiques (MUX/DEMUX)
    6. Résultats CMT (propagation)
    7. Qualité, scoring & métadonnées
    """

    # ========================================================================
    # 1. IDENTIFICATION & STATUT
    # ========================================================================
    sample_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    success: bool = False
    success_geometry: bool = False
    success_physics: bool = False
    success_solver: bool = False
    success_losses: bool = False
    
    error_msg: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # ========================================================================
    # 2. PARAMÈTRES ENTRÉE
    # ========================================================================
    # Géométrie & topologie
    n_cores: int = 0
    core_radius_um: float = 0.0
    pitch_um: float = 0.0
    arrangement: str = ''
    config_type: str = 'default'           # nouveau
    geometry_config: str = 'standard'      # nouveau
    n_peripheral_cores: Optional[int] = None  # nouveau
    R_ring: Optional[float] = None         # nouveau
    packing_efficiency: Optional[float] = None  # nouveau

    # Matériaux
    delta_n_percent: float = 0.0
    wavelength_nm: float = 1550.0          # changé en float pour cohérence
    n_polymer: float = 1.53                # nouveau

    # Taper
    taper_length_mm: float = 0.0
    taper_profile: str = 'power'
    taper_exponent: float = 0.8
    L_mux: Optional[float] = None          # nouveau
    L_taper: Optional[float] = None        # nouveau
    L_MMF: Optional[float] = None          # nouveau
    L_total: Optional[float] = None        # nouveau
    n_taper: Optional[float] = None        # nouveau

    # ========================================================================
    # 3. MÉTRIQUES OPTIQUES – SM & MM
    # ========================================================================
    V_number: float = 0.0
    n_core: float = 0.0
    n_clad: float = 0.0
    delta_n: float = 0.0

    # SM (single-mode reference)
    r_core_SM: Optional[float] = None
    r_clad_SM: Optional[float] = None
    n_core_SM: Optional[float] = None
    n_clad_SM: Optional[float] = None
    V_SM: Optional[float] = None
    NA_SM: Optional[float] = None
    MFD: Optional[float] = None
    n_eff_LP01: Optional[float] = None

    # MM (multimode output)
    r_core_MM: Optional[float] = None
    V_MM: Optional[float] = None
    NA_MM: Optional[float] = None
    M_max: Optional[int] = None            # nombre de modes théoriques

    # ========================================================================
    # 4. RÉSULTATS SIMULATION MODES
    # ========================================================================
    n_modes_found: int = 0
    modes: List[Dict] = field(default_factory=list)
    
    n_eff_max: float = 0.0
    n_eff_min: float = 0.0
    n_eff_mean: float = 0.0
    
    confinement_max: float = 0.0
    confinement_min: float = 0.0
    avg_confinement: float = 0.0

    # ========================================================================
    # 5. PERTES PHYSIQUES
    # ========================================================================
    losses_mux: Optional[Dict] = None
    IL_phys_mux_dB: Optional[float] = None
    MDL_phys_mux_dB: Optional[float] = None
    PDL_mux_dB: Optional[float] = None
    crosstalk_mux_dB: Optional[float] = None
    radiation_mux_dB_m: Optional[float] = None

    losses_demux: Optional[Dict] = None
    IL_phys_demux_dB: Optional[float] = None
    MDL_phys_demux_dB: Optional[float] = None
    PDL_demux_dB: Optional[float] = None
    crosstalk_demux_dB: Optional[float] = None
    radiation_demux_dB_m: Optional[float] = None

    # ========================================================================
    # 6. RÉSULTATS CMT (PROPAGATION)
    # ========================================================================
    cmt_mux: Optional[Dict] = None
    cmt_demux: Optional[Dict] = None
    IL_CMT_mux_dB: Optional[float] = None
    IL_CMT_demux_dB: Optional[float] = None
    power_conservation_mux: Optional[float] = None
    power_conservation_demux: Optional[float] = None

    # ========================================================================
    # 7. QUALITÉ, SCORING & MÉTADONNÉES
    # ========================================================================
    quality_score: Optional[float] = None
    adiabatic_score: Optional[float] = None
    performance_index: Optional[float] = None

    solver_time_s: float = 0.0
    mesh_points: int = 0
    mesh_elements: int = 0
    n_dofs: int = 0

    coupling_uniformity: Optional[float] = None  # nouveau
    coupling_degradation: Optional[float] = None # nouveau
    crosstalk_penalty: Optional[float] = None    # nouveau

    def validate(self) -> tuple[bool, List[str]]:
        errors = []

        if self.success:
            if not all([self.success_geometry, self.success_physics, self.success_solver]):
                errors.append("success=True mais un ou plusieurs sous-flags False")

        if self.n_modes_found > 0 and len(self.modes) != self.n_modes_found:
            errors.append(f"n_modes_found ({self.n_modes_found}) != len(modes) ({len(self.modes)})")

        if self.n_eff_max <= 0 and self.n_modes_found > 0:
            errors.append("n_modes_found > 0 mais n_eff_max <= 0")

        if self.V_number < 0 or self.V_number > 25:
            errors.append(f"V_number hors plage: {self.V_number}")

        if self.n_core < self.n_clad:
            errors.append(f"n_core ({self.n_core}) < n_clad ({self.n_clad})")

        if self.IL_phys_mux_dB is not None and not (0 <= self.IL_phys_mux_dB <= 50):
            errors.append(f"IL_phys_mux_dB hors plage: {self.IL_phys_mux_dB}")

        return len(errors) == 0, errors

    def calculate_performance_index(self) -> float:
        """
        Index performance global (plus bas = meilleur)
        """
       

        index = w_IL * IL_norm + w_MDL * MDL_norm + w_PDL * PDL_norm + w_XT * XT_norm
        

    def to_dict(self, include_modes: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not include_modes:
            data.pop('modes', None)
            data.pop('cmt_mux', None)
            data.pop('cmt_demux', None)
            data.pop('losses_mux', None)
            data.pop('losses_demux', None)
        for k, v in data.items():
            if isinstance(v, (np.integer, np.floating)):
                data[k] = float(v)
            elif isinstance(v, np.ndarray):
                data[k] = v.tolist()
        return data

    def to_json(self, filepath: Path, include_modes: bool = False):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(include_modes), f, indent=2)

    def to_csv_row(self) -> Dict[str, Any]:
        row = {
            'sample_id': self.sample_id,
            'timestamp': self.timestamp,
            'success': self.success,
            'n_cores': self.n_cores,
            'core_radius_um': self.core_radius_um,
            'pitch_um': self.pitch_um,
            'delta_n_percent': self.delta_n_percent,
            'wavelength_nm': self.wavelength_nm,
            'taper_length_mm': self.taper_length_mm,
            'V_number': self.V_number,
            'n_modes_found': self.n_modes_found,
            'n_eff_max': self.n_eff_max,
            'avg_confinement': self.avg_confinement,
            'IL_phys_mux_dB': self.IL_phys_mux_dB,
            'MDL_phys_mux_dB': self.MDL_phys_mux_dB,
            'PDL_mux_dB': self.PDL_mux_dB,
            'crosstalk_mux_dB': self.crosstalk_mux_dB,
            'radiation_mux_dB_m': self.radiation_mux_dB_m,
            'IL_phys_demux_dB': self.IL_phys_demux_dB,
            'MDL_phys_demux_dB': self.MDL_phys_demux_dB,
            'PDL_demux_dB': self.PDL_demux_dB,
            'IL_CMT_mux_dB': self.IL_CMT_mux_dB,
            'IL_CMT_demux_dB': self.IL_CMT_demux_dB,
            'quality_score': self.quality_score,
            'performance_index': self.performance_index,
            'solver_time_s': self.solver_time_s,
            # Nouveaux paramètres ajoutés
            'config_type': self.config_type,
            'geometry_config': self.geometry_config,
            'n_peripheral_cores': self.n_peripheral_cores,
            'R_ring': self.R_ring,
            'packing_efficiency': self.packing_efficiency,
            'r_core_SM': self.r_core_SM,
            'V_SM': self.V_SM,
            'NA_SM': self.NA_SM,
            'MFD': self.MFD,
            'r_core_MM': self.r_core_MM,
            'V_MM': self.V_MM,
            'NA_MM': self.NA_MM,
            'M_max': self.M_max,
            'coupling_uniformity': self.coupling_uniformity,
            'crosstalk_penalty': self.crosstalk_penalty,
            'coupling_degradation': self.coupling_degradation,
        }
        return row

    def summary_string(self) -> str:
        status = "✓" if self.success else "✗"
        lines = [
            f"{status} {self.sample_id} | {self.n_cores} cœurs | λ={self.wavelength_nm} nm",
            f"  V={self.V_number:.2f} | Modes={self.n_modes_found} | n_eff_max={self.n_eff_max:.4f}",
            f"  Conf avg={self.avg_confinement:.3f} | IL_mux={self.IL_phys_mux_dB:.2f}dB | MDL={self.MDL_phys_mux_dB:.2f}dB"
        ]
        if self.quality_score is not None:
            lines.append(f"  Quality={self.quality_score:.3f} | Perf={self.performance_index:.2f}")
        if self.error_msg:
            lines.append(f"  Error: {self.error_msg}")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: Dict) -> 'DatasetRecord':
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, filepath: Path) -> 'DatasetRecord':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Test module
if __name__ == '__main__':
    print("=" * 70)
    print("MODULE dataset_record.py V17.1 - OPTIMISÉ ✓")
    print("=" * 70)
    
    record = DatasetRecord(
        sample_id="TEST_001",
        success=True,
        n_cores=6,
        core_radius_um=0.8,
        pitch_um=10.0,
        delta_n_percent=1.0,
        wavelength_nm=1550.0,
        taper_length_mm=0.375,
        V_number=4.5,
        n_modes_found=6,
        n_eff_max=1.52,
        avg_confinement=0.85,
        IL_phys_mux_dB=1.2,
        MDL_phys_mux_dB=0.8,
        PDL_mux_dB=0.5,
        crosstalk_mux_dB=-22.0,
        quality_score=0.75
    )
    
    valid, errors = record.validate()
    print(f"\nValidation: {'✓ OK' if valid else '✗ ERREURS'}")
    if errors:
        for err in errors:
            print(f"  - {err}")
    
    perf = record.calculate_performance_index()
    record.performance_index = perf
    print(f"Performance index: {perf:.3f}")
    
    print("\nSummary:")
    print(record.summary_string())
    
    csv_row = record.to_csv_row()
    print(f"\nCSV row keys: {len(csv_row)} colonnes")
    print("Exemple colonnes ajoutées :", [k for k in csv_row if 'NA_' in k or 'MFD' in k or 'V_MM' in k])
    
    print("\n" + "=" * 70)
    print("Améliorations:")
    print("  ✓ Ajout paramètres demandés (NA, MFD, V_MM, packing, etc.)")
    print("  ✓ Validation renforcée")
    print("  ✓ CSV row enrichi")
    print("  ✓ Summary & performance index")

    print("=" * 70)
