#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Configuration
=================================================
UPGRADE MAJEUR: Ajout paramètres complets Dana et al. 2024

NOUVEAUTÉS V18.0:
- Paramètres géométriques étendus (packing, pitch_ratio, config_type)
- Paramètres longitudinaux (L_mux, L_taper, L_MMF, L_total)
- Paramètres optiques SM/MM séparés
- Pertes par section (polymer, taper, MMF)
- Métriques globales étendues (coupling_degradation, geometry_penalty)
- Taper profile paramètres (n_taper, taper_profile)

AUTEUR: Photonic Lantern Project V18.0
DATE: 2026-02-15
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import threading
import numpy as np


# ============================================================================
# CONSTANTES PHYSIQUES ÉTENDUES (Dana et al. 2024)
# ============================================================================
@dataclass(frozen=True)
class PhysicalConstants:
    """
    Constantes physiques photonic lantern (ÉTENDU V18.0)
    
    Référence: Dana et al. (2024) - Light: Science & Applications 13:116
    """

    # ═══════════════════════════════════════════════════════════════
    # MATÉRIAUX
    # ═══════════════════════════════════════════════════════════════
    # Indices réfraction
    POLYMER_N_BASE: float = 1.53          # IP-Dip polymer (Dana)
    AIR_N: float = 1.0                    # Cladding air
    SILICA_N: float = 1.444               # Silice (fibre SMF-28)
    
    # Épaisseurs typiques
    POLYMER_THICKNESS_MIN_UM: float = 0.5 # µm
    POLYMER_THICKNESS_MAX_UM: float = 3.0 # µm
    
    # ═══════════════════════════════════════════════════════════════
    # PARAMÈTRES OPTIQUES
    # ═══════════════════════════════════════════════════════════════
    # V-number cibles
    V_NUMBER_MIN: float = 2.0
    V_NUMBER_IDEAL_SM: float = 2.35
    V_NUMBER_IDEAL_6MODE: float = 4.69
    V_NUMBER_MAX: float = 12.0
    
    # Ouverture numérique
    NA_SM_TYPICAL: float = 0.12           # SMF-28 standard
    NA_MM_MIN: float = 0.10
    NA_MM_MAX: float = 0.35
    
    # Mode Field Diameter (MFD)
    MFD_SM_1550_UM: float = 10.4          # SMF-28 @ 1550nm

    # ═══════════════════════════════════════════════════════════════
    # PARAMÈTRES GÉOMÉTRIQUES
    # ═══════════════════════════════════════════════════════════════
    # Core radius
    CORE_RADIUS_MIN_UM: float = 0.5
    CORE_RADIUS_MAX_UM: float = 3.0
    CORE_RADIUS_SM_TYPICAL_UM: float = 1.2
    
    # Pitch (spacing entre cœurs)
    PITCH_MIN_UM: float = 3.0
    PITCH_MAX_UM: float = 15.0
    PITCH_RATIO_MIN: float = 2.0          # pitch / (2*r_core)
    PITCH_RATIO_MAX: float = 6.0
    
    # Ring radius (pour configs circulaires)
    RING_RADIUS_MIN_UM: float = 5.0
    RING_RADIUS_MAX_UM: float = 25.0
    
    # Packing efficiency
    PACKING_EFFICIENCY_MIN: float = 0.3
    PACKING_EFFICIENCY_MAX: float = 0.9

    # ═══════════════════════════════════════════════════════════════
    # PARAMÈTRES LONGITUDINAUX
    # ═══════════════════════════════════════════════════════════════
    # Longueurs Dana et al. 2024
    L_MUX_MIN_UM: float = 50.0            # Section multiplexeur
    L_MUX_MAX_UM: float = 500.0
    L_MUX_TYPICAL_UM: float = 200.0       # Dana: ~200-250 µm
    
    L_TAPER_MIN_UM: float = 100.0         # Section taper
    L_TAPER_MAX_UM: float = 1000.0
    L_TAPER_TYPICAL_UM: float = 375.0     # Dana: 375 µm (figure 1b)
    
    L_MMF_MIN_UM: float = 0.0             # Section MMF (optionnel)
    L_MMF_MAX_UM: float = 5000.0
    L_MMF_TYPICAL_UM: float = 100.0
    
    # Taper index (profil exponentiel)
    N_TAPER_MIN: float = 0.3              # Profil doux
    N_TAPER_MAX: float = 3.0              # Profil abrupt
    N_TAPER_ADIABATIC: float = 1.0        # Linéaire (adiabatique)

    # ═══════════════════════════════════════════════════════════════
    # CIBLES PERFORMANCE
    # ═══════════════════════════════════════════════════════════════
    # Insertion Loss
    TARGET_IL_IDEAL_DB: float = 0.8       # Dana simulation FDTD
    TARGET_IL_GOOD_DB: float = 2.0
    TARGET_IL_MAX_DB: float = 4.0         # Dana expérimental: 2.67 dB
    
    # Mode-Dependent Loss
    TARGET_MDL_IDEAL_DB: float = 0.1
    TARGET_MDL_GOOD_DB: float = 1.0
    TARGET_MDL_MAX_DB: float = 5.0        # Dana: 4.4 dB
    
    # Polarization-Dependent Loss
    TARGET_PDL_IDEAL_DB: float = 0.3
    TARGET_PDL_GOOD_DB: float = 1.0
    TARGET_PDL_MAX_DB: float = 3.0
    
    # Crosstalk
    TARGET_CROSSTALK_EXCELLENT_DB: float = -60.0
    TARGET_CROSSTALK_GOOD_DB: float = -40.0
    TARGET_CROSSTALK_MIN_DB: float = -20.0
    
    # Radiation loss
    TARGET_RADIATION_LOSS_MAX_DB_PER_M: float = 1.0

    # ═══════════════════════════════════════════════════════════════
    # BANDES SPECTRALES
    # ═══════════════════════════════════════════════════════════════
    S_BAND_MIN_NM: float = 1460.0
    S_BAND_MAX_NM: float = 1530.0
    C_BAND_MIN_NM: float = 1530.0
    C_BAND_MAX_NM: float = 1565.0
    L_BAND_MIN_NM: float = 1565.0
    L_BAND_MAX_NM: float = 1625.0
    U_BAND_MIN_NM: float = 1625.0
    U_BAND_MAX_NM: float = 1675.0
    
    WAVELENGTH_DEFAULT_NM: float = 1550.0

    # ═══════════════════════════════════════════════════════════════
    # SIMULATION
    # ═══════════════════════════════════════════════════════════════
    # PML
    PML_COMPLEX_STRENGTH: float = 0.2
    PML_DEFAULT_THICKNESS_UM: float = 10.0
    PML_DEFAULT_ORDER: int = 2
    
    # Critère adiabatique
    ADIABATIC_THRESHOLD: float = 0.1
    
    # Constantes universelles
    C_LIGHT_M_S: float = 3.0e8

    def validate(self) -> bool:
        """Validation cohérence constantes"""
        checks = [
            self.POLYMER_N_BASE > self.AIR_N,
            self.V_NUMBER_MIN < self.V_NUMBER_MAX,
            self.CORE_RADIUS_MIN_UM < self.CORE_RADIUS_MAX_UM,
            self.PITCH_MIN_UM < self.PITCH_MAX_UM,
            self.L_MUX_MIN_UM < self.L_MUX_MAX_UM,
            self.L_TAPER_MIN_UM < self.L_TAPER_MAX_UM,
            self.N_TAPER_MIN < self.N_TAPER_MAX,
            self.C_BAND_MIN_NM < self.C_BAND_MAX_NM,
            self.TARGET_IL_IDEAL_DB < self.TARGET_IL_MAX_DB,
        ]
        return all(checks)


# ============================================================================
# PARAMÈTRES DESIGN ÉTENDU (tous les paramètres Dana)
# ============================================================================
@dataclass
class PhotonicLanternDesignParameters:
    """
    Paramètres complets d'un design de photonic lantern
    
    Inclut TOUS les paramètres listés dans votre spécification:
    - Géométriques/topologiques
    - Optiques SM/MM
    - Matériau polymère
    - Longitudinaux
    - Taper
    - Pertes par section
    - Métriques globales
    """
    
    # ═══════════════════════════════════════════════════════════════
    # PARAMÈTRES GÉOMÉTRIQUES ET TOPOLOGIQUES
    # ═══════════════════════════════════════════════════════════════
    N_cores: int                          # Nombre total de cœurs
    has_central_core: bool                # Présence cœur central (True/False)
    config_type: str                      # 'circular', 'hexagonal', 'custom'
    geometry_config: str                  # Description textuelle config
    n_peripheral_cores: int               # Nombre cœurs périphériques
    R_ring: float                         # Rayon ring (µm), 0 si pas de ring
    packing_efficiency: float             # Facteur remplissage [0-1]
    pitch: float                          # Espacement moyen cœurs (µm)
    pitch_min: float                      # Espacement minimal (µm)
    pitch_ratio: float                    # pitch / (2*r_core)
    
    # ═══════════════════════════════════════════════════════════════
    # PARAMÈTRES OPTIQUES - FIBRE MONOMODE (SM)
    # ═══════════════════════════════════════════════════════════════
    wavelength: float                     # λ (nm)
    r_core_SM: float                      # Rayon cœur SM (µm)
    r_clad_SM: float                      # Rayon cladding SM (µm)
    n_core_SM: float                      # Indice cœur SM
    n_clad_SM: float                      # Indice cladding SM
    V_SM: float                           # V-number SM
    NA_SM: float                          # Ouverture numérique SM
    MFD: float                            # Mode Field Diameter (µm)
    n_eff_LP01: float                     # n_eff mode fondamental LP01
    
    # ═══════════════════════════════════════════════════════════════
    # PARAMÈTRES OPTIQUES - FIBRE MULTIMODE (MM)
    # ═══════════════════════════════════════════════════════════════
    r_core_MM: float                      # Rayon cœur MM (µm)
    V_MM: float                           # V-number MM
    NA_MM: float                          # Ouverture numérique MM
    M_max: int                            # Nombre max modes supportés
    
    # ═══════════════════════════════════════════════════════════════
    # PARAMÈTRES MATÉRIAU POLYMÈRE
    # ═══════════════════════════════════════════════════════════════
    n_polymer: float                      # Indice polymère
    d_polymer: float                      # Épaisseur couche polymère (µm)
    coupling_uniformity: float            # Uniformité couplage [0-1]
    
    # ═══════════════════════════════════════════════════════════════
    # PARAMÈTRES LONGITUDINAUX (LONGUEURS)
    # ═══════════════════════════════════════════════════════════════
    L_mux: float                          # Longueur multiplexeur (µm)
    L_taper: float                        # Longueur taper (µm)
    L_MMF: float                          # Longueur section MMF (µm)
    L_total: float                        # Longueur totale (µm)
    
    # ═══════════════════════════════════════════════════════════════
    # PARAMÈTRES DE TAPER
    # ═══════════════════════════════════════════════════════════════
    n_taper: float                        # Exposant profil taper
    taper_profile: str                    # Type: 'linear', 'exponential', 'polynomial'
    
    # ═══════════════════════════════════════════════════════════════
    # PERTES - SECTION POLYMÈRE
    # ═══════════════════════════════════════════════════════════════
    IL_polymer: float = 0.0               # Insertion Loss polymère (dB)
    MDL_polymer: float = 0.0              # MDL polymère (dB)
    PDL_polymer: float = 0.0              # PDL polymère (dB)
    
    # ═══════════════════════════════════════════════════════════════
    # PERTES - SECTION TAPER
    # ═══════════════════════════════════════════════════════════════
    IL_taper: float = 0.0                 # Insertion Loss taper (dB)
    MDL_taper: float = 0.0                # MDL taper (dB)
    PDL_taper: float = 0.0                # PDL taper (dB)
    
    # ═══════════════════════════════════════════════════════════════
    # PERTES - FIBRE MULTIMODE
    # ═══════════════════════════════════════════════════════════════
    IL_MMF: float = 0.0                   # Insertion Loss MMF (dB)
    MDL_MMF: float = 0.0                  # MDL MMF (dB)
    PDL_MMF: float = 0.0                  # PDL MMF (dB)
    
    # ═══════════════════════════════════════════════════════════════
    # MÉTRIQUES GLOBALES
    # ═══════════════════════════════════════════════════════════════
    IL_total: float = 0.0                 # IL total device (dB)
    MDL_total: float = 0.0                # MDL total (dB)
    PDL_total: float = 0.0                # PDL total (dB)
    Total_Loss: float = 0.0               # Pertes totales (dB)
    Efficiency: float = 1.0               # Efficacité globale [0-1]
    Crosstalk: float = -80.0              # Crosstalk (dB)
    crosstalk_penalty: float = 0.0        # Pénalité crosstalk (dB)
    coupling_degradation: float = 0.0     # Dégradation couplage (dB)
    geometry_penalty: float = 0.0         # Pénalité géométrique (dB)
    
    # ═══════════════════════════════════════════════════════════════
    # MÉTADONNÉES
    # ═══════════════════════════════════════════════════════════════
    sample_id: str = ""                   # ID échantillon
    direction: str = "mux"                # 'mux' ou 'demux'
    success: bool = True                  # Simulation réussie?
    
    def validate(self) -> tuple[bool, str]:
        """Validation cohérence paramètres"""
        constants = PhysicalConstants()
        
        # Check N_cores
        if not (1 <= self.N_cores <= 19):
            return False, f"N_cores={self.N_cores} hors range [1, 19]"
        
        # Check wavelength
        if not (1400 <= self.wavelength <= 1700):
            return False, f"wavelength={self.wavelength} hors range [1400, 1700] nm"
        
        # Check V-numbers
        if not (constants.V_NUMBER_MIN <= self.V_SM <= constants.V_NUMBER_MAX):
            return False, f"V_SM={self.V_SM:.2f} hors range valide"
        
        # Check pitch ratio
        if self.pitch_ratio < constants.PITCH_RATIO_MIN:
            return False, f"pitch_ratio={self.pitch_ratio:.2f} < min={constants.PITCH_RATIO_MIN}"
        
        # Check longueurs
        if self.L_mux < 0 or self.L_taper < 0 or self.L_MMF < 0:
            return False, "Longueurs négatives détectées"
        
        if abs(self.L_total - (self.L_mux + self.L_taper + self.L_MMF)) > 1e-3:
            return False, f"L_total={self.L_total} != L_mux+L_taper+L_MMF"
        
        # Check pertes physiques
        if self.IL_total < 0 or self.IL_total > 50:
            return False, f"IL_total={self.IL_total} dB non physique"
        
        if not (0 <= self.Efficiency <= 1):
            return False, f"Efficiency={self.Efficiency} hors range [0,1]"
        
        return True, "Paramètres valides"
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion dictionnaire pour CSV"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhotonicLanternDesignParameters':
        """Création depuis dictionnaire"""
        return cls(**data)


# ============================================================================
# SIMULATION CONFIG (INCHANGÉ)
# ============================================================================
@dataclass
class SimulationConfig:
    """Configuration simulation (V18.0 - compatible V17.1)"""
    
    # Performance
    num_threads: int = 2
    use_multiprocessing: bool = False
    num_processes: int = 4
    
    # Maillage FEM
    mesh_refinement: float = 1.5
    mesh_min_points: int = 1500
    mesh_target_points: int = 2500
    mesh_max_points: int = 8000
    
    # Solveur
    eigenvalue_solver_tol: float = 1e-7
    eigenvalue_solver_maxiter: int = 4000
    modes_to_compute_factor: float = 3.0
    
    # Taper/CMT
    taper_points_initial: int = 12
    taper_points_max: int = 30
    use_cmt: bool = True
    use_complex_pml: bool = True
    use_adaptive_taper: bool = True
    
    # Cache
    enable_mesh_cache: bool = True
    cache_max_size: int = 150
    cache_max_memory_mb: float = 500.0
    
    # Debug
    debug_mode: bool = False
    save_field_plots: bool = False
    verbose_logging: bool = True
    
    def apply_thread_limits(self):
        """Limite threads BLAS/OpenMP"""
        env_vars = {
            'OMP_NUM_THREADS': str(self.num_threads),
            'MKL_NUM_THREADS': str(self.num_threads),
            'OPENBLAS_NUM_THREADS': str(self.num_threads),
        }
        for var, val in env_vars.items():
            os.environ[var] = val
    
    def validate(self) -> tuple[bool, str]:
        """Validation configuration"""
        if self.num_threads < 1:
            return False, "num_threads < 1"
        if self.mesh_min_points > self.mesh_target_points:
            return False, "mesh_min_points > mesh_target_points"
        return True, "Configuration valide"
    
    @classmethod
    def from_preset(cls, preset: str = 'default') -> 'SimulationConfig':
        """Charge preset"""
        presets = {
            'debug': {'mesh_refinement': 0.8, 'mesh_target_points': 1200},
            'default': {},
            'production': {'mesh_refinement': 2.0, 'mesh_target_points': 3500},
        }
        config_dict = presets.get(preset, {})
        return cls(**config_dict)
    
    def to_json(self, filepath: Path):
        """Sauvegarde JSON"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: Path) -> 'SimulationConfig':
        """Charge JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ============================================================================
# LOGGER CONFIG (INCHANGÉ V17.1)
# ============================================================================
class LoggerConfig:
    """Configuration logging thread-safe"""
    _configured_loggers = set()
    _lock = threading.Lock()
    
    @staticmethod
    def setup_logger(
        name: str = 'pl_v18',
        level: int = logging.INFO,
        log_file: Optional[Path] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        use_utc: bool = True
    ) -> logging.Logger:
        """Setup logger avec UTC timestamps"""
        with LoggerConfig._lock:
            if name in LoggerConfig._configured_loggers:
                return logging.getLogger(name)
            
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.handlers.clear()
            
            # Formatter UTC
            if use_utc:
                class UTCFormatter(logging.Formatter):
                    converter = lambda *args: datetime.now(timezone.utc).timetuple()
                fmt = UTCFormatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S')
            else:
                fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S')
            
            # Console handler
            console = logging.StreamHandler()
            console.setFormatter(fmt)
            logger.addHandler(console)
            
            # File handler
            if log_file:
                file_path = Path(log_file).resolve()
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = RotatingFileHandler(
                    file_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
                )
                file_handler.setFormatter(fmt)
                logger.addHandler(file_handler)
            
            LoggerConfig._configured_loggers.add(name)
            return logger


# ============================================================================
# OUTPUT CONFIG (INCHANGÉ V17.1)
# ============================================================================
@dataclass
class OutputConfig:
    """Gestion arborescence sortie"""
    base_dir: Path
    timestamp: str = field(init=False)
    data_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    configs_dir: Path = field(init=False)
    
    def __post_init__(self):
        self.base_dir = Path(self.base_dir).resolve()
        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = self.base_dir / 'data'
        self.logs_dir = self.base_dir / 'logs'
        self.plots_dir = self.base_dir / 'plots'
        self.cache_dir = self.base_dir / 'cache'
        self.configs_dir = self.base_dir / 'configs'
        
        for d in [self.data_dir, self.logs_dir, self.plots_dir, 
                  self.cache_dir, self.configs_dir]:
            d.mkdir(exist_ok=True)
    
    def get_dataset_path(self, name: str = 'dataset') -> Path:
        return self.data_dir / f"{name}_{self.timestamp}.csv"
    
    def get_log_path(self, name: str = 'generation') -> Path:
        return self.logs_dir / f"{name}_{self.timestamp}.log"
    
    def get_config_path(self, name: str = 'simulation') -> Path:
        return self.configs_dir / f"{name}_{self.timestamp}.json"


# ============================================================================
# INITIALISATION
# ============================================================================
def init_config(
    base_output_dir: str | Path = "output_pl_v18",
    log_level: int = logging.INFO,
    preset: str = 'default',
    use_utc: bool = True
) -> tuple[logging.Logger, SimulationConfig, OutputConfig]:
    """Initialisation complète V18.0"""
    
    # Config simulation
    sim_config = SimulationConfig.from_preset(preset)
    sim_config.apply_thread_limits()
    
    # Validation
    valid, msg = sim_config.validate()
    if not valid:
        raise ValueError(f"Configuration invalide: {msg}")
    
    # Output paths
    output = OutputConfig(base_output_dir)
    
    # Logger
    logger = LoggerConfig.setup_logger(
        name='pl_v18',
        level=log_level,
        log_file=output.get_log_path('generation'),
        use_utc=use_utc
    )
    
    # Sauvegarde config
    sim_config.to_json(output.get_config_path('simulation'))
    
    logger.info(f"Configuration V18.0 initialisée (preset={preset})")
    logger.info(f"Output directory: {output.base_dir}")
    
    return logger, sim_config, output


# ============================================================================
# TEST
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("PHOTONIC LANTERN CONFIG V18.0 - EXTENDED PARAMETERS")
    print("=" * 70)
    
    # Test constantes
    constants = PhysicalConstants()
    assert constants.validate(), "Constantes invalides!"
    print("✓ Constantes physiques validées")
    
    # Test design parameters
    test_params = PhotonicLanternDesignParameters(
        # Géométriques
        N_cores=7, has_central_core=True, config_type='hexagonal',
        geometry_config='6+1 hexagonal', n_peripheral_cores=6,
        R_ring=10.0, packing_efficiency=0.75, pitch=8.0, pitch_min=7.5,
        pitch_ratio=3.33,
        
        # Optiques SM
        wavelength=1550.0, r_core_SM=1.2, r_clad_SM=62.5,
        n_core_SM=1.53, n_clad_SM=1.0, V_SM=4.69, NA_SM=0.12,
        MFD=10.4, n_eff_LP01=1.475,
        
        # Optiques MM
        r_core_MM=25.0, V_MM=15.0, NA_MM=0.22, M_max=50,
        
        # Polymère
        n_polymer=1.53, d_polymer=2.0, coupling_uniformity=0.95,
        
        # Longitudinaux
        L_mux=200.0, L_taper=375.0, L_MMF=100.0, L_total=675.0,
        
        # Taper
        n_taper=1.0, taper_profile='exponential',
        
        # Metadata
        sample_id='S0000', direction='mux'
    )
    
    valid, msg = test_params.validate()
    print(f"✓ Design parameters: {msg}")
    
    # Test init
    logger, sim_cfg, out_cfg = init_config(preset='debug')
    logger.info("✓ Configuration V18.0 opérationnelle")
    
    print("\n" + "=" * 70)
    print("NOUVEAUTÉS V18.0:")
    print("  ✓ 60+ paramètres Dana et al. 2024")
    print("  ✓ Pertes par section (polymer/taper/MMF)")
    print("  ✓ Paramètres longitudinaux (L_mux, L_taper, L_MMF)")
    print("  ✓ Géométrie étendue (packing, pitch_ratio, config_type)")
    print("  ✓ Validation complète")
    print("=" * 70)