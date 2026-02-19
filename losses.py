#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Loss Calculations for Photonic Lantern 
=====================================================
CALCUL DE PERTES PAR SECTION (polymer, taper, MMF)

NOUVEAUTÉS:
- Pertes sectionnées: IL/MDL/PDL pour polymer, taper, MMF
- Calcul L_mux, L_taper, L_MMF depuis géométrie
- Métriques globales: coupling_degradation, geometry_penalty
- Profil taper paramétrable (n_taper, taper_profile)

AUTEUR: Photonic Lantern Project V18.0
DATE: 2026-02-15
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger('pl_v18.losses')


class EnhancedLossCalculator:
    """
    Calculateur de pertes étendu avec sections
    
    Sections:
    1. POLYMER: Section multiplexeur (fibres SM → géométrie groupée)
    2. TAPER: Transition adiabatique (Polymer → MMF)
    3. MMF: Section fibre multimode (sortie)
    """
    
    REQUIRED_MODE_KEYS = {'n_eff', 'beta', 'confinement', 'field_vector'}
    
    @staticmethod
    def calculate_sectional_losses(
        modes: List[Dict],
        geometry,
        design_params,  # PhotonicLanternDesignParameters
        direction: str = 'mux',
        wavelength_nm: float = 1550.0
    ) -> Dict:
        """
        Calcul pertes par section + métriques globales
        
        Args:
            modes: Modes calculés par FEM
            geometry: Géométrie device
            design_params: Paramètres design complets
            direction: 'mux' ou 'demux'
            wavelength_nm: Longueur d'onde (nm)
            
        Returns:
            Dict avec toutes les métriques de pertes
        """
        
        if not modes:
            logger.warning("Liste modes vide")
            return {'success': False, 'error': 'no modes'}
        
        logger.debug(f"Calcul pertes sectionnées: {len(modes)} modes, λ={wavelength_nm:.1f} nm")
        
        try:
            # ═══════════════════════════════════════════════════════════
            # SECTION 1: POLYMER (Multiplexeur)
            # ═══════════════════════════════════════════════════════════
            polymer_losses = EnhancedLossCalculator._calculate_polymer_section(
                modes, geometry, design_params, wavelength_nm
            )
            
            # ═══════════════════════════════════════════════════════════
            # SECTION 2: TAPER (Transition)
            # ═══════════════════════════════════════════════════════════
            taper_losses = EnhancedLossCalculator._calculate_taper_section(
                modes, geometry, design_params, wavelength_nm
            )
            
            # ═══════════════════════════════════════════════════════════
            # SECTION 3: MMF (Fibre multimode)
            # ═══════════════════════════════════════════════════════════
            mmf_losses = EnhancedLossCalculator._calculate_mmf_section(
                modes, geometry, design_params, wavelength_nm
            )
            
            # ═══════════════════════════════════════════════════════════
            # MÉTRIQUES GLOBALES
            # ═══════════════════════════════════════════════════════════
            global_metrics = EnhancedLossCalculator._calculate_global_metrics(
                polymer_losses, taper_losses, mmf_losses, modes, geometry, design_params
            )
            
            # ═══════════════════════════════════════════════════════════
            # COMPILATION RÉSULTATS
            # ═══════════════════════════════════════════════════════════
            result = {
                # Section polymer
                'IL_polymer': polymer_losses['IL'],
                'MDL_polymer': polymer_losses['MDL'],
                'PDL_polymer': polymer_losses['PDL'],
                
                # Section taper
                'IL_taper': taper_losses['IL'],
                'MDL_taper': taper_losses['MDL'],
                'PDL_taper': taper_losses['PDL'],
                
                # Section MMF
                'IL_MMF': mmf_losses['IL'],
                'MDL_MMF': mmf_losses['MDL'],
                'PDL_MMF': mmf_losses['PDL'],   # BUG FIX: était mmf_losses['MDL']
                
                # Métriques globales
                'IL_total': global_metrics['IL_total'],
                'MDL_total': global_metrics['MDL_total'],
                'PDL_total': global_metrics['PDL_total'],
                'Total_Loss': global_metrics['Total_Loss'],
                'Efficiency': global_metrics['Efficiency'],
                'Crosstalk': global_metrics['Crosstalk'],
                'crosstalk_penalty': global_metrics['crosstalk_penalty'],
                'coupling_degradation': global_metrics['coupling_degradation'],
                'geometry_penalty': global_metrics['geometry_penalty'],
                
                # Métriques auxiliaires
                'radiation_loss_dB_per_m': global_metrics['radiation_loss_dB_per_m'],
                'avg_confinement': global_metrics['avg_confinement'],
                'n_modes_used': len(modes),
                'direction': direction,
                'wavelength_nm': float(wavelength_nm),
                'success': True
            }
            
            # Log résumé
            logger.info(f"Pertes → Polymer: IL={result['IL_polymer']:.2f} MDL={result['MDL_polymer']:.2f} PDL={result['PDL_polymer']:.2f} | "
                       f"Taper: IL={result['IL_taper']:.2f} MDL={result['MDL_taper']:.2f} PDL={result['PDL_taper']:.2f} | "
                       f"TOTAL: IL={result['IL_total']:.2f} MDL={result['MDL_total']:.2f} PDL={result['PDL_total']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur calcul pertes sectionnées: {e}")
            return {'error': str(e), 'success': False}
    
    # ═════════════════════════════════════════════════════════════════════
    # SECTION 1: POLYMER (Multiplexeur)
    # ═════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _calculate_polymer_section(
        modes: List[Dict],
        geometry,
        design_params,
        wavelength_nm: float
    ) -> Dict:
        """
        Pertes section polymère (multiplexeur SM → bundle)
        
        Cette section inclut:
        - Couplage SM fibers → polymer cores
        - Propagation dans section droite (L_mux)
        - Début de regroupement spatial
        """
        
        n_cores = design_params.N_cores
        L_mux_um = design_params.L_mux
        
        # IL: Pertes couplage SM → polymer
        confs = np.array([m['confinement'] for m in modes])
        avg_conf = np.mean(confs[confs > 0.01]) if np.any(confs > 0.01) else 0.01
        
        # Pertes de couplage basées sur mismatch géométrique
        coupling_mismatch = 0.5 * (1 - design_params.coupling_uniformity)  # dB
        
        # Pertes confinement
        loss_conf = -10 * np.log10(avg_conf)
        
        # Pertes propagation dans polymère (absorption + diffusion)
        alpha_polymer = 0.5  # dB/m (typique IP-Dip)
        loss_propagation = alpha_polymer * (L_mux_um * 1e-6)
        
        IL_polymer = coupling_mismatch + loss_conf + loss_propagation
        
        # MDL: Variation confinement entre modes
        if len(confs) >= 2:
            conf_min = np.min(confs)
            conf_max = np.max(confs)
            MDL_polymer = -10 * np.log10(conf_min / (conf_max + 1e-12))
            MDL_polymer += 3 * np.std(confs)  # Contribution uniformité
        else:
            MDL_polymer = 0.0
        
        # PDL: Basé sur asymétrie géométrique
        PDL_polymer = EnhancedLossCalculator._calculate_pdl_realistic(
            modes, geometry, wavelength_nm
        )
        
        logger.debug(f"Polymer section: L={L_mux_um:.1f}µm, IL={IL_polymer:.2f}dB, MDL={MDL_polymer:.2f}dB")
        
        return {
            'IL': float(np.clip(IL_polymer, 0.0, 10.0)),
            'MDL': float(np.clip(MDL_polymer, 0.0, 5.0)),
            'PDL': float(np.clip(PDL_polymer, 0.2, 3.0))
        }
    
    # ═════════════════════════════════════════════════════════════════════
    # SECTION 2: TAPER (Transition)
    # ═════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _calculate_taper_section(
        modes: List[Dict],
        geometry,
        design_params,
        wavelength_nm: float
    ) -> Dict:
        """
        Pertes section taper (transition adiabatique)
        
        Cette section inclut:
        - Transition géométrique (pitch reduction)
        - Couplage inter-modal évolutif
        - Pertes par radiation (modes cutoff progressif)
        """
        
        L_taper_um = design_params.L_taper
        n_taper    = design_params.n_taper

        # ── IL taper : critère adiabatique (Snyder & Love 1983) ──────────
        # Physique : un taper long est PLUS adiabatique → MOINS de pertes.
        # η_adiabatic = 1 - exp(-L / L_beat)  [fraction d'énergie couplée]
        # IL_coupling = -10 log10(η)
        # L_beat ≈ longueur de battement inter-modes typique PL polymère
        #   ≈ 2π/Δβ, Δβ ≈ k0×Δn_eff ≈ 4.0×0.03 ≈ 0.12 µm⁻¹ → L_beat ≈ 50 µm
        # En pratique pour des PL polymer/air : L_beat_eff = 100-200 µm
        # On prend L_beat = 150 µm (valeur médiane, cohérente avec Dana 2024)

        L_beat = 150.0   # µm
        eta_adiabatic  = 1.0 - np.exp(-L_taper_um / (L_beat * max(n_taper, 0.5)))
        IL_coupling    = -10 * np.log10(max(eta_adiabatic, 1e-6))

        # Absorption matériau dans le taper (IP-Dip : ~0.5 dB/m)
        alpha_dB_per_m = 0.5
        IL_propagation = alpha_dB_per_m * (L_taper_um * 1e-6)

        # Pertes radiation résiduelles (modes de gaine coupés progressivement)
        # Faibles si confinement élevé ; scaling logarithmique avec n_modes
        confs = np.array([m['confinement'] for m in modes])
        n_modes_val    = len(modes)
        conf_mean      = float(np.mean(confs)) if len(confs) else 0.9
        IL_radiation   = max(0.0, 1.0 - conf_mean) * 0.5 + 0.05 * np.log10(n_modes_val + 1)

        IL_taper = IL_coupling + IL_propagation + IL_radiation

        # MDL taper : modes d'ordre élevé plus sensibles à la transition
        if len(confs) >= 2:
            sorted_confs    = np.sort(confs)
            low_order_conf  = np.mean(sorted_confs[-3:])   # top 3 (LP01-like)
            high_order_conf = np.mean(sorted_confs[:3])    # bottom 3 (LP11-like)
            MDL_taper = float(np.clip(
                -10 * np.log10(high_order_conf / (low_order_conf + 1e-12)),
                0.0, 3.0))
        else:
            MDL_taper = 0.0

        # PDL taper : biréfringence induite, minimale pour profil linéaire (n=1)
        PDL_taper = 0.05 + 0.03 * abs(n_taper - 1.0)

        logger.debug(
            f"Taper section: L={L_taper_um:.0f}µm, n={n_taper:.2f} | "
            f"IL_coupl={IL_coupling:.3f} IL_prop={IL_propagation:.4f} "
            f"IL_rad={IL_radiation:.3f} | IL={IL_taper:.3f}dB"
        )
        
        return {
            'IL': float(np.clip(IL_taper, 0.0, 8.0)),
            'MDL': float(np.clip(MDL_taper, 0.0, 3.0)),
            'PDL': float(np.clip(PDL_taper, 0.05, 2.0))
        }
    
    # ═════════════════════════════════════════════════════════════════════
    # SECTION 3: MMF (Fibre Multimode)
    # ═════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _calculate_mmf_section(
        modes: List[Dict],
        geometry,
        design_params,
        wavelength_nm: float
    ) -> Dict:
        """
        Pertes section MMF (sortie)
        
        Cette section inclut:
        - Propagation dans fibre MM
        - Pertes différentielles entre groupes de modes
        - Couplage résiduel entre modes LP
        """
        
        L_MMF_um = design_params.L_MMF
        
        if L_MMF_um < 1.0:
            # Pas de section MMF
            return {'IL': 0.0, 'MDL': 0.0, 'PDL': 0.0}
        
        # IL: Pertes propagation MMF
        alpha_mmf = 0.2  # dB/km (silice)
        IL_MMF = alpha_mmf * (L_MMF_um * 1e-6 * 1e3)  # Conversion µm → km
        
        # Pertes de raccordement taper → MMF
        mismatch_loss = 0.3  # dB (typique)
        IL_MMF += mismatch_loss
        
        # MDL: Minimal dans MMF bien conçue
        MDL_MMF = 0.05  # dB (négligeable si MMF grande)
        
        # PDL: Biréfringence fibre
        PDL_MMF = 0.1  # dB (typique fibre circulaire)
        
        logger.debug(f"MMF section: L={L_MMF_um:.1f}µm, IL={IL_MMF:.2f}dB")
        
        return {
            'IL': float(np.clip(IL_MMF, 0.0, 5.0)),
            'MDL': float(np.clip(MDL_MMF, 0.0, 1.0)),
            'PDL': float(np.clip(PDL_MMF, 0.05, 0.5))
        }
    
    # ═════════════════════════════════════════════════════════════════════
    # MÉTRIQUES GLOBALES
    # ═════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _calculate_global_metrics(
        polymer_losses: Dict,
        taper_losses: Dict,
        mmf_losses: Dict,
        modes: List[Dict],
        geometry,
        design_params
    ) -> Dict:
        """Calcul métriques globales cumulées"""
        
        # Pertes totales (somme sections)
        IL_total = polymer_losses['IL'] + taper_losses['IL'] + mmf_losses['IL']
        MDL_total = np.sqrt(polymer_losses['MDL']**2 + taper_losses['MDL']**2 + mmf_losses['MDL']**2)
        PDL_total = np.sqrt(polymer_losses['PDL']**2 + taper_losses['PDL']**2 + mmf_losses['PDL']**2)
        
        # Total loss (include everything)
        Total_Loss = IL_total
        
        # Efficiency
        Efficiency = 10**(-IL_total / 10)
        
        # Crosstalk
        Crosstalk = EnhancedLossCalculator._calculate_crosstalk(modes)
        
        # Crosstalk penalty
        crosstalk_penalty = max(0, -20 - Crosstalk) * 0.1  # Penalty if XT > -20dB
        
        # ÉTAPE 3 FIX: coupling_degradation depuis modes FEM réels
        # Physique : dispersion confinements + spread n_eff = dégradation couplage
        if len(modes) >= 2:
            confs_arr    = np.array([m["confinement"] for m in modes])
            n_effs_arr   = np.array([float(m["n_eff"]) for m in modes])
            cv_conf      = float(np.std(confs_arr) / (np.mean(confs_arr) + 1e-9))
            n_core_val   = getattr(geometry, "core_index", getattr(geometry, "n_core", 1.53))
            n_clad_val   = getattr(geometry, "clad_index", getattr(geometry, "n_clad", 1.0))
            delta_n_val  = max(n_core_val - n_clad_val, 1e-6)
            n_eff_spread = float(np.ptp(n_effs_arr) / delta_n_val)
            conf_min_pen = float(max(0.0, 0.70 - float(np.min(confs_arr))))
            coupling_degradation = float(np.clip(
                cv_conf * 1.5 + n_eff_spread * 0.8 + conf_min_pen * 2.0, 0.0, 5.0))
        else:
            coupling_degradation = 5.0  # Pas de MUX possible < 2 modes
        
        # Geometry penalty
        # Basé sur packing efficiency et pitch_ratio
        packing = design_params.packing_efficiency
        pitch_ratio = design_params.pitch_ratio
        
        # Penalty si packing trop serré (< 0.5) ou trop lâche (> 0.85)
        if packing < 0.5:
            packing_penalty = (0.5 - packing) * 3.0
        elif packing > 0.85:
            packing_penalty = (packing - 0.85) * 2.0
        else:
            packing_penalty = 0.0
        
        # Penalty si pitch_ratio non optimal (optimal ~3-4)
        pitch_penalty = abs(pitch_ratio - 3.5) * 0.2
        
        geometry_penalty = packing_penalty + pitch_penalty
        
        # Radiation loss
        radiation_loss_dB_per_m = EnhancedLossCalculator._calculate_radiation_loss(
            modes, design_params.wavelength
        )
        
        # Average confinement
        confs = [m['confinement'] for m in modes if m['confinement'] > 0]
        avg_confinement = float(np.mean(confs)) if confs else 0.0
        
        return {
            'IL_total': float(np.clip(IL_total, 0.0, 40.0)),
            'MDL_total': float(np.clip(MDL_total, 0.0, 10.0)),
            'PDL_total': float(np.clip(PDL_total, 0.2, 8.0)),
            'Total_Loss': float(Total_Loss),
            'Efficiency': float(np.clip(Efficiency, 0.0, 1.0)),
            'Crosstalk': float(Crosstalk),
            'crosstalk_penalty': float(np.clip(crosstalk_penalty, 0.0, 5.0)),
            'coupling_degradation': float(np.clip(coupling_degradation, 0.0, 5.0)),
            'geometry_penalty': float(np.clip(geometry_penalty, 0.0, 5.0)),
            'radiation_loss_dB_per_m': float(radiation_loss_dB_per_m),
            'avg_confinement': float(avg_confinement)
        }
    
    # ═════════════════════════════════════════════════════════════════════
    # FONCTIONS AUXILIAIRES (PDL, Crosstalk, Radiation)
    # ═════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _calculate_pdl_realistic(modes: List[Dict], geometry, wavelength_nm: float) -> float:
        """PDL réaliste (copié de losses_v17_2_corrected.py)"""
        if len(modes) < 2:
            return 0.3
        
        n_effs = np.array([float(m['n_eff']) for m in modes])
        sorted_neff = np.sort(n_effs)[::-1]
        
        # Biréfringence
        degeneracy_gaps = []
        for i in range(len(sorted_neff) - 1):
            gap = abs(sorted_neff[i] - sorted_neff[i+1])
            if gap < 5e-4:
                degeneracy_gaps.append(gap)
        
        if degeneracy_gaps:
            mean_biref = np.mean(degeneracy_gaps)
            L_taper = 375e-6
            k0 = 2 * np.pi / (wavelength_nm * 1e-9)
            pdl_biref = 4.343 * k0 * mean_biref * L_taper
        else:
            delta_neff_global = np.ptp(n_effs)
            pdl_biref = delta_neff_global * 800
        
        # Asymétrie géométrique
        pdl_geom = 0.0
        if hasattr(geometry, 'positions') and len(geometry.positions) >= 3:
            pos = np.array(geometry.positions)
            center = np.mean(pos, axis=0)
            pos_centered = pos - center
            
            Ixx = np.sum(pos_centered[:, 0]**2)
            Iyy = np.sum(pos_centered[:, 1]**2)
            Ixy = np.sum(pos_centered[:, 0] * pos_centered[:, 1])
            
            I_max = (Ixx + Iyy) / 2 + np.sqrt(((Ixx - Iyy) / 2)**2 + Ixy**2)
            I_min = (Ixx + Iyy) / 2 - np.sqrt(((Ixx - Iyy) / 2)**2 + Ixy**2)
            
            asymmetry = abs(I_max - I_min) / (I_max + I_min + 1e-12)
            pdl_geom = asymmetry * 4.0
        
        # Coupling
        pdl_coupling = 0.15 * np.log10(len(modes) + 1)
        
        # Wavelength factor
        wavelength_factor = 1.0
        if wavelength_nm < 1530:
            wavelength_factor = 1 + (1530 - wavelength_nm) / 1000
        elif wavelength_nm > 1565:
            wavelength_factor = 1 + (wavelength_nm - 1565) / 1000
        
        # Confinement
        confs = np.array([m['confinement'] for m in modes])
        conf_spread = np.std(confs)
        pdl_conf = conf_spread * 2.0
        
        pdl_total = (pdl_biref + pdl_geom + pdl_coupling + pdl_conf) * wavelength_factor
        return float(np.clip(pdl_total, 0.2, 6.0))
    
    @staticmethod
    def _calculate_crosstalk_vectorial(modes: List[Dict]) -> float:
        """
        Crosstalk vectoriel calculé depuis P_x, P_y (modes TrueVectorialMaxwellSolver).
        Utilise l'inégalité des puissances inter-polarisations comme proxy d'overlap.
        """
        n = len(modes)
        if n < 2:
            return -80.0

        # Overlap proxy : similarité des ratios P_x/P_y entre modes dégénérés
        # (modes dégénérés LP11a/LP11b ont des ratios P_x/P_y très différents)
        max_overlap = 0.0
        for i in range(n):
            Px_i = modes[i].get('P_x', 1.0)
            Py_i = modes[i].get('P_y', 1.0)
            norm_i = np.sqrt(Px_i**2 + Py_i**2) + 1e-30
            for j in range(i+1, n):
                Px_j = modes[j].get('P_x', 1.0)
                Py_j = modes[j].get('P_y', 1.0)
                norm_j = np.sqrt(Px_j**2 + Py_j**2) + 1e-30
                # Cosinus entre vecteurs de Stokes simplifiés
                cos_sq = ((Px_i * Px_j + Py_i * Py_j) / (norm_i * norm_j))**2
                max_overlap = max(max_overlap, cos_sq)

        xt = -10 * np.log10(max_overlap + 1e-15)

        # Pénalité dégénérescence n_eff
        n_effs = np.sort([float(m['n_eff']) for m in modes])
        if len(n_effs) > 1:
            min_gap = np.min(np.diff(n_effs))
            if min_gap < 1e-4:
                xt -= 15.0 + (1e-4 - min_gap) * 1e6

        return float(np.clip(xt, -70.0, -10.0))

    @staticmethod
    def _calculate_crosstalk(modes: List[Dict]) -> float:
        """Crosstalk entre modes — route automatiquement selon type."""
        if not modes:
            return -80.0

        # Modes vectoriels : pas de field_vector → utiliser proxy polarisation
        if modes[0].get('is_vectorial', False):
            return LossCalculator._calculate_crosstalk_vectorial(modes)

        # Modes scalaires : field_vector disponible
        n = len(modes)
        if n < 2:
            return -80.0
        max_overlap = 0.0
        for i in range(n):
            Ei = modes[i].get('field_vector')
            if Ei is None:
                continue
            Pi = float(np.real(np.vdot(Ei, Ei)))
            if Pi < 1e-12:
                continue
            for j in range(i+1, n):
                Ej = modes[j].get('field_vector')
                if Ej is None:
                    continue
                Pj = float(np.real(np.vdot(Ej, Ej)))
                if Pj < 1e-12:
                    continue
                overlap = float(np.abs(np.vdot(Ei, Ej))**2 / (Pi * Pj + 1e-16))
                max_overlap = max(max_overlap, overlap)
        if max_overlap == 0.0:
            return -80.0
        xt = -10 * np.log10(max_overlap + 1e-15)
        n_effs = np.sort([float(m['n_eff']) for m in modes])
        if len(n_effs) > 1:
            min_gap = np.min(np.diff(n_effs))
            if min_gap < 1e-4:
                xt -= 15.0 + (1e-4 - min_gap) * 1e6
        return float(np.clip(xt, -70.0, -10.0))

    @staticmethod
    def _calculate_radiation_loss(modes: List[Dict], wavelength_nm: float) -> float:
        """Pertes radiatives"""
        rads = []
        wl_factor = 1550.0 / wavelength_nm
        
        for m in modes:
            conf = m['confinement']
            beta = m['beta']
            
            if np.iscomplexobj(beta) and abs(beta.imag) > 1e-9:
                alpha_db_m = 2 * abs(beta.imag) * 1e6 * 8.685889638 * wl_factor
                rads.append(alpha_db_m)
            else:
                penalty = max(0.0, 1.0 - conf) * 100.0
                if conf < 0.95:
                    penalty += (0.95 - conf) * 250
                rads.append(penalty)
        
        return float(np.mean(rads)) if rads else 0.0


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================
class LossCalculator(EnhancedLossCalculator):
    """Alias pour compatibilité V17.x"""
    
    @staticmethod
    def calculate_physical_losses(
        modes: List[Dict],
        geometry,
        direction: str = 'mux',
        wavelength_nm: float = 1550.0
    ) -> Dict:
        """
        Interface V17.x (backward compatible)

        FIX V18.1: Import corrigé (config_v18_enhanced → config.py)
        FIX V18.1: design_params enrichi depuis géométrie réelle
        FIX V18.2: Routage automatique vers VectorialLossCalculator si modes vectoriels
        """
        # ── FIX V18.2 : Routage vectoriel automatique ─────────────────────
        # Les modes produits par TrueVectorialMaxwellSolver ont is_vectorial=True
        # et contiennent Ex_dofs/Ey_dofs/P_x/P_y mais PAS field_vector.
        # LossCalculator (scalaire) tenterait d'accéder à 'field_vector' dans
        # _calculate_crosstalk → KeyError.  On route vers VectorialLossCalculator
        # qui utilise directement P_x/P_y/PDL_dB (PDL exact H-field).
        if modes and modes[0].get('is_vectorial', False):
            # Construire design_params minimal pour VectorialLossCalculator
            try:
                from config import PhotonicLanternDesignParameters
            except ImportError:
                from config import PhotonicLanternDesignParameters

            n_cores = getattr(geometry, 'n_cores', 3)
            r_core  = float(getattr(geometry, 'core_radii', [1.2])[0]
                            if hasattr(geometry, 'core_radii')
                            else getattr(geometry, 'r_core', 1.2))
            n_core  = getattr(geometry, 'n_core', 1.53)
            n_clad  = getattr(geometry, 'n_clad', 1.0)
            k0      = getattr(geometry, 'k0', 2 * np.pi / (wavelength_nm / 1000.0))
            V_num   = getattr(geometry, 'V_number',
                              k0 * r_core * np.sqrt(max(n_core**2 - n_clad**2, 1e-6)))
            NA      = float(np.sqrt(max(n_core**2 - n_clad**2, 1e-6)))
            MFD     = 2.0 * r_core * (0.65 + 1.619 / max(V_num, 0.5)**1.5
                                      + 2.879 / max(V_num, 0.5)**6)
            if hasattr(geometry, 'positions') and len(geometry.positions) >= 2:
                pos = np.array(geometry.positions)
                dists = [np.linalg.norm(pos[i] - pos[j])
                         for i in range(len(pos)) for j in range(i+1, len(pos))]
                pitch_val = float(np.min(dists))
            else:
                pitch_val = 8.0
            R_ring_val = float(np.max(np.linalg.norm(
                np.array(geometry.positions), axis=1))) if (
                hasattr(geometry, 'positions') and len(geometry.positions) > 0) else pitch_val
            packing_val = min(n_cores * np.pi * r_core**2 /
                              (np.pi * max(R_ring_val + r_core, 1.0)**2), 0.9)
            has_central = False
            if hasattr(geometry, 'positions'):
                norms = np.linalg.norm(np.array(geometry.positions), axis=1)
                has_central = bool(np.any(norms < 0.5 * r_core))
            config_type_val = 'hexagonal' if n_cores in (7, 19) else 'circular'
            n_eff_lp01 = float(modes[0]['n_eff']) if modes else float(n_core - 0.01)
            taper_len = getattr(geometry, 'taper_length', None)
            L_mux_val   = max(float(taper_len) * 0.5, 100.0) if taper_len else 200.0
            L_taper_val = float(taper_len) if taper_len and taper_len > 0 else 375.0
            L_MMF_val   = 100.0

            design_params_v = PhotonicLanternDesignParameters(
                N_cores=n_cores, has_central_core=has_central,
                config_type=config_type_val,
                geometry_config=f'{n_cores}-{config_type_val}',
                n_peripheral_cores=n_cores - (1 if has_central else 0),
                R_ring=R_ring_val, packing_efficiency=float(packing_val),
                pitch=pitch_val, pitch_min=pitch_val,
                pitch_ratio=float(pitch_val / (2.0 * r_core + 1e-9)),
                wavelength=float(wavelength_nm),
                r_core_SM=float(r_core), r_clad_SM=62.5,
                n_core_SM=float(n_core), n_clad_SM=float(n_clad),
                V_SM=float(V_num), NA_SM=float(NA), MFD=float(MFD),
                n_eff_LP01=n_eff_lp01,
                r_core_MM=25.0, V_MM=float(np.sqrt(n_cores) * V_num),
                NA_MM=0.22, M_max=max(int(n_cores * V_num**2 / 4), 1),
                n_polymer=float(n_core), d_polymer=2.0,
                coupling_uniformity=0.95,
                L_mux=L_mux_val, L_taper=L_taper_val, L_MMF=L_MMF_val,
                L_total=L_mux_val + L_taper_val + L_MMF_val,
                n_taper=1.0, taper_profile='exponential',
            )

            result_v = VectorialLossCalculator.calculate_vectorial_losses(
                modes, geometry, design_params_v, direction, wavelength_nm)

            if result_v.get('success', False):
                # Ajouter crosstalk vectoriel (calculé depuis P_x/P_y)
                crosstalk_dB = LossCalculator._calculate_crosstalk_vectorial(modes)
                confs = [m.get('confinement', 0) for m in modes]
                avg_conf = float(np.mean(confs)) if confs else 0.0
                radiation = EnhancedLossCalculator._calculate_radiation_loss(
                    modes, wavelength_nm)
                return {
                    'IL_dB':                  result_v['IL_total'],
                    'MDL_dB':                 result_v['MDL_total'],
                    'PDL_dB':                 result_v['PDL_total'],
                    'crosstalk_dB':           crosstalk_dB,
                    'radiation_loss_dB_per_m': radiation,
                    'avg_confinement':         avg_conf,
                    'n_modes_used':            result_v['n_modes_used'],
                    'direction':               direction,
                    'wavelength_nm':           wavelength_nm,
                    'is_vectorial':            True,
                    'success':                 True,
                }
            # Fallback vers scalaire si vectoriel échoue

        # ── FIX V18.2 : Import depuis config.py (module local existant) ────
        try:
            from config import PhotonicLanternDesignParameters
        except ImportError:
            from config import PhotonicLanternDesignParameters

        # === Extraire paramètres RÉELS depuis la géométrie ===
        n_cores = getattr(geometry, 'n_cores', 3)
        r_core  = getattr(geometry, 'core_radii', [1.2])[0] if hasattr(geometry, 'core_radii') \
                  else getattr(geometry, 'r_core', 1.2)
        n_core  = getattr(geometry, 'core_index',
                  getattr(geometry, 'n_core', 1.53))
        n_clad  = getattr(geometry, 'clad_index',
                  getattr(geometry, 'n_clad', 1.0))
        k0      = getattr(geometry, 'k0', 2 * np.pi / (wavelength_nm / 1000.0))
        V_num   = getattr(geometry, 'V_number',
                          k0 * r_core * np.sqrt(max(n_core**2 - n_clad**2, 1e-6)))
        NA      = float(np.sqrt(max(n_core**2 - n_clad**2, 1e-6)))
        MFD     = 2.0 * r_core * (0.65 + 1.619 / max(V_num, 0.5)**1.5
                                  + 2.879 / max(V_num, 0.5)**6)

        # Pitch depuis positions cœurs si disponible
        if hasattr(geometry, 'positions') and len(geometry.positions) >= 2:
            positions = np.array(geometry.positions)
            dists = [np.linalg.norm(positions[i] - positions[j])
                     for i in range(len(positions))
                     for j in range(i + 1, len(positions))]
            pitch_val = float(np.min(dists))
        elif hasattr(geometry, 'core_positions') and len(geometry.core_positions) >= 2:
            positions = np.array(geometry.core_positions)
            dists = [np.linalg.norm(positions[i] - positions[j])
                     for i in range(len(positions))
                     for j in range(i + 1, len(positions))]
            pitch_val = float(np.min(dists))
        else:
            pitch_val = 8.0

        pitch_ratio_val = pitch_val / (2.0 * r_core + 1e-9)

        # R_ring (distance max au centre)
        if hasattr(geometry, 'positions') and len(geometry.positions) > 0:
            R_ring_val = float(np.max(np.linalg.norm(
                np.array(geometry.positions), axis=1)))
        elif hasattr(geometry, 'core_positions') and len(geometry.core_positions) > 0:
            R_ring_val = float(np.max(np.linalg.norm(
                np.array(geometry.core_positions), axis=1)))
        else:
            R_ring_val = pitch_val

        # Packing efficiency = surface cœurs / surface ring
        packing_val = min(
            n_cores * np.pi * r_core**2 / (np.pi * max(R_ring_val + r_core, 1.0)**2),
            0.9
        )

        # has_central_core : vrai si un cœur est proche du centre
        has_central = False
        if hasattr(geometry, 'core_positions') and len(geometry.core_positions) > 0:
            norms = np.linalg.norm(np.array(geometry.core_positions), axis=1)
            has_central = bool(np.any(norms < 0.5 * r_core))

        config_type_val = 'hexagonal' if n_cores in (7, 19) else 'circular'

        # n_eff LP01 : premier mode si disponible
        n_eff_lp01 = float(float(modes[0]['n_eff'])) if modes else float(n_core - 0.01)

        # Longueurs (taper_length si disponible, sinon valeurs Dana typiques)
        taper_len_mm = getattr(geometry, 'taper_length', None)
        if taper_len_mm is not None and taper_len_mm > 0:
            L_taper_val = float(taper_len_mm)
            L_mux_val   = max(L_taper_val * 0.5, 100.0)
            L_MMF_val   = 100.0
        else:
            L_mux_val  = 200.0
            L_taper_val = 375.0
            L_MMF_val  = 100.0

        design_params = PhotonicLanternDesignParameters(
            N_cores=n_cores,
            has_central_core=has_central,
            config_type=config_type_val,
            geometry_config=f'{n_cores}-{config_type_val}',
            n_peripheral_cores=n_cores - (1 if has_central else 0),
            R_ring=R_ring_val,
            packing_efficiency=float(packing_val),
            pitch=pitch_val,
            pitch_min=pitch_val,
            pitch_ratio=float(pitch_ratio_val),
            wavelength=float(wavelength_nm),
            r_core_SM=float(r_core),
            r_clad_SM=62.5,
            n_core_SM=float(n_core),
            n_clad_SM=float(n_clad),
            V_SM=float(V_num),
            NA_SM=float(NA),
            MFD=float(MFD),
            n_eff_LP01=n_eff_lp01,
            r_core_MM=25.0,
            V_MM=float(np.sqrt(n_cores) * V_num),
            NA_MM=0.22,
            M_max=max(int(n_cores * V_num**2 / 4), 1),
            n_polymer=float(n_core),
            d_polymer=2.0,
            coupling_uniformity=0.95,
            L_mux=L_mux_val,
            L_taper=L_taper_val,
            L_MMF=L_MMF_val,
            L_total=L_mux_val + L_taper_val + L_MMF_val,
            n_taper=1.0,
            taper_profile='exponential',
        )

        # Calcul complet avec design_params réels
        result_full = EnhancedLossCalculator.calculate_sectional_losses(
            modes, geometry, design_params, direction, wavelength_nm
        )

        if not result_full.get('success', False):
            return {'success': False, 'error': result_full.get('error', 'unknown')}

        # Retourner format V17.x
        return {
            'IL_dB': result_full['IL_total'],
            'MDL_dB': result_full['MDL_total'],
            'PDL_dB': result_full['PDL_total'],
            'crosstalk_dB': result_full['Crosstalk'],
            'radiation_loss_dB_per_m': result_full['radiation_loss_dB_per_m'],
            'avg_confinement': result_full['avg_confinement'],
            'n_modes_used': result_full['n_modes_used'],
            'direction': direction,
            'wavelength_nm': wavelength_nm,
            'success': True,
        }


# ============================================================================
# CALCUL PERTES VECTORIEL (PDL Exact) - V18.1
# ============================================================================

class VectorialLossCalculator:
    """
    Calculateur de pertes avec PDL exact depuis modes vectoriels (Nédélec)
    
    DIFFÉRENCE vs EnhancedLossCalculator:
    -------------------------------------
    ✓ PDL calculé depuis Ex, Ey (pas approximation)
    ✓ Biréfringence captée naturellement
    ✓ Pertes par polarisation séparées
    
    UTILISATION:
    -----------
    modes_vectorial = TrueVectorialMaxwellSolver().solve_vectorial_modes(...)
    losses = VectorialLossCalculator.calculate_vectorial_losses(modes_vectorial, ...)
    """
    
    @staticmethod
    def calculate_vectorial_losses(
        modes_vectorial: List[Dict],
        geometry,
        design_params,
        direction: str = 'mux',
        wavelength_nm: float = 1550.0
    ) -> Dict:
        """
        Calcul pertes avec PDL exact depuis modes vectoriels
        
        Args:
            modes_vectorial: Modes avec Ex_dofs, Ey_dofs, PDL_dB
            geometry: PhotonicLanternGeometry
            design_params: PhotonicLanternDesignParameters
            direction: 'mux' ou 'demux'
            wavelength_nm: Longueur d'onde (nm)
        
        Returns:
            Dict avec IL, MDL, PDL exact par section
        """
        if not modes_vectorial:
            return {'success': False, 'error': 'no modes'}
        
        # Vérifier que modes sont vectoriels
        if not modes_vectorial[0].get('is_vectorial', False):
            logger.warning("Modes non-vectoriels passés à VectorialLossCalculator")
            logger.warning("Utiliser EnhancedLossCalculator pour modes scalaires")
            return {'success': False, 'error': 'modes not vectorial'}
        
        logger.info(f"Calcul pertes vectorielles: {len(modes_vectorial)} modes")
        
        try:
            # ═══════════════════════════════════════════════════════════
            # SECTION POLYMER
            # ═══════════════════════════════════════════════════════════
            polymer_losses = VectorialLossCalculator._calculate_polymer_vectorial(
                modes_vectorial, design_params, wavelength_nm
            )
            
            # ═══════════════════════════════════════════════════════════
            # SECTION TAPER (avec PDL exact)
            # ═══════════════════════════════════════════════════════════
            taper_losses = VectorialLossCalculator._calculate_taper_vectorial(
                modes_vectorial, design_params, wavelength_nm
            )
            
            # ═══════════════════════════════════════════════════════════
            # SECTION MMF
            # ═══════════════════════════════════════════════════════════
            mmf_losses = VectorialLossCalculator._calculate_mmf_vectorial(
                modes_vectorial, design_params, wavelength_nm
            )
            
            # ═══════════════════════════════════════════════════════════
            # TOTAUX
            # ═══════════════════════════════════════════════════════════
            IL_total = (
                polymer_losses['IL'] + 
                taper_losses['IL'] + 
                mmf_losses['IL']
            )
            
            MDL_total = np.sqrt(
                polymer_losses['MDL']**2 + 
                taper_losses['MDL']**2 + 
                mmf_losses['MDL']**2
            )
            
            # PDL total = somme des PDL (en dB)
            PDL_total = (
                polymer_losses['PDL'] + 
                taper_losses['PDL'] + 
                mmf_losses['PDL']
            )
            
            return {
                'success': True,
                'is_vectorial': True,
                
                # Par section
                'IL_polymer': polymer_losses['IL'],
                'MDL_polymer': polymer_losses['MDL'],
                'PDL_polymer': polymer_losses['PDL'],
                'PDL_x_polymer': polymer_losses['PDL_x'],
                'PDL_y_polymer': polymer_losses['PDL_y'],
                
                'IL_taper': taper_losses['IL'],
                'MDL_taper': taper_losses['MDL'],
                'PDL_taper': taper_losses['PDL'],
                'PDL_x_taper': taper_losses['PDL_x'],
                'PDL_y_taper': taper_losses['PDL_y'],
                
                'IL_MMF': mmf_losses['IL'],
                'MDL_MMF': mmf_losses['MDL'],
                'PDL_MMF': mmf_losses['PDL'],
                'PDL_x_MMF': mmf_losses['PDL_x'],
                'PDL_y_MMF': mmf_losses['PDL_y'],
                
                # Totaux
                'IL_total': IL_total,
                'MDL_total': MDL_total,
                'PDL_total': PDL_total,
                
                # Métriques additionnelles
                'n_modes_used': len(modes_vectorial),
                'direction': direction,
                'wavelength_nm': wavelength_nm
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul pertes vectorielles: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def _calculate_polymer_vectorial(modes_vectorial, design_params, wavelength_nm):
        """
        Pertes section polymer avec PDL exact
        
        PDL polymer vient de:
        - Absorption différentielle Ex vs Ey
        - Biréfringence matériau
        """
        n_polymer = design_params.n_polymer
        d_polymer = design_params.d_polymer  # µm
        
        # Absorption matériau (identique scalaire)
        alpha_dB_per_m = 0.2  # dB/m typique pour polymers
        IL_polymer = alpha_dB_per_m * (d_polymer * 1e-6)
        
        # MDL depuis uniformité couplage
        confinements = [m['confinement'] for m in modes_vectorial]
        if len(confinements) > 1:
            MDL_polymer = 10 * np.log10(
                max(confinements) / (min(confinements) + 1e-12)
            )
        else:
            MDL_polymer = 0.0
        
        # PDL exact depuis Ex, Ey
        P_x_list = [m['P_x'] for m in modes_vectorial]
        P_y_list = [m['P_y'] for m in modes_vectorial]
        
        P_x_total = np.sum(P_x_list)
        P_y_total = np.sum(P_y_list)
        
        if P_y_total > 1e-12 and P_x_total > 1e-12:
            PDL_polymer = 10 * np.log10(max(P_x_total, P_y_total) / min(P_x_total, P_y_total))
        else:
            PDL_polymer = 0.1  # Minimal si une polarisation domine
        
        return {
            'IL': float(np.clip(IL_polymer, 0, 1.0)),
            'MDL': float(np.clip(MDL_polymer, 0, 2.0)),
            'PDL': float(np.clip(PDL_polymer, 0.05, 1.0)),
            'PDL_x': float(P_x_total),
            'PDL_y': float(P_y_total)
        }
    
    @staticmethod
    def _calculate_taper_vectorial(modes_vectorial, design_params, wavelength_nm):
        """
        Pertes section taper avec PDL exact
        
        NOUVEAUTÉ: PDL exact depuis Ex, Ey
        - Pas d'approximation biréfringence
        - Capture couplage cross-polarisation
        """
        L_taper_um = design_params.L_taper
        n_taper = design_params.n_taper
        
        # IL taper (adiabaticity)
        # Plus long = plus adiabatique = moins de pertes
        adiabatic_quality = np.exp(-0.001 * L_taper_um * n_taper)
        IL_adiabatic = 2.0 * (1 - adiabatic_quality)
        
        # Pertes radiatives (dépend longueur)
        radiation_penalty = 5.0 * np.exp(-L_taper_um / 200.0)
        
        # Couplage modal
        n_modes = len(modes_vectorial)
        mode_coupling_loss = 0.5 * np.log10(n_modes + 1)
        
        IL_taper = IL_adiabatic + radiation_penalty + mode_coupling_loss
        
        # MDL depuis uniformité Ex vs Ey
        P_x_list = [m['P_x'] for m in modes_vectorial]
        P_y_list = [m['P_y'] for m in modes_vectorial]
        
        if len(P_x_list) > 1:
            var_x = np.var(P_x_list)
            var_y = np.var(P_y_list)
            MDL_taper = 10 * np.log10(1 + (var_x + var_y) / 2)
        else:
            MDL_taper = 0.0
        
        # PDL exact = moyenne des PDL individuels pondérée par puissance
        PDL_individual = [m['PDL_dB'] for m in modes_vectorial]
        powers = [m['P_x'] + m['P_y'] for m in modes_vectorial]
        
        if sum(powers) > 1e-12:
            PDL_taper = np.average(PDL_individual, weights=powers)
        else:
            PDL_taper = np.mean(PDL_individual)
        
        # Ajouter contribution biréfringence taper
        # Δn_biref ≈ 1e-5 pour polymers
        k0 = 2 * np.pi / (wavelength_nm * 1e-3)  # µm⁻¹
        biref_contribution = 4.343 * k0 * 1e-5 * L_taper_um
        PDL_taper += biref_contribution
        
        logger.debug(f"Taper vectoriel: L={L_taper_um:.1f}µm, PDL={PDL_taper:.2f}dB (exact)")
        
        return {
            'IL': float(np.clip(IL_taper, 0, 10.0)),
            'MDL': float(np.clip(MDL_taper, 0, 5.0)),
            'PDL': float(np.clip(PDL_taper, 0.05, 3.0)),
            'PDL_x': float(np.sum(P_x_list)),
            'PDL_y': float(np.sum(P_y_list))
        }
    
    @staticmethod
    def _calculate_mmf_vectorial(modes_vectorial, design_params, wavelength_nm):
        """
        Pertes section MMF avec PDL exact
        """
        L_MMF_um = design_params.L_MMF
        
        # Pertes MMF (faibles, propagation courte)
        IL_MMF = 0.32  # dB typique
        
        # MDL minimal dans MMF (mélange modal)
        MDL_MMF = 0.05
        
        # PDL dans MMF (biréfringence fibre)
        # MMF typique: faible biréfringence
        PDL_MMF = 0.05
        
        # Puissances par polarisation (moyenne)
        P_x_mean = np.mean([m['P_x'] for m in modes_vectorial])
        P_y_mean = np.mean([m['P_y'] for m in modes_vectorial])
        
        return {
            'IL': float(IL_MMF),
            'MDL': float(MDL_MMF),
            'PDL': float(PDL_MMF),
            'PDL_x': float(P_x_mean),
            'PDL_y': float(P_y_mean)
        }


if __name__ == '__main__':
    print("=" * 70)
    print("ENHANCED LOSS CALCULATOR V18.1 - SCALAIRE + VECTORIEL")
    print("=" * 70)
    print("\nV18.0 - Calcul de pertes par section:")
    print("  ✓ POLYMER (multiplexeur)")
    print("  ✓ TAPER (transition adiabatique)")
    print("  ✓ MMF (fibre multimode)")
    print()
    print("Métriques globales étendues:")
    print("  ✓ coupling_degradation")
    print("  ✓ geometry_penalty")
    print("  ✓ crosstalk_penalty")
    print()
    print("NOUVEAU V18.1 - Pertes Vectorielles:")
    print("  ✓ VectorialLossCalculator avec PDL exact")
    print("  ✓ PDL calculé depuis Ex, Ey (pas approximation)")
    print("  ✓ Pertes par polarisation (PDL_x, PDL_y)")
    print("  ✓ Biréfringence captée naturellement")
    print()
    print("Utilisation:")
    print("  # Scalaire (rapide)")
    print("  EnhancedLossCalculator.calculate_sectional_losses(modes, ...)")
    print()
    print("  # Vectoriel (PDL exact)")
    print("  VectorialLossCalculator.calculate_vectorial_losses(modes_vectorial, ...)")
    print("=" * 70)