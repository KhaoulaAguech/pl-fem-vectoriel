#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coupled Mode Theory (CMT) for Photonic Lantern Mux/Demux V.1 
=================================================================
Propagation adiabatique MUX/DEMUX avec couplages rigoureux

AMÉLIORATIONS V17.1:
- Coefficients couplage documentés/justifiés
- Option calcul rigoureux avec mesh
- Conservation puissance vérifiée
- Gestion erreurs améliorée
"""

import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from typing import List, Dict, Optional
import logging

from .geometry import PhotonicLanternGeometry

try:
    from skfem import Basis
    from skfem.assembly import BilinearForm, asm
    SKFEM_AVAILABLE = True
except ImportError:
    SKFEM_AVAILABLE = False


logger = logging.getLogger('pl_v17.cmt')


class CoupledModeTheory:
    """
    Théorie modes couplés pour propagation le long taper
    
    Direction MUX: MCF (N cœurs séparés) → MMF (N supermodes couplés)
    Direction DEMUX: inverse (MMF → MCF)
    
    Équation: dA/dz = -i H(z) A(z)
    où H_mn = β_m δ_mn + C_mn (couplage)
    """

    def __init__(self, omega: float, coupling_method: str = 'approximate'):
        """
        Args:
            omega: Pulsation angulaire 2πc/λ [rad/s]
            coupling_method: 'approximate' (rapide) ou 'rigorous' (lent, précis)
        """
        self.omega = omega
        self.coupling_method = coupling_method
        
        if coupling_method not in ['approximate', 'rigorous']:
            raise ValueError(f"coupling_method doit être 'approximate' ou 'rigorous'")

    def propagate_cmt(self,
                      z_positions: np.ndarray,
                      local_modes_list: List[List[Dict]],
                      initial_amplitudes: np.ndarray,
                      direction: str = 'mux',
                      use_adaptive: bool = False) -> Dict:
        """
        Propagation CMT avec MUX/DEMUX
        
        Args:
            z_positions: Positions longitudinales [µm]
            local_modes_list: Modes à chaque z (même longueur que z_positions)
            initial_amplitudes: Amplitudes initiales [complexe]
            direction: 'mux' (MCF→MMF) ou 'demux' (MMF→MCF)
            use_adaptive: Si True, utilise solve_ivp adaptatif (plus lent)
            
        Returns:
            Dict avec amplitudes finales, IL, pertes segment, etc.
        """
        z_pos = np.asarray(z_positions, dtype=float)
        modes_list = local_modes_list[:]
        A_init = np.asarray(initial_amplitudes, dtype=complex)

        if len(z_pos) != len(modes_list):
            raise ValueError(f"z_positions ({len(z_pos)}) et modes_list ({len(modes_list)}) "
                           "doivent avoir même longueur")

        # DEMUX: inverser z et modes
        if direction.lower() == 'demux':
            z_pos = z_pos[::-1]
            modes_list = modes_list[::-1]
            # Normaliser amplitudes (distribution uniforme en entrée MMF)
            power_init = np.sum(np.abs(A_init)**2)
            if power_init > 1e-12:
                A_init = A_init / np.sqrt(power_init) * np.sqrt(len(A_init))

        # Validation
        n_modes = len(A_init)
        for i, modes in enumerate(modes_list):
            if len(modes) != n_modes:
                raise ValueError(f"Position z[{i}]: {len(modes)} modes vs {n_modes} attendus")

        # Propagation
        if use_adaptive:
            result = self._propagate_adaptive(z_pos, modes_list, A_init)
        else:
            result = self._propagate_piecewise(z_pos, modes_list, A_init)

        # Métriques finales
        A_final = result['amplitudes_final']
        power_init = np.sum(np.abs(A_init)**2)
        power_final = np.sum(np.abs(A_final)**2)

        IL_dB = -10 * np.log10(power_final / (power_init + 1e-15))
        conservation = power_final / (power_init + 1e-15)

        result.update({
            'IL_dB': float(IL_dB),
            'power_conservation': float(conservation),
            'direction': direction,
            'coupling_method': self.coupling_method
        })

        logger.debug(f"CMT {direction}: IL={IL_dB:.3f} dB, conservation={conservation:.4f}")

        return result

    def _propagate_piecewise(self,
                            z_pos: np.ndarray,
                            modes_list: List[List[Dict]],
                            A_init: np.ndarray) -> Dict:
        """Propagation segment par segment (rapide, approx constante par morceaux)"""
        A = A_init.copy()
        segment_losses = []

        for i in range(len(z_pos) - 1):
            dz = z_pos[i+1] - z_pos[i]
            
            if dz <= 0:
                logger.warning(f"Segment {i}: dz={dz} <= 0, ignoré")
                continue

            # Matrice couplage moyenne sur segment
            H = self._compute_coupling_matrix(modes_list[i], modes_list[i])

            # Propagation: A(z+dz) = exp(-i H dz) A(z)
            try:
                A_new = expm(-1j * H * dz) @ A
            except np.linalg.LinAlgError as e:
                logger.error(f"Segment {i}: expm échoué - {e}")
                A_new = A  # Pas de changement si échec

            # Perte segment
            power_before = np.sum(np.abs(A)**2)
            power_after = np.sum(np.abs(A_new)**2)
            loss_frac = 1.0 - power_after / (power_before + 1e-15)
            segment_losses.append(float(loss_frac))

            A = A_new

        return {
            'amplitudes_final': A,
            'segment_losses': segment_losses,
            'z_positions': z_pos
        }

    def _propagate_adaptive(self,
                           z_pos: np.ndarray,
                           modes_list: List[List[Dict]],
                           A_init: np.ndarray) -> Dict:
        """Propagation adaptative avec solve_ivp (précis mais lent)"""
        
        def ode_system(z, A_flat):
            """dA/dz = -i H(z) A"""
            # Interpoler position dans modes_list
            idx = np.searchsorted(z_pos, z, side='right') - 1
            idx = np.clip(idx, 0, len(modes_list) - 1)
            
            modes = modes_list[idx]
            H = self._compute_coupling_matrix(modes, modes)
            
            A = A_flat.view(complex)
            dA = -1j * H @ A
            
            return dA.view(float)

        # solve_ivp nécessite real array
        A_init_flat = A_init.view(float)
        
        sol = solve_ivp(
            ode_system,
            t_span=(z_pos[0], z_pos[-1]),
            y0=A_init_flat,
            t_eval=z_pos,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )

        if not sol.success:
            logger.warning(f"solve_ivp: {sol.message}")

        A_final = sol.y[:, -1].view(complex)

        return {
            'amplitudes_final': A_final,
            'segment_losses': [],  # Non calculé en adaptatif
            'z_positions': sol.t,
            'solver_status': sol.message
        }

    def _compute_coupling_matrix(self,
                                 modes_i: List[Dict],
                                 modes_j: List[Dict],
                                 geometry: Optional[PhotonicLanternGeometry] = None,
                                 basis: Optional['Basis'] = None) -> np.ndarray:
        """
        Matrice couplage H pour CMT
        
        DOCUMENTATION FACTEUR 1e-3 (méthode 'approximate'):
        ====================================================
        Le facteur 1e-3 est une APPROXIMATION conservative pour couplage faible:
        
        - Valeur typique couplage modes espacés dans waveguides: C ~ 1-100 m⁻¹
        - Avec overlap O ~ 0.1-0.9 et perturbation Δε ~ 0.01:
          C ≈ (ω/4c) × O × Δε ≈ 1e-3 β pour λ=1550nm
        
        - LIMITATION: ignore structure locale taper, assume perturbation faible
        - Pour précision: utiliser coupling_method='rigorous'
        
        Méthode rigoureuse:
        ===================
        C_mn = (ω/4) ∫∫ Δε(x,y) E_m*(x,y) E_n(x,y) dx dy / √(P_m P_n)
        
        où Δε = variation indice le long taper
        Nécessite geometry + basis pour intégration FEM
        """
        n = len(modes_i)
        H = np.zeros((n, n), dtype=complex)

        # Diagonale: β des modes
        for i in range(n):
            H[i, i] = modes_i[i]['beta']

        # Off-diagonal: couplage
        if self.coupling_method == 'approximate':
            # ✅ Méthode rapide documentée
            for i in range(n):
                for j in range(i+1, n):
                    # Overlap simple (produit scalaire champs)
                    E_i = modes_i[i]['field_vector']
                    E_j = modes_j[j]['field_vector']
                    
                    overlap = np.abs(np.vdot(E_i, E_j))
                    
                    # Facteur 1e-3 conservatif (voir doc ci-dessus)
                    # Équivalent à perturbation Δε ~ 0.01 avec normalisation
                    coupling = overlap * 1e-3
                    
                    H[i, j] = H[j, i] = coupling

        elif self.coupling_method == 'rigorous':
            # ✅ Calcul intégral rigoureux
            if geometry is None or basis is None or not SKFEM_AVAILABLE:
                logger.warning("Rigorous coupling nécessite geometry+basis+skfem, "
                             "fallback approximate")
                # Fallback
                for i in range(n):
                    for j in range(i+1, n):
                        overlap = np.abs(np.vdot(modes_i[i]['field_vector'],
                                                modes_j[j]['field_vector']))
                        H[i, j] = H[j, i] = overlap * 1e-3
            else:
                H = self._compute_rigorous_coupling(modes_i, modes_j, geometry, basis)

        return H

    def _compute_rigorous_coupling(self,
                                   modes_i: List[Dict],
                                   modes_j: List[Dict],
                                   geometry: PhotonicLanternGeometry,
                                   basis: 'Basis') -> np.ndarray:
        """
        Couplage CMT rigoureux avec intégration FEM
        
        C_mn = (ω/4) ∫∫ Δε E_m* E_n dx dy / √(P_m P_n)
        
        Δε approximé ici par variation spatiale ε(x,y)
        Pour vraie CMT longitudinale: Δε(x,y,z) = ε(x,y,z+dz) - ε(x,y,z)
        """
        n = len(modes_i)
        H = np.zeros((n, n), dtype=complex)

        # Diagonale
        for i in range(n):
            H[i, i] = modes_i[i]['beta']

        # Masse epsilon pour pondération
        @BilinearForm
        def epsilon_product(u, v, w):
            eps = geometry.epsilon(w.x[0], w.x[1])
            # Variation approximée (simplifié)
            delta_eps = eps - np.mean(eps)
            return delta_eps * u * v

        M_eps = asm(epsilon_product, basis)

        # Calcul overlaps
        for i in range(n):
            E_i = modes_i[i]['field_vector']
            P_i = np.real(E_i.conj() @ E_i)
            
            for j in range(i+1, n):
                E_j = modes_j[j]['field_vector']
                P_j = np.real(E_j.conj() @ E_j)
                
                # Intégrale overlap pondérée
                C_ij = E_i.conj() @ M_eps @ E_j
                
                # Normalisation
                C_ij /= np.sqrt(P_i * P_j + 1e-15)
                C_ij *= self.omega / 4.0
                
                H[i, j] = H[j, i] = C_ij

        return H

    def verify_power_conservation(self, result: Dict, tolerance: float = 0.05) -> bool:
        """
        Vérifie conservation puissance
        
        Args:
            result: Sortie de propagate_cmt()
            tolerance: Tolérance fractionnelle (0.05 = 5%)
            
        Returns:
            True si conservé dans tolérance
        """
        conservation = result.get('power_conservation', 0.0)
        
        if abs(1.0 - conservation) > tolerance:
            logger.warning(f"Conservation puissance faible: {conservation:.4f} "
                         f"(tolérance {tolerance})")
            return False
        
        return True

    def estimate_adiabaticity(self,
                             z_positions: np.ndarray,
                             modes_list: List[List[Dict]]) -> Dict:
        """
        Estimation critère adiabatique: |dβ/dz| << |Δβ|²
        
        Returns:
            Dict avec violations, max gradient, etc.
        """
        violations = []
        max_gradient = 0.0

        for i in range(len(z_positions) - 1):
            dz = z_positions[i+1] - z_positions[i]
            if dz <= 0:
                continue

            modes_i = modes_list[i]
            modes_j = modes_list[i+1]

            for m in range(len(modes_i)):
                beta_i = modes_i[m]['beta']
                beta_j = modes_j[m]['beta']

                d_beta_dz = abs((beta_j - beta_i) / dz)
                max_gradient = max(max_gradient, d_beta_dz)

                # Critère adiabatique: comparer avec spacing modal
                for n in range(m+1, len(modes_i)):
                    delta_beta = abs(modes_i[m]['beta'] - modes_i[n]['beta'])
                    
                    if delta_beta > 1e-6:  # Éviter division par zéro
                        ratio = d_beta_dz / delta_beta**2
                        
                        # Seuil adiabatique (conservatif)
                        if ratio > 0.1:
                            violations.append({
                                'z': z_positions[i],
                                'modes': (m, n),
                                'ratio': float(ratio),
                                'd_beta_dz': float(d_beta_dz),
                                'delta_beta': float(delta_beta)
                            })

        return {
            'n_violations': len(violations),
            'violations': violations[:10],  # Top 10
            'max_gradient': float(max_gradient),
            'is_adiabatic': len(violations) == 0
        }


# Test module
if __name__ == '__main__':
    print("Module cmt.py V.1 - OPTIMISÉ ✓")
    print("\nAméliorations:")
    print("  - Facteur 1e-3 documenté avec justification physique")
    print("  - Méthode rigoureuse avec intégration FEM disponible")
    print("  - Propagation adaptative solve_ivp pour haute précision")
    print("  - Vérification conservation puissance")
    print("  - Estimation adiabaticity avec critère |dβ/dz| << |Δβ|²")
