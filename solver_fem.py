#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vectorial H-field FEM Solver — V18.10 Optimized
==============================================

Formulation H-field transverse avec pénalité div (Rahman & Davies 1984 + Hayata & Koshiba 1986)
  [1] Rahman & Davies, IEEE Trans. MTT-32 (1984)
  [2] Hayata et al., IEEE Trans. MTT-34 (1986)
  [3] Koshiba et al., J. Lightwave Technol. 12 (1994)

Problème aux valeurs propres : [A]{Ht} = β² [B]{Ht}
  - H_t = (Hx, Hy) discrétisé en éléments P2 nodaux
  - Pénalité div αp = 1 pour éliminer les modes spurieux
  - Matrice B : masse pondérée 1/ε
"""

import sys, os, logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.sparse import bmat
from scipy.sparse.linalg import eigsh

# --- skfem ---
try:
    from skfem import Basis
    from skfem.assembly import BilinearForm, asm
    from skfem.helpers import grad, dot
    from skfem.element import ElementTriP2
    SKFEM_AVAILABLE = True
except ImportError:
    SKFEM_AVAILABLE = False
    print("skfem non disponible")

# --- Imports projet ---
from geometry import PhotonicLanternGeometry
from config import SimulationConfig, PhysicalConstants

# --- Logger ---
logger = logging.getLogger('pl_v18.solver_fem')
logging.basicConfig(level=logging.INFO)

# =============================================================================
# UTILITAIRES
# =============================================================================

def _confinement_from_dofs(evec: np.ndarray,
                            x_dof: np.ndarray,
                            y_dof: np.ndarray,
                            geometry,
                            eps_at_dofs: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Confinement depuis les DOFs scalaires P2.
    """
    energy = np.abs(evec)**2
    if eps_at_dofs is not None and len(eps_at_dofs) == len(evec):
        energy /= (np.abs(eps_at_dofs) + 1e-30)

    in_core = np.zeros(len(x_dof), dtype=bool)
    for (cx, cy), r in zip(geometry.positions, geometry.core_radii):
        in_core |= (x_dof - cx)**2 + (y_dof - cy)**2 <= r**2

    total = np.sum(energy) + 1e-30
    conf  = float(np.clip(np.sum(energy[in_core]) / total, 0.0, 1.0))
    return conf, conf


def _polarization_from_interp(evec_x: np.ndarray,
                              evec_y: np.ndarray,
                              x_dof: np.ndarray,
                              y_dof: np.ndarray,
                              geometry) -> Tuple[str, float, float, float]:
    """
    Polarisation depuis Hx et Hy — V18.11 CORRIGÉ.

    Calcul sur les DOFs situés dans les cœurs uniquement (physiquement correct).
    Classification affinée :
      ratio = P_x / P_y  (énergie H dans les cœurs)
      > 10  → TE-like    (dominante x, quasi-linéaire)
      > 2.5 → HE-like    (x dominant mais couplé)
      > 0.4 → Hybrid     (vrai hybride LP-like)
      > 0.1 → EH-like    (y dominant mais couplé)
      ≤ 0.1 → TM-like    (dominante y)

    La PDL est calculée en dB à partir du même ratio.
    P_x et P_y sont retournés pour le calcul XT vectoriel.
    """
    in_core = np.zeros(len(x_dof), dtype=bool)
    for (cx, cy), r in zip(geometry.positions, geometry.core_radii):
        in_core |= (x_dof - cx)**2 + (y_dof - cy)**2 <= r**2

    # Fallback sur tout le domaine si aucun DOF dans les cœurs
    mask = in_core if np.any(in_core) else np.ones(len(x_dof), dtype=bool)

    P_x = float(np.sum(evec_x[mask]**2)) + 1e-30
    P_y = float(np.sum(evec_y[mask]**2)) + 1e-30
    ratio = P_x / P_y
    PDL = float(np.clip(10. * np.log10(max(P_x, P_y) / min(P_x, P_y)), 0., 50.))

    # Seuils affinés V18.11 — classification physique LP/HE/EH
    if   ratio > 10.0:  pol = 'TE-like'
    elif ratio >  2.5:  pol = 'HE-like'
    elif ratio >  0.4:  pol = 'Hybrid'
    elif ratio >  0.1:  pol = 'EH-like'
    else:               pol = 'TM-like'

    return pol, PDL, P_x, P_y

# =============================================================================
# SOLVEUR VECTORIEL H-FIELD
# =============================================================================

class TrueVectorialMaxwellSolver:
    def __init__(self, geometry, use_pml: bool = False):
        if not SKFEM_AVAILABLE:
            raise RuntimeError("scikit-fem requis")
        self.geometry = geometry
        self.k0 = geometry.k0
        self.use_pml = use_pml
        logger.info(f"Solveur H-field initialisé - k₀={self.k0:.4f} µm⁻¹")

    def assemble_hfield_system(self, mesh):
        """
        Assemblage des matrices H-field 2N×2N (Rahman & Davies 1984)
        """
        basis = Basis(mesh, ElementTriP2())
        N = basis.N
        eps_fn = self.geometry.epsilon

        # --- Formes bilinéaires ---
        @BilinearForm
        def kxx(u,v,w): return (1.0/np.real(eps_fn(*w.x))) * grad(u)[1]*grad(v)[1]
        @BilinearForm
        def kyy(u,v,w): return (1.0/np.real(eps_fn(*w.x))) * grad(u)[0]*grad(v)[0]
        @BilinearForm
        def kxy(u,v,w): return -(1.0/np.real(eps_fn(*w.x))) * grad(u)[1]*grad(v)[0]
        @BilinearForm
        def kyx(u,v,w): return -(1.0/np.real(eps_fn(*w.x))) * grad(u)[0]*grad(v)[1]

        @BilinearForm
        def div_xx(u,v,_): return grad(u)[0]*grad(v)[0]
        @BilinearForm
        def div_yy(u,v,_): return grad(u)[1]*grad(v)[1]
        @BilinearForm
        def div_xy(u,v,_): return grad(u)[0]*grad(v)[1]

        @BilinearForm
        def mass(u,v,_): return u*v
        @BilinearForm
        def mass_eps_inv(u,v,w): return (1.0/np.real(eps_fn(*w.x))) * u*v

        # --- Assemblage ---
        Kxx = asm(kxx, basis); Kyy = asm(kyy, basis)
        Kxy = asm(kxy, basis); Kyx = asm(kyx, basis)
        Dxx = asm(div_xx, basis); Dyy = asm(div_yy, basis); Dxy = asm(div_xy, basis)
        M   = asm(mass, basis); M_inv = asm(mass_eps_inv, basis)

        alpha_p = 1.0
        k0sq = self.k0**2
        A_xx = Kxx + alpha_p*Dxx - k0sq*M
        A_yy = Kyy + alpha_p*Dyy - k0sq*M
        A_xy = Kxy + alpha_p*Dxy
        A_yx = Kyx + alpha_p*Dxy.T

        B_xx = M_inv; B_yy = M_inv
        A = bmat([[A_xx,A_xy],[A_yx,A_yy]], format='csr')
        B = bmat([[B_xx,None],[None,B_yy]], format='csr')
        logger.info(f"Assemblage terminé - {N} DOFs P2, matrice {2*N}×{2*N}")
        return A,B,basis,Dxx,Dyy,Dxy,M_inv

    def solve_vectorial_modes(self, mesh, n_modes_target:int=20) -> List[Dict]:
        """
        Résout [A]{Ht} = β² [B]{Ht} avec filtrage DOF-mask pour radiation/spurieux
        """
        A,B,basis,Dxx,Dyy,Dxy,M_inv = self.assemble_hfield_system(mesh)
        N = basis.N

        # --- DOFs intérieurs (Dirichlet bord H=0) ---
        boundary_dofs = basis.get_dofs().all()
        interior_dofs = np.setdiff1d(np.arange(N), boundary_dofs)
        idx_x = interior_dofs; idx_y = interior_dofs + N; idx = np.concatenate([idx_x,idx_y])
        A_int = A[idx,:][:,idx]; B_int = B[idx,:][:,idx]
        x_dof_int = basis.doflocs[0][interior_dofs]; y_dof_int = basis.doflocs[1][interior_dofs]
        N_solve = len(interior_dofs)

        # --- Estimation n_eff LP01 ---
        n_core, n_clad = self.geometry.n_core, self.geometry.n_clad
        NA = np.sqrt(max(n_core**2 - n_clad**2,1e-6))
        r_mean = np.mean(self.geometry.core_radii)
        V_geom = self.k0*r_mean*NA
        b_approx = max((1.0 - 2.405/max(V_geom,2.41))**2, 0.05)
        n_eff_est = np.sqrt(n_clad**2 + b_approx*(n_core**2-n_clad**2))
        sigma = (self.k0*float(np.clip(n_eff_est,n_clad+0.05,n_core-0.005)))**2

        # --- Résolution eigsh ---
        n_req = min(n_modes_target+12, 2*N_solve-4)
        beta_sq,evecs = eigsh(A_int,k=n_req,M=B_int,sigma=sigma,which='LM',tol=1e-7,maxiter=12000)

        # --- Filtrage DOF-mask (confinement) et divergence ---
        in_core_int = np.zeros(N_solve,dtype=bool)
        for (cx,cy), r in zip(self.geometry.positions,self.geometry.core_radii):
            in_core_int |= (x_dof_int - cx)**2 + (y_dof_int - cy)**2 <= r**2
        frac_core = np.sum(in_core_int)/N_solve

        modes_raw = []
        for i in range(len(beta_sq)):
            b2 = beta_sq[i]
            if b2<=0: continue
            beta = np.sqrt(b2); ne = beta/self.k0
            if ne<=n_clad or ne>=n_core*1.01: continue

            vx = evecs[:N_solve,i].copy(); vy = evecs[N_solve:,i].copy()
            nrm = np.sqrt(np.sum(vx**2)+np.sum(vy**2))+1e-30; vx/=nrm; vy/=nrm
            div_energy = float(vx@(Dxx[interior_dofs,:][:,interior_dofs]@vx) + 2*vx@(Dxy[interior_dofs,:][:,interior_dofs]@vy) + vy@(Dyy[interior_dofs,:][:,interior_dofs]@vy))
            div_ratio = div_energy / max(b2,1e-12)

            energy_sq = vx**2 + vy**2
            conf_raw = float(np.sum(energy_sq[in_core_int]) / np.sum(energy_sq))
            ovlp = conf_raw
            pol, PDL_dB, P_x, P_y = _polarization_from_interp(vx,vy,x_dof_int,y_dof_int,self.geometry)

            modes_raw.append({'n_eff':float(ne),'beta':float(beta),'Ex_dofs':vx,'Ey_dofs':vy,
                              'P_x':P_x,'P_y':P_y,'PDL_dB':PDL_dB,'polarization':pol,
                              'confinement':conf_raw,'core_overlap':ovlp,'div_ratio':div_ratio,
                              'is_vectorial':True,'method':'H-field_V18.10'})

        # --- Filtrage divergence ---
        dr_arr = np.array([m['div_ratio'] for m in modes_raw])
        dr_med = np.median(dr_arr)
        dr_thresh = max(dr_med*10, dr_arr.min()*50, 1e-6)
        modes_phys = [m for m in modes_raw if m['div_ratio']<=dr_thresh]

        # --- Filtrage radiation ---
        conf_threshold_rad = max(5.0*frac_core, 0.05)
        modes_guided = [m for m in modes_phys if m['confinement']>=conf_threshold_rad]
        if not modes_guided: modes_guided = modes_phys

        modes_guided.sort(key=lambda x:x['n_eff'],reverse=True)
        return modes_guided

# =============================================================================
# SOLVEUR SCALAIRE P2
# =============================================================================

class ScalarHelmholtzSolver:
    def __init__(self, geometry):
        self.geometry = geometry
        self.k0 = geometry.k0

    def solve(self, mesh, n_modes_target:int=20) -> List[Dict]:
        basis = Basis(mesh, ElementTriP2())
        @BilinearForm
        def stiff(u,v,_): return dot(grad(u),grad(v))
        @BilinearForm
        def mass_s(u,v,_): return u*v
        @BilinearForm
        def eps_m(u,v,w): return np.real(self.geometry.epsilon(*w.x))*u*v

        K = asm(stiff,basis); M = asm(mass_s,basis); Me = asm(eps_m,basis)
        sigma = -(self.k0*(self.geometry.n_core-0.008))**2
        evals,evecs = eigsh(K-self.k0**2*Me,k=min(n_modes_target+8,basis.N-4),M=M,sigma=sigma,which='LM',tol=1e-6,maxiter=6000)

        x_dof,y_dof = basis.doflocs
        modes=[]
        for i in range(len(evals)):
            if evals[i]>=0: continue
            ne = np.sqrt(-evals[i])/self.k0
            if ne<=self.geometry.n_clad or ne>=self.geometry.n_core*1.005: continue
            v = evecs[:,i].copy(); nrm = np.sqrt(float(v@(M@v)))+1e-30; v/=nrm
            in_core = np.zeros(len(x_dof),dtype=bool)
            for (cx,cy),r in zip(self.geometry.positions,self.geometry.core_radii):
                in_core |= (x_dof-cx)**2 + (y_dof-cy)**2 <= r**2
            conf = float(np.sum(v[in_core]**2)/np.sum(v**2))
            modes.append({'n_eff':float(ne),'beta':float(self.k0*ne),'field_vector':v,'confinement':conf,'core_overlap':conf,'PDL_dB':0.0,'polarization':'scalar','is_vectorial':False})
        modes.sort(key=lambda x:x['n_eff'],reverse=True)
        return modes

# =============================================================================
# MAIN
# =============================================================================

if __name__=='__main__':
    print("solver_fem.py V18.10 — H-field vectoriel P2 + αp div")
