#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Mesh Generation for Photonic Lantern V17.1 (OPTIMIZED)
================================================================
Génération maillage FEM adaptatif avec cache intelligent

AMÉLIORATIONS V.1:
- Limite mémoire cache (pas juste nombre entrées)
- Protection boucle infinie raffinement
- Validation robuste qualité mesh
- Stats cache détaillées
- Support mesh sauvegarde/chargement

AUTEUR: Photonic Lantern Project V.1
DATE: 2024
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.spatial import Delaunay, QhullError
from collections import OrderedDict
import logging
import hashlib
import sys
import pickle
from pathlib import Path

try:
    from skfem import Basis
    from skfem.mesh import MeshTri
    from skfem.element import ElementTriP2
    SKFEM_AVAILABLE = True
except ImportError:
    SKFEM_AVAILABLE = False
    MeshTri = None
    Basis = None

try:
    from geometry import PhotonicLanternGeometry
    from config import SimulationConfig, PhysicalConstants
except ImportError:
    from .geometry import PhotonicLanternGeometry
    from .config import SimulationConfig, PhysicalConstants


logger = logging.getLogger('pl_v17.mesh')


class MeshGenerator:
    """
    Générateur mesh adaptatif triangulaire pour FEM
    
    Stratégie raffinement:
    ---------------------
    1. Base: grille cartésienne uniforme
    2. Raffinement radial autour cœurs:
       - Intérieur cœur: très dense (modes)
       - Interface cœur-cladding: ultra-dense (gradient indice)
       - Extérieur proche: dense (couplage)
    3. Raffinement PML: annulaire extérieur
    4. Raffinement adaptatif global si besoin
    
    Cache:
    ------
    - Clé: hash(géométrie + refinement)
    - Limite: nombre entrées ET mémoire totale
    - Politique: FIFO (First-In-First-Out)
    """

    # Cache classe (partagé entre instances)
    _cache: OrderedDict = OrderedDict()
    _cache_hits: int = 0
    _cache_misses: int = 0
    _cache_max_size: int = 150
    _cache_max_memory_mb: float = 500.0  # ✅ NOUVEAU
    
    # Protection raffinement
    MAX_REFINEMENT_ITERATIONS = 5  # ✅ NOUVEAU

    @classmethod
    def generate(cls,
                 geometry: PhotonicLanternGeometry,
                 refinement: float = 1.0,
                 config: Optional[SimulationConfig] = None) -> Tuple:
        """
        Génération mesh avec cache
        
        Args:
            geometry: PhotonicLanternGeometry
            refinement: Facteur densité (1.0=normal, 2.0=2x plus dense)
            config: SimulationConfig
            
        Returns:
            (mesh, basis) tuple scikit-fem
            
        Raises:
            RuntimeError: Si scikit-fem non disponible
        """
        if not SKFEM_AVAILABLE:
            raise RuntimeError("scikit-fem requis: pip install scikit-fem")

        config = config or SimulationConfig()

        # Clé cache
        cache_key = cls._create_cache_key(geometry, refinement)

        # Lookup cache
        if config.enable_mesh_cache and cache_key in cls._cache:
            cls._cache_hits += 1
            logger.debug(f"✓ Cache hit ({cls._cache_hits} total)")
            
            # Déplacer en fin (LRU-like)
            cls._cache.move_to_end(cache_key)
            
            return cls._cache[cache_key]

        # Cache miss
        cls._cache_misses += 1
        logger.debug(f"✗ Cache miss ({cls._cache_misses} total)")

        # Génération mesh
        mesh, basis = cls._generate_mesh(geometry, refinement, config)

        # Ajout cache
        if config.enable_mesh_cache:
            cls._add_to_cache(cache_key, (mesh, basis), config)

        return mesh, basis

    @classmethod
    def _create_cache_key(cls,
                         geometry: PhotonicLanternGeometry,
                         refinement: float) -> str:
        """
        Clé cache robuste
        
        Inclut:
        - Hash géométrie (positions, rayons, indices)
        - Refinement factor
        - n_cores
        - PML params
        """
        h = hashlib.sha256()
        
        # Géométrie hash (si disponible)
        if hasattr(geometry, 'hash'):
            h.update(geometry.hash.encode())
        else:
            # Fallback: hash manuel (attributs unifiés)
            pos = getattr(geometry, 'positions',
                  getattr(geometry, 'core_positions', np.zeros((1,2))))
            h.update(np.asarray(pos).tobytes())
            h.update(np.asarray(geometry.core_radii).tobytes())
            n_core = getattr(geometry, 'n_core',
                     getattr(geometry, 'core_index', 1.5))
            h.update(f"{n_core:.6f}".encode())
        
        # Params mesh
        h.update(f"{refinement:.4f}".encode())
        h.update(str(geometry.n_cores).encode())
        h.update(f"{geometry.pml_thickness:.2f}".encode())
        h.update(str(geometry.use_complex_pml).encode())
        
        return h.hexdigest()[:24]  # 24 chars suffisant

    @classmethod
    def _add_to_cache(cls,
                     key: str,
                     value: Tuple,
                     config: SimulationConfig):
        """
        Ajout cache avec gestion mémoire
        
        ✅ AMÉLIORATION: Vérifie limite mémoire en plus du nombre entrées
        """
        mesh, basis = value
        
        # Estimer taille mémoire (approximatif)
        size_mb = (mesh.p.nbytes + mesh.t.nbytes) / (1024**2)
        
        # Calculer mémoire cache actuelle
        total_mb = cls._estimate_cache_memory_mb()
        
        # Éviction si nécessaire
        while (len(cls._cache) >= config.cache_max_size or
               total_mb + size_mb > cls._cache_max_memory_mb):
            
            if len(cls._cache) == 0:
                break
            
            # Supprimer plus ancien (FIFO)
            old_key, old_value = cls._cache.popitem(last=False)
            logger.debug(f"Cache éviction: {old_key[:8]}... "
                        f"(taille={len(cls._cache)}, mem={total_mb:.1f}MB)")
            
            # Recalculer mémoire
            total_mb = cls._estimate_cache_memory_mb()
        
        # Ajouter nouveau
        cls._cache[key] = value
        logger.debug(f"Cache ajout: {key[:8]}... (size={size_mb:.1f}MB, "
                    f"total={len(cls._cache)}/{config.cache_max_size})")

    @classmethod
    def _estimate_cache_memory_mb(cls) -> float:
        """Estime mémoire cache (MB)"""
        total_bytes = 0
        
        for mesh, basis in cls._cache.values():
            total_bytes += sys.getsizeof(mesh)
            total_bytes += sys.getsizeof(basis)
            
            # Ajouter arrays
            if hasattr(mesh, 'p'):
                total_bytes += mesh.p.nbytes
            if hasattr(mesh, 't'):
                total_bytes += mesh.t.nbytes
        
        return total_bytes / (1024**2)

    @classmethod
    def _generate_mesh(cls,
                      geometry: PhotonicLanternGeometry,
                      refinement: float,
                      config: SimulationConfig) -> Tuple:
        """
        Génération mesh adaptative
        
        ✅ PROTECTION: Limite iterations raffinement pour éviter boucle infinie
        """
        R = geometry.domain_radius
        n_base = int(25 + 20 * refinement)   # V18.11 : réduit (était 35+35) — gaine peu critique
        n_base = max(n_base, 16)  # Minimum

        logger.debug(f"Mesh generation: R={R:.1f}µm, n_base={n_base}, refinement={refinement:.2f}")

        # 1. Grille cartésienne base
        x = np.linspace(-R, R, n_base, dtype=np.float64)
        y = np.linspace(-R, R, n_base, dtype=np.float64)
        X, Y = np.meshgrid(x, y)
        points = np.vstack([X.ravel(), Y.ravel()])

        # 2. Raffinement radial cœurs — V18.11 OPTIMISÉ
        # Stratégie : densité λ/12 dans les cœurs, λ/6 à l'interface, λ/3 en gaine proche.
        # On réduit n_r_interior et n_r_interface qui étaient surdimensionnés
        # (52k pts pour 9 cœurs → objectif ~15-20k pts sans perte de précision).
        n_refine_theta = max(int(16 * refinement), 12)
        theta = np.linspace(0, 2*np.pi, n_refine_theta, endpoint=False)

        # Attributs unifiés (MCFGeometry / PhotonicLanternGeometry)
        positions  = getattr(geometry, 'positions',
                     getattr(geometry, 'core_positions', np.zeros((1,2))))
        positions  = np.atleast_2d(np.asarray(positions))
        core_radii = np.asarray(geometry.core_radii)

        # r_core : rayon représentatif pour le raffinement
        r_core_ref = float(getattr(geometry, 'r_core', np.mean(core_radii)))

        for (cx, cy), r in zip(positions, core_radii):

            # Intérieur cœur (dense — λ/12)
            n_r_interior = max(int(14 * refinement), 10)
            r_interior = np.linspace(0, r * 0.95, n_r_interior)
            Rg, Tg = np.meshgrid(r_interior, theta)
            x_int = cx + Rg.ravel() * np.cos(Tg.ravel())
            y_int = cy + Rg.ravel() * np.sin(Tg.ravel())
            points = np.hstack([points, np.vstack([x_int, y_int])])

            # Interface cœur-gaine (ultra-dense — gradient d'indice)
            n_r_interface = max(int(18 * refinement), 14)
            r_interface = np.linspace(r * 0.90, r * 1.20, n_r_interface)
            Rg, Tg = np.meshgrid(r_interface, theta)
            x_intf = cx + Rg.ravel() * np.cos(Tg.ravel())
            y_intf = cy + Rg.ravel() * np.sin(Tg.ravel())
            points = np.hstack([points, np.vstack([x_intf, y_intf])])

        # 3. Raffinement PML annulaire
        pml_start = R - geometry.pml_thickness * 1.1
        if pml_start > 0:
            n_r_pml = max(int(18 * refinement), 12)
            n_theta_pml = max(int(36 * refinement), 24)
            theta_pml = np.linspace(0, 2*np.pi, n_theta_pml, endpoint=False)
            
            r_pml = np.linspace(pml_start, R * 0.98, n_r_pml)
            Rg, Tg = np.meshgrid(r_pml, theta_pml)
            x_pml = Rg.ravel() * np.cos(Tg.ravel())
            y_pml = Rg.ravel() * np.sin(Tg.ravel())
            points = np.hstack([points, np.vstack([x_pml, y_pml])])

        # 4. Filtrage domaine
        r_dist = np.linalg.norm(points, axis=0)
        points = points[:, r_dist <= R * 1.01]

        # 5. Remove duplicates (tolérance)
        points = np.round(points.T, decimals=8).T
        points = np.unique(points, axis=1)

        logger.debug(f"Points total avant Delaunay: {points.shape[1]:,}")

        # 6. Triangulation Delaunay
        try:
            tri = Delaunay(points.T, qhull_options='QJ Pp')  # QJ: joggle, Pp: trace
        except QhullError as e:
            logger.error(f"Delaunay échoué: {e}")
            raise RuntimeError(f"Triangulation échouée: {e}") from e

        mesh = MeshTri(tri.points.T, tri.simplices.T)

        logger.debug(f"Mesh initial: {mesh.p.shape[1]} points, {mesh.t.shape[1]} triangles")

        # 7. Raffinement adaptatif (avec protection boucle)
        target_min = config.mesh_min_points
        target_ideal = config.mesh_target_points
        
        iteration = 0
        while mesh.p.shape[1] < target_min and iteration < cls.MAX_REFINEMENT_ITERATIONS:
            logger.debug(f"Refinement {iteration+1}/{cls.MAX_REFINEMENT_ITERATIONS}: "
                        f"{mesh.p.shape[1]} < {target_min} pts")
            
            mesh = mesh.refined()
            iteration += 1
            
            # Protection explosion
            if mesh.p.shape[1] > target_ideal * 2.5:
                logger.warning(f"Mesh trop dense ({mesh.p.shape[1]:,} pts), arrêt raffinement")
                break

        # Raffinement partiel si entre min et ideal
        if mesh.p.shape[1] < target_ideal and refinement > 0.8 and iteration < cls.MAX_REFINEMENT_ITERATIONS:
            logger.debug(f"Semi-refinement: {mesh.p.shape[1]} < {target_ideal}")
            mesh = mesh.refined(0.5)  # 50% éléments raffinés

        # 8. Basis FEM
        basis = Basis(mesh, ElementTriP2())

        logger.info(f"✓ Mesh final: {mesh.p.shape[1]:,} pts, {mesh.t.shape[1]:,} triangles, "
                   f"{basis.N:,} DOFs (P2)")

        return mesh, basis

    @classmethod
    def clear_cache(cls):
        """Vide cache complètement"""
        n_entries = len(cls._cache)
        mem_mb = cls._estimate_cache_memory_mb()
        
        cls._cache.clear()
        cls._cache_hits = 0
        cls._cache_misses = 0
        
        logger.info(f"Cache vidé: {n_entries} entrées, {mem_mb:.1f} MB libérés")

    @classmethod
    def get_cache_stats(cls) -> Dict:
        """Statistiques cache détaillées"""
        total = cls._cache_hits + cls._cache_misses
        hit_rate = cls._cache_hits / total if total > 0 else 0.0
        
        return {
            'size': len(cls._cache),
            'hits': cls._cache_hits,
            'misses': cls._cache_misses,
            'hit_rate': hit_rate,
            'memory_mb': cls._estimate_cache_memory_mb(),
            'max_size': cls._cache_max_size,
            'max_memory_mb': cls._cache_max_memory_mb
        }

    @classmethod
    def print_cache_stats(cls):
        """Affiche stats cache"""
        stats = cls.get_cache_stats()
        
        print("=" * 60)
        print("MESH CACHE - STATISTIQUES")
        print("=" * 60)
        print(f"Entrées         : {stats['size']} / {stats['max_size']}")
        print(f"Mémoire         : {stats['memory_mb']:.1f} / {stats['max_memory_mb']:.1f} MB")
        print(f"Hits            : {stats['hits']:,}")
        print(f"Misses          : {stats['misses']:,}")
        print(f"Hit rate        : {stats['hit_rate']*100:.1f}%")
        print("=" * 60)

    @classmethod
    def save_cache(cls, filepath: Path):
        """Sauvegarde cache sur disque"""
        filepath = Path(filepath)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'cache': cls._cache,
                'hits': cls._cache_hits,
                'misses': cls._cache_misses
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Cache sauvegardé: {filepath} ({cls._estimate_cache_memory_mb():.1f} MB)")

    @classmethod
    def load_cache(cls, filepath: Path):
        """Charge cache depuis disque"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"Fichier cache inexistant: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        cls._cache = data['cache']
        cls._cache_hits = data['hits']
        cls._cache_misses = data['misses']
        
        logger.info(f"Cache chargé: {len(cls._cache)} entrées, "
                   f"{cls._estimate_cache_memory_mb():.1f} MB")


class MeshQualityAnalyzer:
    """Analyse qualité mesh (angles, aspect, équilateralité)"""

    @staticmethod
    def analyze(mesh) -> Dict:
        """
        Analyse qualité complète
        
        Métriques:
        ---------
        - Aspect ratio: max_edge / min_edge (idéal = 1)
        - Quality: 4√3 A / (a²+b²+c²) (1 = équilatéral parfait)
        - Min angle: éviter < 20° (mal conditionné)
        
        Returns:
            Dict avec stats
        """
        if not SKFEM_AVAILABLE or mesh is None:
            return {}

        p, t = mesh.p, mesh.t

        # Aires triangles
        v1 = p[:, t[1]] - p[:, t[0]]
        v2 = p[:, t[2]] - p[:, t[0]]
        areas = 0.5 * np.abs(v1[0] * v2[1] - v1[1] * v2[0])

        # Longueurs arêtes
        edges = [
            p[:, t[(i+1)%3]] - p[:, t[i]]
            for i in range(3)
        ]
        edge_lens = np.array([np.linalg.norm(e, axis=0) for e in edges])

        # Aspect ratio
        min_len = np.min(edge_lens, axis=0)
        max_len = np.max(edge_lens, axis=0)
        aspect = max_len / (min_len + 1e-12)

        # Quality équilateral
        sum_sq = np.sum(edge_lens**2, axis=0)
        quality = 4 * np.sqrt(3) * areas / (sum_sq + 1e-12)

        # Angles minimum (loi cosinus)
        cos_angles = []
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            
            a2 = edge_lens[j]**2
            b2 = edge_lens[k]**2
            c2 = edge_lens[i]**2
            
            cos_val = (a2 + b2 - c2) / (2 * np.sqrt(a2 * b2) + 1e-12)
            cos_angles.append(cos_val)
        
        # Max cosinus → min angle
        max_cos = np.max(cos_angles, axis=0)
        min_angle = np.degrees(np.arccos(np.clip(max_cos, -1, 1)))

        return {
            'n_points': p.shape[1],
            'n_elements': t.shape[1],
            'area_min': float(np.min(areas)),
            'area_max': float(np.max(areas)),
            'area_mean': float(np.mean(areas)),
            'aspect_min': float(np.min(aspect)),
            'aspect_max': float(np.max(aspect)),
            'aspect_mean': float(np.mean(aspect)),
            'quality_min': float(np.min(quality)),
            'quality_max': float(np.max(quality)),
            'quality_mean': float(np.mean(quality)),
            'min_angle_min': float(np.min(min_angle)),
            'min_angle_mean': float(np.mean(min_angle)),
            'poor_quality_frac': float(np.sum(quality < 0.35) / len(quality)),
            'bad_aspect_frac': float(np.sum(aspect > 8.0) / len(aspect)),
            'small_angle_frac': float(np.sum(min_angle < 20.0) / len(min_angle))
        }

    @staticmethod
    def print_analysis(mesh, logger_inst=None):
        """Affiche analyse"""
        log = logger_inst or logger
        metrics = MeshQualityAnalyzer.analyze(mesh)
        
        if not metrics:
            log.warning("Mesh invalide, pas d'analyse")
            return

        log.info("=" * 70)
        log.info("MESH QUALITY ANALYSIS")
        log.info("=" * 70)
        log.info(f"Points      : {metrics['n_points']:,}")
        log.info(f"Triangles   : {metrics['n_elements']:,}")
        log.info(f"Aire        : [{metrics['area_min']:.2e}, {metrics['area_max']:.2e}] "
                f"(moy {metrics['area_mean']:.2e})")
        log.info(f"Aspect ratio: [{metrics['aspect_min']:.2f}, {metrics['aspect_max']:.2f}] "
                f"(moy {metrics['aspect_mean']:.2f})")
        log.info(f"Quality     : [{metrics['quality_min']:.3f}, {metrics['quality_max']:.3f}] "
                f"(moy {metrics['quality_mean']:.3f}) [1=parfait]")
        log.info(f"Angle min   : {metrics['min_angle_min']:.1f}° "
                f"(moy {metrics['min_angle_mean']:.1f}°)")
        log.info(f"Éléments poor quality (<0.35) : {metrics['poor_quality_frac']*100:.1f}%")
        log.info(f"Éléments high aspect (>8)     : {metrics['bad_aspect_frac']*100:.1f}%")
        log.info(f"Éléments small angle (<20°)   : {metrics['small_angle_frac']*100:.1f}%")
        log.info("=" * 70)

    @staticmethod
    def validate_mesh_quality(mesh, strict: bool = False) -> Tuple[bool, str]:
        """
        Valide qualité mesh
        
        Args:
            mesh: scikit-fem MeshTri
            strict: Critères plus stricts si True
            
        Returns:
            (valid, message)
        """
        metrics = MeshQualityAnalyzer.analyze(mesh)
        
        if not metrics:
            return False, "Mesh invalide (analyse échouée)"

        issues = []

        # Critères base
        if metrics['min_angle_min'] < 10.0:
            issues.append(f"Angle min critique: {metrics['min_angle_min']:.1f}° < 10°")
        
        if metrics['aspect_max'] > 20.0:
            issues.append(f"Aspect ratio excessif: {metrics['aspect_max']:.1f} > 20")
        
        if metrics['poor_quality_frac'] > 0.2:
            issues.append(f"Trop d'éléments poor quality: {metrics['poor_quality_frac']*100:.0f}%")

        # Critères stricts
        if strict:
            if metrics['min_angle_min'] < 20.0:
                issues.append(f"[Strict] Angle min faible: {metrics['min_angle_min']:.1f}°")
            
            if metrics['aspect_mean'] > 3.0:
                issues.append(f"[Strict] Aspect moyen élevé: {metrics['aspect_mean']:.1f}")
            
            if metrics['quality_mean'] < 0.7:
                issues.append(f"[Strict] Quality moyenne faible: {metrics['quality_mean']:.2f}")

        if issues:
            return False, "; ".join(issues)
        
        return True, "Mesh qualité acceptable"


# Test module
if __name__ == '__main__':
    print("=" * 70)
    print("MODULE mesh.py V17.1 - OPTIMISÉ ✓")
    print("=" * 70)
    print("\nAméliorations:")
    print("  ✓ Limite mémoire cache (pas juste nombre)")
    print("  ✓ Protection boucle raffinement (MAX_ITERATIONS=5)")
    print("  ✓ Validation qualité robuste")
    print("  ✓ Save/load cache sur disque")
    print("  ✓ Stats cache détaillées")
    print("\nUtilisation:")
    print("  mesh, basis = MeshGenerator.generate(geometry, refinement=1.5)")
    print("  MeshGenerator.print_cache_stats()")
    print("  valid, msg = MeshQualityAnalyzer.validate_mesh_quality(mesh)")
    print("=" * 70)
