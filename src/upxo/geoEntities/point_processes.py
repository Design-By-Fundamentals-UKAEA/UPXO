"""
Point Process Generation Module
================================

Generate synthetic point patterns for microstructure modeling and testing.

Includes:
  - Poisson (uniform random)
  - Poisson cluster process
  - Matérn hard-core process
  - Regular lattice
  - Gibbs processes
  - Strauss process
  - Log-Gaussian Cox process (LGCP)
  - Thomas cluster process

Example:
    from upxo.pxtalops.point_processes import PoissonPointProcess, MaternHardCore
    
    # Generate Poisson points
    ppp = PoissonPointProcess(intensity=0.01, window=(100, 100))
    points = ppp.generate()
    
    # Generate hard-core process
    mhc = MaternHardCore(intensity=0.005, hard_core_radius=5, window=(100, 100))
    points = mhc.generate()
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from ripleyk import calculate_ripley
import pointpats
from pointpats import PointPattern, window
from pointpats.distance_statistics import k

@dataclass
class Window:
    """Spatial window for point process"""
    xmin: float = 0.0
    xmax: float = 100.0
    ymin: float = 0.0
    ymax: float = 100.0
    
    @property
    def width(self) -> float:
        return self.xmax - self.xmin
    
    @property
    def height(self) -> float:
        return self.ymax - self.ymin
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @classmethod
    def from_tuple(cls, bounds: Tuple[float, float, float, float]):
        """Create from (xmin, xmax, ymin, ymax)"""
        return cls(xmin=bounds[0], xmax=bounds[1], ymin=bounds[2], ymax=bounds[3])


class PointProcess(ABC):
    """Abstract base class for point processes"""
    
    def __init__(self, 
                 window: Optional[Tuple[float, float, float, float]] = None,
                 seed: Optional[int] = None):
        """
        Args:
            window: (xmin, xmax, ymin, ymax) or None for (0,100,0,100)
            seed: random seed for reproducibility
        """
        if window is None:
            window = (0, 100, 0, 100)
        self.window = Window.from_tuple(window) if isinstance(window, tuple) else window
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    @abstractmethod
    def generate(self) -> pd.DataFrame:
        """Generate point pattern. Returns DataFrame with columns [x, y, ...]"""
        pass

    def calculate_k_function(self, points: pd.DataFrame, 
                             r_max: float = 10.0, 
                             r_count: int = 50) -> Dict[str, np.ndarray]:
        """
        Calculates Ripley's K-function K(r) using PySAL's pointpats.K.
        """
        if len(points) < 2:
            return {'r': np.array([0]), 'K_r': np.array([0])}

        coords = points[['x', 'y']].values
        
        # 1. Define the Window for PySAL
        # Define the vertices of the rectangle, ensuring the polygon is closed 
        # (Start and end points are the same).
        window_coords = np.array([
            [self.window.xmin, self.window.ymin],
            [self.window.xmax, self.window.ymin],
            [self.window.xmax, self.window.ymax],
            [self.window.xmin, self.window.ymax],
            [self.window.xmin, self.window.ymin] # Closing the loop
        ])
        
        # Construct the Window object using the 'parts' argument
        pats_window = Window([window_coords]) 
        
        # 2. Create the Point Pattern Object
        pp = PointPattern(coords, window=pats_window)
        
        # 3. Calculate K-function
        k_func = k(pp, support=r_max)
        
        return {'r': k_func.support, 'K_r': k_func.K}

    def calculate_g_r(self, points: pd.DataFrame, 
                      r_max: float = 10.0, 
                      r_count: int = 50, 
                      dimension: int = 2) -> Dict[str, np.ndarray]:
        """
        Calculates the Pair Correlation Function g(r) by deriving it 
        numerically from the K-function obtained via pointpats.K.
        
        Args:
            points (pd.DataFrame): DataFrame with 'x', 'y' (and 'z' if 3D) columns.
            r_max (float): Maximum radius (r) for the calculation.
            r_count (int): Number of radii to sample between 0 and r_max.
            dimension (int): The spatial dimension (2 or 3) of the pattern.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing 'r_g' (radii for g(r)) 
                                   and 'g_r' (the Pair Correlation function estimate).
        """
        
        # 1. Calculate K(r) using the accurate pointpats method
        k_results = self.calculate_k_function_pysal(points, r_max, r_count)
        r = k_results['r']
        K_r = k_results['K_r']
        
        # 2. Numerical Differentiation (dK(r)/dr)
        
        # Calculate the derivative of K_r with respect to r
        dK_dr = np.diff(K_r) / np.diff(r)
        
        # The resulting r vector for g(r) is the midpoint of the original r segments
        r_g = (r[:-1] + r[1:]) / 2 
        
        # 3. Calculate g(r) using the formula: g(r) = (1 / (C * r^(D-1))) * dK(r)/dr
        if dimension == 2:
            # 2D formula: g(r) = (1 / (2 * pi * r)) * dK(r)/dr
            g_r = dK_dr / (2 * np.pi * r_g)
        
        elif dimension == 3:
            # 3D formula: g(r) = (1 / (4 * pi * r^2)) * dK(r)/dr
            g_r = dK_dr / (4 * np.pi * r_g**2)
            
        else:
            raise ValueError(f"Unsupported dimension for g(r) calculation: {dimension}")
            
        return {'r_g': r_g, 'g_r': g_r}
    
    def plot(self, points: pd.DataFrame, title: str = "Point Pattern", ax=None):
        """Plot generated points"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(points['x'], points['y'], alpha=0.6, s=30, edgecolors='k')
        ax.set_xlim(self.window.xmin, self.window.xmax)
        ax.set_ylim(self.window.ymin, self.window.ymax)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        return ax


class PoissonPointProcess(PointProcess):
    """
    Homogeneous Poisson point process.
    
    Points are uniformly distributed; counts follow Poisson distribution.
    """
    
    def __init__(self, 
                 intensity: float = 0.01,
                 window: Optional[Tuple] = None,
                 seed: Optional[int] = None):
        """
        Args:
            intensity: λ = mean number of points per unit area
            window: spatial bounds
            seed: random seed
        """
        super().__init__(window, seed)
        self.intensity = intensity
    
    def generate(self) -> pd.DataFrame:
        """Generate Poisson points"""
        n_points = np.random.poisson(self.intensity * self.window.area)
        
        x = np.random.uniform(self.window.xmin, self.window.xmax, n_points)
        y = np.random.uniform(self.window.ymin, self.window.ymax, n_points)
        
        return pd.DataFrame({'x': x, 'y': y})


class InhomogeneousPoissonPointProcess(PointProcess):
    """
    Inhomogeneous Poisson point process with spatially varying intensity.
    """
    
    def __init__(self,
                 intensity_func,
                 max_intensity: float,
                 window: Optional[Tuple] = None,
                 seed: Optional[int] = None):
        """
        Args:
            intensity_func: callable(x, y) -> intensity at location
            max_intensity: maximum intensity (for thinning)
            window: spatial bounds
            seed: random seed
        """
        super().__init__(window, seed)
        self.intensity_func = intensity_func
        self.max_intensity = max_intensity
    
    def generate(self) -> pd.DataFrame:
        """Generate inhomogeneous Poisson via thinning"""
        # Generate homogeneous background
        n_background = np.random.poisson(self.max_intensity * self.window.area)
        x = np.random.uniform(self.window.xmin, self.window.xmax, n_background)
        y = np.random.uniform(self.window.ymin, self.window.ymax, n_background)
        
        # Thin by intensity ratio
        intensity = self.intensity_func(x, y)
        u = np.random.uniform(0, 1, n_background)
        keep = u < (intensity / self.max_intensity)
        
        return pd.DataFrame({'x': x[keep], 'y': y[keep]})


class PoissonClusterProcess(PointProcess):
    """
    Poisson cluster process (Neyman-Scott).
    
    Parents follow Poisson; offspring cluster around parents.
    """
    
    def __init__(self,
                 parent_intensity: float = 0.005,
                 n_offspring_per_parent: int = 10,
                 offspring_radius: float = 5.0,
                 window: Optional[Tuple] = None,
                 seed: Optional[int] = None):
        """
        Args:
            parent_intensity: intensity of parent Poisson
            n_offspring_per_parent: mean number of offspring per parent
            offspring_radius: std dev of offspring around parent
            window: spatial bounds
            seed: random seed
        """
        super().__init__(window, seed)
        self.parent_intensity = parent_intensity
        self.n_offspring = n_offspring_per_parent
        self.offspring_radius = offspring_radius
    
    def generate(self) -> pd.DataFrame:
        """Generate cluster points"""
        n_parents = np.random.poisson(self.parent_intensity * self.window.area)
        
        parent_x = np.random.uniform(self.window.xmin, self.window.xmax, n_parents)
        parent_y = np.random.uniform(self.window.ymin, self.window.ymax, n_parents)
        
        points = []
        for px, py in zip(parent_x, parent_y):
            n_off = np.random.poisson(self.n_offspring)
            ox = px + np.random.normal(0, self.offspring_radius, n_off)
            oy = py + np.random.normal(0, self.offspring_radius, n_off)
            
            # Clip to window
            in_window = (
                (ox >= self.window.xmin) & (ox <= self.window.xmax) &
                (oy >= self.window.ymin) & (oy <= self.window.ymax)
            )
            points.extend(zip(ox[in_window], oy[in_window]))
        
        if not points:
            return pd.DataFrame({'x': [], 'y': []})
        
        x, y = zip(*points)
        return pd.DataFrame({'x': x, 'y': y})


class MaternHardCore(PointProcess):
    """
    Matérn hard-core process.
    
    Points repel each other: no two points within hard_core_radius.
    Generated via thinning of Poisson.
    """
    
    def __init__(self,
                 intensity: float = 0.005,
                 hard_core_radius: float = 5.0,
                 window: Optional[Tuple] = None,
                 seed: Optional[int] = None):
        """
        Args:
            intensity: target intensity (achieved intensity may be lower)
            hard_core_radius: minimum allowed distance between points
            window: spatial bounds
            seed: random seed
        """
        super().__init__(window, seed)
        self.intensity = intensity
        self.hard_core_radius = hard_core_radius
    
def generate(self) -> pd.DataFrame:
        """Generate hard-core points via rejection sampling (optimized with KDTree)"""
        
        points_list = []
        
        # Generate background Poisson with higher intensity
        background_intensity = self.intensity * 2  # buffer
        n_candidates = np.random.poisson(background_intensity * self.window.area)
        
        x_cand = np.random.uniform(self.window.xmin, self.window.xmax, n_candidates)
        y_cand = np.random.uniform(self.window.ymin, self.window.ymax, n_candidates)
        
        for xi, yi in zip(x_cand, y_cand):
            
            if points_list:
                # OPTIMIZATION: Use KDTree for O(log N) nearest neighbor search
                points_arr = np.array(points_list)
                tree = KDTree(points_arr)
                
                # Check for nearest neighbor distance (k=1)
                distance, _ = tree.query((xi, yi), k=1)
                
                # If the distance is greater than or equal to the hard-core radius, keep the point
                if distance >= self.hard_core_radius:
                    points_list.append((xi, yi))
            else:
                # Always accept the first point
                points_list.append((xi, yi))
        
        if not points_list:
            return pd.DataFrame({'x': [], 'y': []})
        
        x, y = zip(*points_list)
        return pd.DataFrame({'x': x, 'y': y})


class ThomasClusterProcess(PointProcess):
    """
    Thomas cluster process.
    
    Offspring distributed normally around parent locations.
    Special case of Neyman-Scott.
    """
    
    def __init__(self,
                 parent_intensity: float = 0.001,
                 mean_offspring: float = 25,
                 offspring_std: float = 3.0,
                 window: Optional[Tuple] = None,
                 seed: Optional[int] = None):
        """
        Args:
            parent_intensity: intensity of parent Poisson
            mean_offspring: mean offspring per parent
            offspring_std: standard deviation of offspring displacement
            window: spatial bounds
            seed: random seed
        """
        super().__init__(window, seed)
        self.parent_intensity = parent_intensity
        self.mean_offspring = mean_offspring
        self.offspring_std = offspring_std
    
    def generate(self) -> pd.DataFrame:
        """Generate Thomas cluster points"""
        n_parents = np.random.poisson(self.parent_intensity * self.window.area)
        
        parent_x = np.random.uniform(self.window.xmin, self.window.xmax, n_parents)
        parent_y = np.random.uniform(self.window.ymin, self.window.ymax, n_parents)
        
        points = []
        for px, py in zip(parent_x, parent_y):
            n_off = np.random.poisson(self.mean_offspring)
            # 2D normal distribution
            dx = np.random.normal(0, self.offspring_std, n_off)
            dy = np.random.normal(0, self.offspring_std, n_off)
            
            ox = px + dx
            oy = py + dy
            
            # Clip to window
            in_window = (
                (ox >= self.window.xmin) & (ox <= self.window.xmax) &
                (oy >= self.window.ymin) & (oy <= self.window.ymax)
            )
            points.extend(zip(ox[in_window], oy[in_window]))
        
        if not points:
            return pd.DataFrame({'x': [], 'y': []})
        
        x, y = zip(*points)
        return pd.DataFrame({'x': x, 'y': y})


class StraussProcess(PointProcess):
    """
    Strauss process: pair-interaction point process.
    
    Inhibitory: points less likely to appear near existing points.
    Interaction range and strength parameterized.
    """
    
    def __init__(self,
                 beta: float = 0.05,
                 gamma: float = 0.5,
                 interaction_range: float = 10.0,
                 window: Optional[Tuple] = None,
                 seed: Optional[int] = None,
                 max_iterations: int = 1000):
        """
        Args:
            beta: intensity parameter
            gamma: interaction parameter (0 < gamma < 1 for inhibition)
            interaction_range: radius within which interaction occurs
            window: spatial bounds
            seed: random seed
            max_iterations: MCMC iterations
        """
        super().__init__(window, seed)
        self.beta = beta
        self.gamma = gamma
        self.interaction_range = interaction_range
        self.max_iterations = max_iterations
    
    def _count_neighbors(self, x: float, y: float, points: List) -> int:
        """Count points within interaction range"""
        count = 0
        for px, py in points:
            if np.sqrt((x - px)**2 + (y - py)**2) <= self.interaction_range:
                count += 1
        return count
    
    def generate(self) -> pd.DataFrame:
        """Generate Strauss points via MCMC"""
        points = []
        
        for _ in range(self.max_iterations):
            # Propose new point
            x_new = np.random.uniform(self.window.xmin, self.window.xmax)
            y_new = np.random.uniform(self.window.ymin, self.window.ymax)
            
            n_neighbors = self._count_neighbors(x_new, y_new, points)
            
            # Acceptance probability
            log_accept = np.log(self.beta) - n_neighbors * np.log(1.0 / self.gamma + 1e-10)
            
            if np.log(np.random.uniform()) < log_accept:
                points.append((x_new, y_new))
            
            # Occasionally remove a point
            if len(points) > 0 and np.random.uniform() < 0.1:
                points.pop(np.random.randint(len(points)))
        
        if not points:
            return pd.DataFrame({'x': [], 'y': []})
        
        x, y = zip(*points)
        return pd.DataFrame({'x': x, 'y': y})


class RegularLattice(PointProcess):
    """
    Regular lattice (grid) of points.
    """
    
    def __init__(self,
                 spacing: float = 10.0,
                 window: Optional[Tuple] = None,
                 jitter: float = 0.0):
        """
        Args:
            spacing: distance between grid points
            window: spatial bounds
            jitter: random displacement std dev (0 = no jitter)
        """
        super().__init__(window, seed=None)
        self.spacing = spacing
        self.jitter = jitter
    
    def generate(self) -> pd.DataFrame:
        """Generate regular lattice"""
        x = np.arange(self.window.xmin, self.window.xmax + self.spacing, self.spacing)
        y = np.arange(self.window.ymin, self.window.ymax + self.spacing, self.spacing)
        
        xx, yy = np.meshgrid(x, y)
        points_x = xx.flatten()
        points_y = yy.flatten()
        
        if self.jitter > 0:
            points_x += np.random.normal(0, self.jitter, len(points_x))
            points_y += np.random.normal(0, self.jitter, len(points_y))
        
        return pd.DataFrame({'x': points_x, 'y': points_y})


# Convenience functions
def compare_processes(window: Tuple = (0, 100, 0, 100), seed: int = 42):
    """Generate and plot comparison of different processes"""
    processes = [
        ('Poisson', PoissonPointProcess(intensity=0.01, window=window, seed=seed)),
        ('Cluster', PoissonClusterProcess(parent_intensity=0.005, window=window, seed=seed)),
        ('Hard-Core', MaternHardCore(intensity=0.008, hard_core_radius=5, window=window, seed=seed)),
        ('Thomas', ThomasClusterProcess(parent_intensity=0.001, window=window, seed=seed)),
        ('Regular Lattice', RegularLattice(spacing=10, window=window)),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax, (name, proc) in zip(axes, processes):
        points = proc.generate()
        proc.plot(points, title=f"{name} (n={len(points)})", ax=ax)
    
    plt.tight_layout()
    return fig

