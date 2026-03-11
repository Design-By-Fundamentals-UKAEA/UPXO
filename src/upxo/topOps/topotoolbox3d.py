import os
import numpy as np
from scipy import ndimage
from skimage import morphology, measure
import upxo.gsdataops.gid_ops as gidOps
import upxo.gsdataops.grid_ops as gridOps
import upxo.propOps.mpropOps as mpropOps
import upxo.uiOps.outputDisplay as opDisp
import upxo.gbops.grainBoundOps3d as gbOps
import upxo.flags_and_controls.flags as FLAGS

def repair_nonManifold_voxels(lfi, articulation_mask, bboxes, library_path):
    """
    Orchestrates Tiers 0-4 to resolve non-manifold pinches in the LFI.
    
    1. Tier 0: DNA & Connectivity Setup
    2. Tier 1: Fast Pass Stencils (Global)
    3. Tier 2: Connectivity Audit (Local)
    4. Tier 3: Manifold Cell Fetcher (Signature-based)
    5. Tier 4: Atomization (Fail-safe)
    """
    # Tier 0: Metadata pass
    dna = mpropOps.analyze_grain_shapes(lfi, bboxes)
    neigh_fids = gidOps.find_neighs3d(lfi, 6)
    curr_max = np.max(lfi)
    
    # Tier 1: Global Stencil Fix (Fast Pass)
    lfi = apply_fast_pass_stencils(lfi, articulation_mask)
    
    # Process remaining stubborn articulation points
    pinch_coords = np.argwhere(articulation_mask)
    for coord in pinch_coords:
        c_tuple = tuple(coord)
        # Tier 2: Only repair verified manifold breaks
        # if not connectivity_auditor(lfi, c_tuple): continue  # Temportatyily skip audit to enfore
        
        target_fid = int(lfi[c_tuple])
        if target_fid == 0: continue
        
        # Tier 3: Manifold Cell Library Match
    
        sig = mpropOps.get_neighborhood_signature(target_fid, neigh_fids, dna, n_order=1)

        patch_path = manifold_cell_fetcher(library_path, sig)
        
        if patch_path:
            patch = np.load(patch_path)
            lfi = apply_manifold_patch(lfi, c_tuple, patch, target_fid)
        else:
            # Tier 4: Fail-safe Atomization
            lfi, curr_max = atomize_pinch(lfi, c_tuple, curr_max)
            
    return lfi

def apply_fast_pass_stencils(lfi, articulation_mask):
    """Tier 1: Global morphological welding of 26-connected pinches into 6-connected manifold interfaces."""
    struct = ndimage.generate_binary_structure(3, 1)
    p_ids = np.unique(lfi[articulation_mask])
    for fid in p_ids:
        if fid == 0: continue
        mask = (lfi == fid)
        # Weld edge-contacts into face-contacts
        ironed = ndimage.binary_closing(mask, structure=struct)
        lfi[ironed] = fid
    return lfi

def connectivity_auditor(lfi, coord, neighborhood=3):
    """Tier 2: Local 6-connectivity audit to confirm manifold breaks."""
    lim = neighborhood // 2
    z, y, x = coord
    # Extract local cube; handle boundary clipping
    z_s, y_s, x_s = lfi.shape
    sub = lfi[max(0, z-lim):min(z_s, z+lim+1), 
              max(0, y-lim):min(y_s, y+lim+1), 
              max(0, x-lim):min(x_s, x+lim+1)]
    s6 = ndimage.generate_binary_structure(3, 1)
    s26 = ndimage.generate_binary_structure(3, 3)
    _, n6 = ndimage.label(sub > 0, structure=s6)
    _, n26 = ndimage.label(sub > 0, structure=s26)
    return n6 > n26 # True if a 6-connectivity gap exists

def manifold_cell_fetcher(library_path, local_sig, tolerance=0.15):
    """Tier 3: Queries pre-verified manifold cell library using DNA signatures"""
    if not os.path.exists(library_path): return None
    best_m, min_d = None, float('inf')
    l_ar, l_vol = local_sig
    for m_file in os.listdir(library_path):
        try:
            parts = m_file.replace('.npy','').split('_')
            m_ar = float(parts[parts.index('AR')+1])
            m_vol = float(parts[parts.index('VOL')+1])
            # Distance in log-volume space
            d = np.sqrt((l_ar-m_ar)**2 + (np.log10(l_vol)-np.log10(m_vol))**2)
            if d < min_d: best_m, min_d = m_file, d
        except (ValueError, IndexError): continue
    return os.path.join(library_path, best_m) if min_d < tolerance else None

def atomize_pinch(lfi, coord, current_max_id):
    """
    Tier 4: The Fail-safe. Resolves knots by creating a unique single-voxel grain.
    
    Parameters:
    -----------
    lfi : ndarray
        The 3D grain ID array.
    coord : tuple or ndarray
        The (z, y, x) coordinate of the articulation point.
    current_max_id : int
        The highest grain ID currently in the volume.
        
    Returns:
    --------
    lfi : ndarray
        Updated volume with the new 'atom' grain.
    new_id : int
        The ID assigned to the atom (current_max_id + 1).
    """
    new_id = current_max_id + 1
    lfi[tuple(coord)] = new_id
    return lfi, new_id

def apply_manifold_patch(lfi, coord, patch, target_fid):
    """Surgically splices a Tier 3 patch into the LFI."""
    z, y, x = coord
    dz, dy, dx = patch.shape
    sz, sy, sx = dz//2, dy//2, dx//2
    # Define destination slices centered on coord
    lfi[z-sz:z+sz+1, y-sy:y+sy+1, x-sx:x+sx+1][patch] = target_fid
    return lfi

def generate_oriented_blob(coord, target_fid, dna, window_size=5):
    """Tier 5: Creates an oriented ellipsoidal mask aligned with grain DNA."""
    if target_fid not in dna or not dna[target_fid]['valid']:
        return np.ones((window_size, window_size, window_size), dtype=bool)
    rot, ar = dna[target_fid]['R'], dna[target_fid]['AR']
    lim = window_size // 2
    z, y, x = np.ogrid[-lim:lim+1, -lim:lim+1, -lim:lim+1]
    # Flatten and rotate coordinates to match grain orientation
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()])
    rot_pts = rot.T @ pts
    # Ellipsoid: (x/AR)^2 + y^2 + z^2 <= 1 (centered at 0,0,0)
    dist = (rot_pts[0]/ar)**2 + (rot_pts[1])**2 + (rot_pts[2])**2
    return (dist <= 1.0).reshape((window_size, window_size, window_size))

