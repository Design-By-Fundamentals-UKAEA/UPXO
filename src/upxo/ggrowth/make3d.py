import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import seaborn as sns
import upxo.viz.gsviz as viz

class voxel_from_pixel:

    __slots__ = ('gsstack', 'sstack', 's', 'q',
                 'lfi', 'pvgrid',
                 'meta_dict', 'fid', 's_fid', 'fid_s', 
                 'coords', 'rep_coords'
                 )
    """
    Slot variables
    --------------
    gsstack: dict
        A dictionary containing the UPXO grain structure stack.
    sstack: dict
        A dictionary containing the State array stack.
    s: np.ndarray
        A 3D numpy array reconstructed from the (gs/s)stack.
    lfi: np.ndarray
        A 3D numpy array of Local Feature IDs, analgous to `lgi` in MCGS.
    pvgrid: pyvista.UniformGrid
        A PyVista UniformGrid object for 3D visualization.
    """
    def __init__(self, STACK, meta_dict={'creation': 'from_sstack'}):
        self.meta_dict = meta_dict

        if meta_dict['creation'] == 'from_sstack':
            self.sstack = STACK
            self.gsstack = None
        if meta_dict['creation'] == 'from_gsstack':
            self.gsstack = STACK
            self.sstack = {k: gs.s for k, gs in STACK.items()}

        # Build the 3D State array stack
        self.stack()
        self.build_pvgrid()

    def __repr__(self):
        if self.meta_dict['creation'] == 'from_sstack':
            stack_size = len(self.sstack)
        elif self.meta_dict['creation'] == 'from_gsstack':
            stack_size = len(self.gsstack)
        return ''.join([f"<vox_frm_pix: {id(self)}, ", 
                        f"{self.meta_dict['creation']}, ", 
                        f"{stack_size} slices>",
                ])

    @classmethod
    def from_gsstack(cls, gsstack, meta_dict=dict()):
        meta_dict.setdefault('creation', 'from_gsstack')
        return cls(gsstack, meta_dict=meta_dict)

    @classmethod
    def from_sstack(cls, sstack, meta_dict=dict()):
        meta_dict.setdefault('creation', 'from_sstack')
        return cls(sstack, meta_dict=meta_dict)

    @classmethod
    def from_gsgen(cls, input_dashboard='C:\\Development\\UPXO\\upxo_library\\src\\upxo\\demos\\profiling\\input_files\\Profiling-001-Alg200-Q20-M10.xls',
                   meta_dict=dict(),
                   retain_gsdb=False):
        # ---------- Import MCGS ----------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs(input_dashboard=input_dashboard)
        pxt.simulate()
        # ---------------------------------
        meta_dict.setdefault('retain_gsdb', retain_gsdb)
        # ---------------------------------
        if retain_gsdb:
            STACK = pxt.gs
            meta_dict.setdefault('creation', 'from_gsstack')
            return cls.from_gsstack(STACK, meta_dict=meta_dict)
        else:
            STACK = {k: gs.s for k, gs in pxt.gs.items()}
            meta_dict.setdefault('creation', 'from_sstack')
            return cls.from_sstack(STACK, meta_dict=meta_dict)

    @classmethod
    def from_empty(cls):
        meta_dict = {'creation': 'from_empty'}
        STACK = dict()
        return cls(STACK, meta_dict=meta_dict)

    def stack(self):
        self.s = np.stack([self.sstack[k] 
                           for k in sorted(self.sstack.keys())],
                           axis=-1, dtype=np.int16)
        self.q = np.unique(self.s)

    def add_slice(self, location, gs, gstype='2D'):
        self.gsstack[location] = gs

    def find_cells(self, method, connectivity=3):
        if method == 1:
            from upxo.gsdataops.grid_ops import detect_grains_3d
            self.lfi = detect_grains_3d(self.s, connectivity=connectivity,
                                        return_num_grains=False)
        elif method == 2:
            from upxo.gsdataops.grid_ops import detect_grains_3d_optimized
            self.lfi = detect_grains_3d_optimized(self.s, connectivity=connectivity,
                                        return_num_grains=False)
        elif method == 3:
            from upxo.gsdataops.grid_ops import detect_grains_greedy_3d
            self.lfi = detect_grains_greedy_3d(self.s)
        elif method == 4:
            from upxo.gsdataops.grid_ops import detect_grains_using_s
            self.lfi = detect_grains_using_s(self.s, self.q)
        elif method == 5:
            from upxo.gsdataops.grid_ops import detect_grains_fast_v3
            self.lfi = detect_grains_fast_v3(self.s, self.q)
        else:
            raise ValueError("method must be 1, 2, or 3.")
        self.build_fids()
        self.build_coords()
        self.build_lfi_s()
        self.build_s_lfi()

    def build_fids(self):
        self.fid = np.asarray(np.unique(self.lfi), dtype=np.int64)

    def build_coords(self):
        coords = {}
        for fid in self.fid:
            coords[int(fid)] = np.argwhere(self.lfi == fid)
        self.coords = coords
        self.build_rep_coords()
    
    def build_rep_coords(self):
        rep_coords = []
        for coord in self.coords.values():
            rep_coords.append(coord[0])
        self.rep_coords = rep_coords


    def build_lfi_s(self):
        self.fid_s = {int(fid): int(self.s[coord[0], coord[1], coord[2]]) 
                 for fid, coord in zip(self.fid, self.rep_coords)}

    def build_s_lfi(self):
        self.s_fid = {int(s): [] for s in self.q}
        for fid, s in self.fid_s.items():
            self.s_fid[s].append(fid)

    def build_pvgrid(self, db='s',
                     origin=(0, 0, 0),
                     spacing=(1.0, 1.0, 1.0)):
        if not hasattr(self, 'pvgrid'):
            self.pvgrid = dict()

        if db == 's':
            data = self.s
        elif db == 'lfi':
            data = self.lfi
        else:
            raise ValueError("db must be 's' or 'lfi'.")

        from upxo.gsdataops.grid_ops import build_pvgrid
        self.pvgrid[db] = build_pvgrid(data=data, origin=origin,
                                        spacing=spacing)

    def see_stack(self, db='s', opacity='linear', cmap='nipy_spectral'):
        self.pvgrid[db].plot(cmap=cmap, show_scalar_bar=True,
                   opacity=opacity, jupyter_backend='pythreejs')

    def section(self, db='s', axis=0, location=0):
        if db == 's':
            DB = self.s
        elif db == 'lfi':
            DB = self.lfi
        else:
            raise ValueError("db must be 's' or 'lfi'.")
       
        from upxo.gsdataops.grid_ops import section_from_3d
        return section_from_3d(DB, axis=axis, location=location)
    
    def see_section(self, db='s', axis=0, location=0, preset='minimal',):
        section = self.section(db=db, axis=axis, location=location)
        viz.see_map(section, cmap='viridis', preset=preset,
                    title=f"Section of {db} at axis {axis}, loc {location}",
                    xlabel=['Simulation time', 'Simulation time', 'Y-axis'][axis],
                    ylabel=['Y-axis', 'X-axis', 'X-axis'][axis]
                    )