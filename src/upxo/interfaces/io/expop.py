import numpy as np
import os

class ArrExp2d:
    """
    Orchastrator class for exporting numpy arrays.
    
    Import
    ------
    from upxo.interfaces.io.expop import ArrExp2d as arex2
    """

    @staticmethod
    def save(data: np.ndarray, filename: str, compression=False, **kwargs):
        """
        Main entry point. Detects extension and dispatches to the correct handler.
        """
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        dispatch_map = {
            '.npy': ArrExp2d._save_npy,
            '.npz': ArrExp2d._save_npz,
            '.csv': ArrExp2d._save_csv,
            '.txt': ArrExp2d._save_csv,
            '.mat': ArrExp2d._save_mat,
            '.h5': ArrExp2d._save_hdf5,
            '.hdf5': ArrExp2d._save_hdf5,
            '.xlsx': ArrExp2d._save_excel,
            '.vtk': ArrExp2d._save_vtk,
            '.vti': ArrExp2d._save_vtk,
            '.png': ArrExp2d._save_image,
            '.jpg': ArrExp2d._save_image,
            '.jpeg': ArrExp2d._save_image,
            '.bmp': ArrExp2d._save_image,
            '.tiff': ArrExp2d._save_image,
            '.tif': ArrExp2d._save_image,
        }
        if ext not in dispatch_map:
            raise ValueError(f"Unsupported file format: {ext}")
        print(f"Exporting {data.shape} array to {filename}...")
        dispatch_map[ext](data, filename, compression=compression, **kwargs)

    @staticmethod
    def _save_npy(data, filename, compression=False, **kwargs):
        if compression:
            np.savez_compressed(filename, data)
        else:
            np.save(filename, data)

    @staticmethod
    def _save_npz(data, filename, compression=False, **kwargs):
        if compression:
            np.savez_compressed(filename, data)
        else:
            np.savez(filename, data)

    @staticmethod
    def _save_npz_compressed_(data, filename, compression=False, **kwargs):
        np.savez_compressed(filename, data)

    @staticmethod
    def _save_csv(data, filename, compression=False, **kwargs):
        delimiter = kwargs.get('delimiter', ',')
        fmt = kwargs.get('fmt', '%.18e')
        np.savetxt(filename, data, delimiter=delimiter, fmt=fmt)

    @staticmethod
    def _save_mat(data, filename, compression=False, **kwargs):
        try:
            from scipy.io import savemat
        except ImportError:
            raise ImportError("Saving to .mat requires scipy. Run `pip install scipy`")
        var_name = kwargs.get('var_name', 'data')
        savemat(filename, {var_name: data})

    @staticmethod
    def _save_hdf5(data, filename, compression=False, **kwargs):
        try:
            import h5py
        except ImportError:
            raise ImportError("Saving to HDF5 requires h5py. Run `pip install h5py`")
        dataset_name = kwargs.get('dataset_name', 'dataset_1')
        mode = kwargs.get('mode', 'w')
        with h5py.File(filename, mode) as f:
            f.create_dataset(dataset_name, data=data)

    @staticmethod
    def _save_excel(data, filename, **kwargs):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Saving to Excel requires pandas. Run `pip install pandas`")
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False, header=False)

    @staticmethod
    def _save_vtk(data, filename, **kwargs):
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("Saving to VTK requires pyvista. Run `pip install pyvista`")
        rows, cols = data.shape
        spacing = kwargs.get('spacing', (1, 1, 1))
        origin = kwargs.get('origin', (0, 0, 0))
        grid = pv.ImageData(dimensions=(rows, cols, 1), spacing=spacing, origin=origin)
        grid.point_data["values"] = data.flatten(order='F') 
        grid.save(filename)

    @staticmethod
    def _save_image(data, filename, **kwargs):
        """
        Saves as a common image format.
        
        kwargs:
          mode (str): 'visual' (default) or 'raw'.
          cmap (str): Colormap for visual mode (e.g., 'viridis', 'gray', 'inferno').
        """
        mode = kwargs.get('mode', 'visual')
        if mode == 'visual':
            # Option A: Presentations, quick checks, saving floats as images
            import matplotlib.pyplot as plt
            # origin='lower' puts (0,0) at bottom-left (scientific), 
            # default is often top-left (image standard)
            origin = kwargs.get('origin', 'upper')
            plt.imsave(filename, data,
                       cmap=kwargs.get('cmap', 'viridis'),
                       origin=origin,
                       vmin=kwargs.get('vmin', np.min(data)),
                       vmax=kwargs.get('vmax', np.max(data)) )
        elif mode == 'raw':
            # Option B: Raw Pixel Mapping (Uses PIL). Use for Masks, Computer Vision input,
            # Heightmaps
            try:
                from PIL import Image
            except ImportError:
                raise ImportError("Raw image export requires Pillow (pip install pillow).")
            # Warning: PIL expects uint8 (0-255) for standard formats
            # or uint16 (0-65535) for specific TIFF modes.
            if data.dtype != np.uint8 and not filename.endswith(('tif', 'tiff')):
                print("Warning: Converting non-uint8 data to image may result in clipping.")
                img_data = data.astype(np.uint8)
            else:
                img_data = data
            img = Image.fromarray(img_data)
            img.save(filename)