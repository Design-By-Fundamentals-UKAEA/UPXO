from upxo.ggrowth.mcgs import mcgs
from upxo.interfaces.io.expop import ArrExp2d as arex2
import os

class mcgs2d_Surrogate():
    """
    Import
    ------
    from upxo.surrModelOps.SModelJobs import mcgs2d_Surrogate as smod2d
    """

    __slots__ = ('input_dashboard', 'dim', 'Nsim', 'basePath',
                 'baseFilename', 'fileType', 'tslices', 'currentExportDirs',
                 'fileNamePadLength'
                 )

    def __init__(self, input_dashboard: str):
        self.input_dashboard = input_dashboard
        self.dim = 2
        self.currentExportDirs = dict()

    def set_NumberOfSimulations(self, Nsim: int):
        self.Nsim = Nsim

    def set_export_parameters(self, basePath: str, baseFilename: str, 
                              fileType: str, fileNamePadLength: int = 4):
        self.basePath = basePath
        self.baseFilename = baseFilename
        self.fileType = fileType
        self.fileNamePadLength = fileNamePadLength

    def makeImages(self, TemporalOrSimFolders='temporal', 
                   compression=False, fileLevelCompression=False):
        for simCount in range(self.Nsim):
            UPXO_PXTAL = mcgs(input_dashboard=self.input_dashboard)
            UPXO_PXTAL.simulate()
            UPXO_PXTAL.detect_grains(library='connected_components_3d')
            if TemporalOrSimFolders.lower() == 'temporal':
                self.saveInTemporalFolders(simCount, UPXO_PXTAL,
                            fileLevelCompression=fileLevelCompression)
            elif TemporalOrSimFolders.lower() == 'sim':
                self.saveInSimulationFolders(simCount, UPXO_PXTAL,
                            fileLevelCompression=fileLevelCompression)
            else:
                raise ValueError("TemporalOrSimFolders must be either 'temporal' or 'sim'.")

    def saveInSimulationFolders(self, simCount, UPXO_PXTAL, fileLevelCompression=False):
            sim_dir_name = f"Path_{simCount+1:0{self.fileNamePadLength}d}"
            self.currentExportDirs[simCount] = os.path.join(self.basePath, sim_dir_name)
            os.makedirs(self.currentExportDirs[simCount], exist_ok=True)
            # -----------------------------------------------
            for ts in UPXO_PXTAL.gs.keys():
                # Apply dynamic padding to Time Slice file name
                filename = f"Ensemble_{ts:0{self.fileNamePadLength}d}.{self.fileType}"
                full_export_path = os.path.join(self.currentExportDirs[simCount], filename)
                arex2.save(UPXO_PXTAL.gs[ts].lfi, full_export_path,
                           compression=fileLevelCompression)

    def saveInTemporalFolders(self, simCount, UPXO_PXTAL, fileLevelCompression=False):
        for ts in UPXO_PXTAL.gs.keys():
            # Apply dynamic padding to Time Slice Directory name
            ts_dir_name = f"EnsembleSet_{ts:0{self.fileNamePadLength}d}"
            self.currentExportDirs[ts] = os.path.join(self.basePath, ts_dir_name)
            os.makedirs(self.currentExportDirs[ts], exist_ok=True)
            
            # Apply dynamic padding to Simulation file name
            filename = f"Realization_{simCount+1:0{self.fileNamePadLength}d}.{self.fileType}"
            full_export_path = os.path.join(self.currentExportDirs[ts], filename)
            arex2.save(UPXO_PXTAL.gs[ts].lfi, full_export_path,
                       compression=fileLevelCompression)