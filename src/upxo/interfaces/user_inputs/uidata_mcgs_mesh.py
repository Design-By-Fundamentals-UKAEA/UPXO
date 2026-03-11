from termcolor import colored
class _uidata_mcgs_mesh_():
    """
    mesh_gb_conformity
    mesh_target_fe_software
    mesh_meshing_package
    mesh_reduced_integration
    mesh_element_type
    """
    __slots__ = ('mesh_gb_conformity',
                 'mesh_target_fe_software',
                 'mesh_meshing_package',
                 'mesh_reduced_integration',
                 'mesh_element_type',
                 '__uimesh_lock__'
                 )
    DEV = True
    def __init__(self, uidata):
        self.mesh_gb_conformity = uidata['mesh_gb_conformity']
        self.mesh_target_fe_software = uidata['mesh_target_fe_software']
        self.mesh_meshing_package = uidata['mesh_meshing_package']
        self.mesh_reduced_integration = bool(uidata['mesh_reduced_integration'])
        self.mesh_element_type = uidata['mesh_element_type']

    def __repr__(self):
        _ = ' '*5
        retstr = "Attriutes of meshing parameters: \n"
        retstr += _ + f"{colored('MESH_GB_CONFORMITY', 'red')}: {colored(self.mesh_gb_conformity, 'green')}\n"
        retstr += _ + f"{colored('MESH_TARGET_FE_SOFTWARE', 'red')}: {colored(self.mesh_target_fe_software, 'green')}\n"
        retstr += _ + f"{colored('MESH_MESHING_PACKAGE', 'red')}: {colored(self.mesh_meshing_package, 'green')}\n"
        retstr += _ + f"{colored('MESH_REDUCED_INTEGRATION', 'red')}: {colored(self.mesh_reduced_integration, 'green')}\n"
        retstr += _ + f"{colored('MESH_ELEMENT_TYPE', 'red')}: {colored(self.mesh_element_type, 'green')}\n"
        return retstr