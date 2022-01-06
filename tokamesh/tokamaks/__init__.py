from numpy import load
import os

tokamaks_dir = os.path.abspath(__file__)[:-11]


def mastu_boundary(lower_divertor=False):
    """
    Function for accessing boundary data for the MAST-Upgrade tokamak.

    :param bool lower_divertor: \
        If set to `True`, then only the boundary for the lower super-x divertor is returned.

    :return R_boundary, z_boundary: \
         The major-radius and z-height values of the boundary are returned as two 1D
         numpy arrays `R_boundary` and `z_boundary`.
    """
    boundary_data = load(tokamaks_dir + "mastu_boundary_data.npz")
    if lower_divertor:
        return boundary_data["R_lower_divertor"], boundary_data["z_lower_divertor"]
    else:
        return boundary_data["R"], boundary_data["z"]
