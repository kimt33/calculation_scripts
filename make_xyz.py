import re
import numpy as np


def divide_path(start_point, end_point, num_divisions: int=10):
    """Divide the path set by two points.

    Parameters
    ----------
    start_point : np.ndarray(K, )
    end_point : np.ndarray(K, )
    num_divisions : int

    """
    # check numpy array
    if not (isinstance(start_point, np.ndarray) and isinstance(end_point, np.ndarray)):
        raise TypeError('Each point must be given as a numpy array.')
    # check same shape
    if start_point.shape != end_point.shape:
        raise ValueError('The two pints must have the same shape.')
    # check num_divisions
    if not isinstance(num_divisions, int):
        raise TypeError('Number of divisions must be provided as an integer.')
    if num_divisions <= 0:
        raise ValueError('Number of divisions must be greater than zero.')

    step = (end_point - start_point) / (num_divisions - 1)
    for i in range(num_divisions):
        yield start_point + i * step


def parse_xyz(xyz_file: str):
    """Parse xyz file into numpy file.

    Parameters
    ----------
    xyz_file : str
        XYZ file.

    Returns
    -------
    titles : list of str
        Titles of each titles.
    atoms : K-list of str
        Names of the atoms that corresponds to the coordinates.
    coords : np.ndarray(K, 3)
        Coordinates of each atom.

    """
    all_titles = []
    all_atoms = []
    all_coords = []
    with open(xyz_file, 'r') as f:
        for line in f:
            re_num = re.search(r'^\s*(\d+)\n', line)
            if re_num is None:
                continue

            num_atoms = int(re_num.group(1))
            title = next(f).strip()
            atoms = []
            coords = []

            for i in range(num_atoms):
                atom, *coord = next(f).strip().split()
                if len(coord) != 3:
                    raise ValueError('XYZ files must have coordinates in 3D.')
                atoms.append(atom)
                coords.append(coord)

            all_titles.append(title)
            all_atoms.append(atoms)
            all_coords.append(np.array(coords, dtype=float))
    return all_titles, all_atoms, all_coords


def xyz_from_templates(start_template, end_template, num_divisions: int=10):
    """Yield xyz from one template to another.

    Parameters
    ----------
    start_template : str
        Name of the XYZ file that will be used as a template for the starting point.
    end_template : str
        Name of the XYZ file that will be used as a template for the end point.

    Yields
    ------
        Coordinates at each step.

    """
    _, start_atoms, start_coords = parse_xyz(start_template)
    _, end_atoms, end_coords = parse_xyz(end_template)
    if not (len(start_atoms) == 1 and len(start_coords) == 1):
        raise ValueError('Template for the starting point contains more than one set of '
                         'coordiantes.')
    if not (len(end_atoms) == 1 and len(end_coords) == 1):
        raise ValueError('Template for the end point contains more than one set of coordiantes.')

    start_atoms, start_coords = start_atoms[0], start_coords[0]
    end_atoms, end_coords = end_atoms[0], end_coords[0]
    if start_atoms != end_atoms:
        raise ValueError('Given templates are not compatible with one another.')

    for coords in divide_path(start_coords, end_coords, num_divisions=num_divisions):
        yield '\n'.join(f'{atom} {coordx:.6f} {coordy:.6f} {coordz:.6f}'
                        for atom, (coordx, coordy, coordz) in zip(start_atoms, coords))
