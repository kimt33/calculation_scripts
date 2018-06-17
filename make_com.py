import os


def parse_gbs(gbs_file: str):
    """Parse the given basis file.

    Parameters
    ----------
    gbs_file : str
        Name of the gbs file.

    Returns
    -------
    gbs_dict : dict

    """
    gbs_dict = {}
    with open(gbs_file, 'r') as f:
        all_lines = f.read()
        sections = all_lines.split('****')
        # first section is the comments
        for section in sections[1:]:
            section = section.strip()
            # find atom
            atom = section[:2].strip()
            # assign dict
            gbs_dict[atom] = section
    return gbs_dict


def get_gen(basis: str, atoms: list):
    """Make the gen basis part of gaussian com file.

    Parameters
    ----------
    basis : str
         Name of the basis set.

    Returns
    -------
    gen_part : str
        Gen basis part of Gaussian com file.

    Notes
    -----
    The basis set should be stored as a `.gbs` file in the current directory. Only the Gaussian
    basis format is supported.

    """
    basis_filename = basis + '.gbs'
    if not os.path.isfile(basis_filename):
        raise FileNotFoundError(f'The gbs file, {basis_filename} for the basis set, {basis}, was '
                                'not found.')
    gbs_dict = parse_gbs(basis_filename)
    return '\n****\n'.join(gbs_dict[atom] for atom in set(atoms)) + '\n****\n'


def make_com(xyz, basis, chkfile='temp.chk', memory='3gb', route=None, title='', charge=0,
             multiplicity=1):
    """Make a Gaussian com file.

    Parameters
    ----------
    xyz : str
        XYZ of the molecule.
    basis : str
        Basis set used for the calculation.
    chkfile : str
        Name of the `.chk` file produced by Gaussian.
    memory : str
        Amount of memory used by Gaussian.
    route : str
        Specification for the Gaussian calculation.
    title : str
        Title of the calculation.
    charge : int
        Charge of the molecule.
    multiplicity : int
        Spin multiplicity of the molecule.
        Singlet is 1, doublet is 2, triplet is 3, etc.

    """
    if route is None:
        route = '#p rhf/gen scf=(tight,xqc,fermi) integral=grid=ultrafine nosymmetry units=AU'
    output = f'%chk={chkfile}\n'
    output += f'%mem={memory}\n'
    output += f'{route}\n'
    output += '\n'
    output += f'{title}\n'
    output += '\n'
    output += f'{charge:d} {multiplicity:d}\n'
    output += f'{xyz}\n'
    output += '\n'

    # extract out atoms
    atoms = xyz.split()[::4]
    # make gen part
    gen = get_gen(basis, atoms)
    output += f'{gen}'
    output += '\n\n\n'

    return output
