import os
import glob
import shutil
import make_xyz
from make_com import make_com


def make_dirs(name: str, start_template: str, end_template: str, basis: str, num_steps=10):
    """Make the directory, xyz, and gbs file for each step in the path between the two templates.

    Parameters
    ----------
    name : str
        Name of the calculations.
    start_template : str
        Name of the XYZ file that will be used as a template for the starting point.
    end_template : str
        Name of the XYZ file that will be used as a template for the end point.
    basis : str
        Name of the basis set.
    num_steps : int
        Number of steps in the path.

    Notes
    -----
    xyz file is stored as 'system.xyz2'.

    """
    start_template_index = int(start_template.split('_')[1])
    end_template_index = int(end_template.split('_')[1])

    dir_basename = f'{name}_{start_template_index}{end_template_index}_{{}}_{basis}'
    for i, xyz in enumerate(make_xyz.xyz_from_templates(start_template, end_template, num_steps)):
        dirname = dir_basename.format(i)

        # if directory already exists
        if os.path.isdir(dirname):
            # find all others
            other_dirnames = glob.glob(dir_basename.format('*'))
            # check if xyz file matches any of the others
            has_matched = False
            for other_dirname in other_dirnames:
                xyzfile = os.path.join(other_dirname, 'system.xyz2')
                if os.path.isfile(xyzfile):
                    with open(xyzfile, 'r') as f:
                        if f.read() == xyz:
                            has_matched = True
                            break
            # if xyz file does not match any of the others
            else:
                # change the index (move to the end)
                i += len(other_dirnames)
                # update directory name
                dirname = dir_basename.format(i)

            if has_matched:
                continue

        # create directory
        os.mkdir(dirname)

        # make xyz
        xyzfile = os.path.join(dirname, 'system.xyz2')
        with open(xyzfile, 'w') as f:
            f.write(xyz)

        # make gbs
        gbsfile = os.path.join('basis', basis + '.gbs')
        # if basis set exists in basis directory
        if os.path.isfile(gbsfile):
            shutil.copyfile(gbsfile, os.path.join(dirname, basis + '.gbs'))
        else:
            print(f'Cannot find `.gbs` file that correspond to the given basis, {basis}')


def write_coms(pattern: str):
    """Write the Gaussian com files for the directories that match the given pattern.

    Parameters
    ----------
    pattern : str
        Pattern for selecting the directories.

    Notes
    -----
    The directories are assumed to have the following format: `./system_templates_index_basis/`
    where

      - `system` is the chemical structure
      - `templates` is the indices for the templates used
      - `index` is the  index of the point in the path
      - `basis` is the basis set used

    Gaussian will only be used to run HF.

    """
    for parent in glob.glob(pattern):
        if not os.path.isdir(parent):
            continue
        system, templates, index, basis = os.path.split(parent)[1].split('_')
        basis = os.path.join('basis', basis)

        dirname = os.path.join(parent, 'mo')
        # make directory if it does not exist
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        # get xyz
        with open(os.path.join(parent, 'system.xyz2'), 'r') as f:
            xyz_content = f.read()
        # get com content
        com_content = make_com(xyz_content, basis, chkfile='hf_sp.chk', memory='2gb',
                               title=f'HF/{basis} calculation for {system}/{templates}/{index}',
                               charge=0, multiplicity=1)
        # make com file
        with open(os.path.join(dirname, 'hf_sp.com'), 'w')as f:
            f.write(com_content)
