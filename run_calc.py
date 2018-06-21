import os
import glob
import shutil
import subprocess
import numpy as np
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


def write_coms(pattern: str, memory='2gb', charge=0, multiplicity=1):
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
        com_content = make_com(xyz_content, basis, chkfile='hf_sp.chk', memory=memory,
                               title=f'HF/{basis} calculation for {system}/{templates}/{index}',
                               charge=charge, multiplicity=multiplicity)
        # make com file
        with open(os.path.join(dirname, 'hf_sp.com'), 'w')as f:
            f.write(com_content)


def make_wfn_dirs(pattern: str, wfn_name: str, num_runs: int):
    """Make directories for running the wavefunction calculations.

    Parameters
    ----------
    pattern : str
        Pattern for the directories on which the new directories will be created.
        These directories must contain the files `oneint.npy` and `twoint.npy`.
    wfn_name : str
        Name of the wavefunction.
    num_runs : int
        Number of calculations that will be run.

    """
    for parent in glob.glob(pattern):
        if not os.path.isdir(parent):
            continue
        if not (os.path.isfile(os.path.join(parent, 'oneint.npy')) and
                os.path.isfile(os.path.join(parent, 'twoint.npy'))):
            continue

        newdir = os.path.join(parent, wfn_name)
        if not os.path.isdir(newdir):
            os.mkdir(newdir)

        for i in range(num_runs):
            try:
                os.mkdir(os.path.join(newdir, str(i)))
            except FileExistsError:
                pass


def write_wfn_py(pattern: str, nelec: int, wfn_type: str, optimize_orbs: bool=False,
                 pspace_exc=None, objective=None, solver=None,
                 load_orbs=None, load_ham=None, load_wfn=None):
    """Make a script for running calculations.

    Parameters
    ----------
    nelec : int
        Number of electrons.
    one_int_file : str
        Path to the one electron integrals (for restricted orbitals).
        One electron integrals should be stored as a numpy array of dimension (nspin/2, nspin/2).
    two_int_file : str
        Path to the two electron integrals (for restricted orbitals).
        Two electron integrals should be stored as a numpy array of dimension
        (nspin/2, nspin/2, nspin/2, nspin/2).
    wfn_type : str
        Type of wavefunction.
        One of `fci`, `doci`, `mps`, `determinant-ratio`, `ap1rog`, `apr2g`, `apig`, `apsetg`, and
        `apg`.
    optimize_orbs : bool
        If True, orbitals are optimized.
        If False, orbitals are not optimized.
        By default, orbitals are not optimized.
        Not compatible with solvers that require a gradient (everything except cma).
    pspace_exc : list of int
        Orders of excitations that will be used to build the projection space.
        Default is first and second order excitations of the HF ground state.
    objective : str
        Form of the Schrodinger equation that will be solved.
        Use `system` to solve the Schrodinger equation as a system of equations.
        Use `least_squares` to solve the Schrodinger equation as a squared sum of the system of
        equations.
        Use `variational` to solve the Schrodinger equation variationally.
        Must be one of `system`, `least_squares`, and `variational`.
        By default, the Schrodinger equation is solved as system of equations.
    solver : str
        Solver that will be used to solve the Schrodinger equation.
        Keyword `cma` uses Covariance Matrix Adaptation - Evolution Strategy (CMA-ES).
        Keyword `diag` results in diagonalizing the CI matrix.
        Keyword `minimize` uses the BFGS algorithm.
        Keyword `least_squares` uses the Trust Region Reflective Algorithm.
        Keyword `root` uses the MINPACK hybrd routine.
        Must be one of `cma`, `diag`, `least_squares`, or `root`.
        Must be compatible with the objective.
    load_orbs : str
        Numpy file of the orbital transformation matrix that will be applied to the initial
        Hamiltonian.
        If the initial Hamiltonian parameters are provided, the orbitals will be transformed
        afterwards.
    load_ham : str
        Numpy file of the Hamiltonian parameters that will overwrite the parameters of the initial
        Hamiltonian.
    load_wfn : str
        Numpy file of the wavefunction parameters that will overwrite the parameters of the initial
        wavefunction.

    """
    cwd = os.getcwd()
    for parent in glob.glob(pattern):
        if not os.path.isdir(parent):
            continue

        os.chdir(parent)

        filename = 'calculate.py'
        oneint = os.path.abspath('../oneint.npy')
        twoint = os.path.abspath('../twoint.npy')
        hf_energies = os.path.abspath('../hf_energies.npy')

        nspin = np.load(oneint).shape[1] * 2
        nucnuc = np.load(hf_energies)[1]
        if pspace_exc is None:
            pspace_exc = [1, 2, 3, 4]
        pspace_exc = [str(i) for i in pspace_exc]
        if objective is None:
            objective = 'variational'
        if solver is None:
            solver = 'cma'

        load_files = []
        if load_orbs:
            load_files += ['--load_orbs', load_orbs]
        if load_ham:
            load_files += ['--load_ham', load_ham]
        if load_wfn:
            load_files += ['--load_wfn', load_wfn]

        subprocess.run(['python', '/project/def-ayers/kimt33/fanpy/scripts/wfns_make_script.py',
                        str(nelec), str(nspin), oneint, twoint, wfn_type,
                        '--nuc_repulsion', f'{nucnuc}', '--optimize_orbs'*optimize_orbs,
                        '--pspace', *pspace_exc, '--objective', objective, '--solver', solver,
                        *load_files,
                        '--save_ham', 'hamiltonian.npy',
                        '--save_wfn', 'wavefunction.npy',
                        '--save_chk', 'checkpoint.npy',
                        '--filename', filename])

        os.chdir(cwd)


def run_calcs(pattern: str, time='1d', memory='2GB', outfile='outfile'):
    """Run the calculations for the selected files/directories.

    Parameters
    ----------
    pattern : str
        Pattern for selecting the files.

    Notes
    -----
    Can only execute at the base directory.

    """
    cwd = os.getcwd()

    time = time.lower()
    if time[-1] == 'd':
        time = int(time[:-1]) * 24 * 60
    elif time[-1] == 'h':
        time = int(time[:-1]) * 60
    elif time[-1] == 'm':
        time = int(time[:-1])
    else:
        raise ValueError('Time must be given in minutes, hours, or days (e.g. 1440m, 24h, 1d).')

    memory = memory.upper()
    if memory[-2:] not in ['MB', 'GB']:
        raise ValueError('Memory must be given as a MB or GB (e.g. 1024MB, 1GB)')

    for filename in glob.glob(pattern):
        if os.path.commonpath([cwd, os.path.abspath(filename)]) != cwd:
            continue
        filename = os.path.abspath(filename)[len(cwd)+1:]

        _, orbital, *wfn = filename.split(os.sep)
        dirname, filename = os.path.split(filename)
        os.chdir(dirname)
        submit_job = False

        if orbital == 'mo' and os.path.splitext(filename)[1] == '.com':
            # write script (because sbatch only takes one command)
            with open('hf_sp.sh', 'w') as f:
                f.write('#!/bin/bash\n')
                f.write(f'g16 {filename}\n')
            command = ['hf_sp.sh']
            submit_job = True
        elif orbital == 'mo' and os.path.splitext(filename)[1] == '.chk':
            command = ['formchk', filename]
            submit_job = False
        elif orbital == 'mo' and os.path.splitext(filename)[1] == '.fchk':
            command = [os.environ.get('HORTONPYTHON'),
                       '/project/def-ayers/kimt33/fanpy/scripts/horton_gaussian_fchk.py',
                       'hf_energies.npy', 'oneint.npy', 'twoint.npy', 'fchk_file', filename]
            submit_job = False
        elif len(wfn) == 2:
            command = ['python', '../calculate.py']
            submit_job = True

        # print(' '.join(['sbatch', f'--time={time}', f'--output={outfile}', f'--mem={memory}',
        #                 '--account=rrg-ayers-ab', command]))
        if submit_job:
            subprocess.run(['sbatch', f'--time={time}', f'--output={outfile}', f'--mem={memory}',
                            '--account=rrg-ayers-ab', *command])
        else:
            subprocess.run(command)

        # change directory
        os.chdir(cwd)
