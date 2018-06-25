import glob
import re
import os
import fnmatch
import numpy as np


def status(pattern: str):
    """Get the statuses of the calculations that match the given pattern.

    Pattern
    -------
    pattern : str
        Unix shell style wildcard pattern to find the calculations.

    """
    success = []
    opt_failed = []
    code_failed = []
    running = []
    for filename in glob.glob(pattern):
        with open(filename, 'r') as f:
            results = f.read()

        if re.search('Optimization was successful', results):
            success.append(filename)
        elif re.search('Optimization was not successful: ', results):
            opt_failed.append(filename)
        elif re.search('Traceback (most recent call last):', results):
            code_failed.append(filename)
        else:
            running.append(filename)

    return success, opt_failed, code_failed, running


def extract_results(pattern: str):
    """Get results from the calculations that match the given pattern.

    Pattern
    -------
    pattern : str
        Unix shell style wildcard pattern to find the calculations.

    """
    filenames, *_ = status(pattern)
    cwd = os.getcwd()
    output = []
    for filename in filenames:
        filename = os.path.abspath(filename)
        if os.path.commonpath([cwd, filename]) != cwd:
            continue
        filename = filename[len(cwd)+1:]

        with open(filename, 'r') as f:
            results = f.read()

        split_filename = filename.split(os.sep)
        dict_calc = {}
        if len(split_filename) == 6:
            _, system_basis, orbital, wfn, index, filename = split_filename
            dict_calc['wfn'] = wfn
            dict_calc['index'] = index
        elif len(split_filename) == 4:
            _, system_basis, orbital, filename = split_filename
            dict_calc['wfn'] = 'hf'
        else:
            raise NotImplementedError(f'Unsupported file/directory: {filename}')
        system, basis = system_basis.rsplit('_', 1)
        dict_calc['system'] = system
        dict_calc['basis'] = basis
        dict_calc['orbital'] = orbital
        dict_calc['filename'] = filename

        re_energy = re.search(r'Final Energy: (.+)$', results)
        energy = re_energy.group(1)
        dict_calc['energy'] = energy

        output.append(dict_calc)

    return output


def select_results(results: dict, system: str, basis: str, orbital: str, wfn: str):
    """Select results that match the given options.

    Pattern
    -------
    results : dict
        Dictionary of the results from `extract_results` 
    system : str
        Unix shell style wildcard pattern to select the systems.
    basis : str
        Basis set of the calculations.
    orbital : str
        Orbitals used for the calculation.
    wfn : str
        Wavefunction of the calculation.

    """
    output_x = []
    output_y = []
    for result in results:
        if not (fnmatch.fnmatch(result['system'], system) and
                result['basis'] == basis and
                result['orbital'] == orbital and
                result['wfn'] == wfn):
            continue
        xval = result['system'].rsplit('_', 1)[1]
        output_x.append(xval)
        output_y.append(result['energy'])
    return np.array(output_x, dtype=int), np.array(output_y, dtype=float)
