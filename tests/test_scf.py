"""
test for scf
"""

import projects
from projects import our_scf
from projects import diis
import numpy as np
import psi4
import pytest


def test_our_scf():
    nel = 5
    geom_h2o = """
    O
    H 1 1.1
    H 1 1.1 2 104
    """

    nel_h2 = 1
    geom_h2 = """
    H
    H 1 0.9
    """

    basis1 = 'sto-3g'
    basis2 = 'aug-cc-pVDZ'

    # Case 1 test w/ H2O and basis1
    H, A, g, E_nuc, S = our_scf.setup(geom_h2o, basis1)
    eps, C = our_scf.diag(H, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T
    E_total = our_scf.scf(A, H, g, D, E_nuc, S)
    mol_h2o_b1 = psi4.geometry(geom_h2o)
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/" + basis1, molecule=mol_h2o_b1)
    assert np.allclose(psi4_energy, E_total)

    # Case 2 test w/ H2O and basis2
    H, A, g, E_nuc, S = our_scf.setup(geom_h2o, basis2)
    eps, C = our_scf.diag(H, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T
    E_total2 = our_scf.scf(A, H, g, D, E_nuc, S)
    mol_h2o_b2 = psi4.geometry(geom_h2o)
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/" + basis2, molecule=mol_h2o_b2)
    assert np.allclose(psi4_energy, E_total2)

    # Case 3 test w/ H2 and basis1
    H, A, g, E_nuc, S = our_scf.setup(geom_h2, basis1)
    eps, C = our_scf.diag(H, A)
    Cocc = C[:, :nel_h2]
    D = Cocc @ Cocc.T
    E_total3 = our_scf.scf(A, H, g, D, E_nuc, S)
    mol_h2_b1 = psi4.geometry(geom_h2)
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/" + basis1, molecule=mol_h2_b1)
    assert np.allclose(psi4_energy, E_total3)

    our_scf_nbf = 110
    our_scf.setup.nbf = our_scf_nbf
    assert Exception

def test_diis():
    nel = 5
    geom_h2o = """
    O
    H 1 1.1
    H 1 1.1 2 104
    """

    nel_h2 = 1
    geom_h2 = """
    H
    H 1 0.9
    """

    basis1 = 'sto-3g'
    basis2 = 'aug-cc-pVDZ'

    # Case 1 w/ DIIS
    H, A, g, E_nuc, S = our_scf.setup(geom_h2o, basis1)
    eps, C = our_scf.diag(H, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T
    E_total = diis.diis(A, H, g, D, E_nuc, S, nel)
    mol_h2o_b1 = psi4.geometry(geom_h2o)
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/" + basis1, molecule=mol_h2o_b1)
    assert np.allclose(psi4_energy, E_total)

    # Case 2 test w/ H2O and basis2
    H, A, g, E_nuc, S = our_scf.setup(geom_h2o, basis2)
    eps, C = our_scf.diag(H, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T
    E_total2 = diis.diis(A, H, g, D, E_nuc, S, nel)
    mol_h2o_b2 = psi4.geometry(geom_h2o)
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/" + basis2, molecule=mol_h2o_b2)
    assert np.allclose(psi4_energy, E_total2)

    # Case 3 test w/ H2 and basis1
    H, A, g, E_nuc, S = our_scf.setup(geom_h2, basis1)
    eps, C = our_scf.diag(H, A)
    Cocc = C[:, :nel_h2]
    D = Cocc @ Cocc.T
    E_total3 = diis.diis(A, H, g, D, E_nuc, S, nel_h2)
    mol_h2_b1 = psi4.geometry(geom_h2)
    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/" + basis1, molecule=mol_h2_b1)
    assert np.allclose(psi4_energy, E_total3)
