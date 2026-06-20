"""
simp.py
=======
Minimal, vectorized SIMP topology optimization for 2D compliance minimization,
written so that the *compliance sensitivity field* dc/drho is a first-class,
reusable output rather than an internal throwaway.

This module serves two roles in the "Does attention attend to mechanics?" study:

  1. Data generator  -- produce (early-iteration density, converged density)
                        pairs to train / probe AG-ResU-Net.
  2. Ground-truth     -- expose the mechanical field S = dc/drho that the
                        attention maps are compared against.

The physics is the textbook minimum-compliance SIMP problem (Sigmund 2001 /
Andreassen et al. 2011), plane-stress Q4 elements:

    c        = U^T K U = sum_e  E_e(rho_e) * (u_e^T k0 u_e)
    E_e(rho) = Emin + rho^p (E0 - Emin)              (modified SIMP)
    dc/drho_e= -p rho_e^(p-1) (E0 - Emin) (u_e^T k0 u_e)

`ce_field(x)` returns the *strain-energy density per element*  ce = u_e^T k0 u_e,
which is the rho-independent core of the sensitivity. The full signed
sensitivity is then a closed-form function of (ce, rho). Keeping ce separate
lets the study evaluate sensitivity at ANY density state (input vs. converged)
from a single FE solve at that state.

No external deps beyond numpy / scipy.
"""

from __future__ import annotations

import numpy as np
import warnings
from dataclasses import dataclass, field
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


class SingularBCError(RuntimeError):
    """Raised when the FE stiffness matrix is singular (under-constrained
    boundary conditions from random sampling). The caller should resample."""
    pass


# --------------------------------------------------------------------------- #
# Element stiffness (unit Young's modulus), plane stress Q4                    #
# --------------------------------------------------------------------------- #
def _element_stiffness(nu: float = 0.3) -> np.ndarray:
    """8x8 stiffness of a unit square bilinear element, E = 1."""
    k = np.array([
        1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
        -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8
    ])
    KE = 1 / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
    ])
    return KE


@dataclass
class BoundaryConditions:
    """Loads + supports for a rectangular nelx*nely domain.

    DOF numbering follows the 88-line convention: node (i,j) with
    i in [0,nelx], j in [0,nely], node id = i*(nely+1)+j, dofs (2*id, 2*id+1).
    """
    name: str
    fixed_dofs: np.ndarray          # constrained global dof indices
    load_dofs: np.ndarray           # loaded global dof indices
    load_vals: np.ndarray           # corresponding force magnitudes

    # spatial markers (element grid, shape (nely, nelx)) used later as a
    # geometric "distance-to-BC" baseline. Filled by the factory functions.
    support_mask: np.ndarray = field(default=None)
    load_mask: np.ndarray = field(default=None)


def _node(i, j, nely):
    return i * (nely + 1) + j


def mbb_beam(nelx: int, nely: int) -> BoundaryConditions:
    """Half-MBB beam: roller on left edge, symmetric; point load top-left."""
    # Left edge: fix x-dof (symmetry). Bottom-right corner: fix y-dof (roller).
    left_nodes = np.array([_node(0, j, nely) for j in range(nely + 1)])
    fixed = np.concatenate([2 * left_nodes,                      # u_x = 0 on left
                            [2 * _node(nelx, nely, nely) + 1]])  # u_y=0 bottom-right
    load_dofs = np.array([2 * _node(0, 0, nely) + 1])            # top-left, downward
    load_vals = np.array([-1.0])
    sup = np.zeros((nely, nelx)); sup[:, 0] = 1.0
    ld = np.zeros((nely, nelx));  ld[0, 0] = 1.0
    return BoundaryConditions("mbb", fixed, load_dofs, load_vals, sup, ld)


def cantilever(nelx: int, nely: int, load_y: str = "mid") -> BoundaryConditions:
    """Cantilever: left edge fully clamped, point load on right edge."""
    left_nodes = np.array([_node(0, j, nely) for j in range(nely + 1)])
    fixed = np.concatenate([2 * left_nodes, 2 * left_nodes + 1])
    if load_y == "mid":
        jload = nely // 2
    elif load_y == "tip":
        jload = nely
    else:
        jload = 0
    load_dofs = np.array([2 * _node(nelx, jload, nely) + 1])
    load_vals = np.array([-1.0])
    sup = np.zeros((nely, nelx)); sup[:, 0] = 1.0
    ld = np.zeros((nely, nelx));  ld[min(jload, nely - 1), nelx - 1] = 1.0
    return BoundaryConditions(f"cantilever_{load_y}", fixed, load_dofs, load_vals, sup, ld)


# --------------------------------------------------------------------------- #
# The solver                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class SIMPConfig:
    nelx: int = 48
    nely: int = 48
    volfrac: float = 0.4
    penal: float = 3.0
    rmin: float = 1.5
    E0: float = 1.0
    Emin: float = 1e-9
    nu: float = 0.3
    max_iter: int = 60
    move: float = 0.2
    tol: float = 0.01
    # --- Heaviside projection (three-field SIMP). Off by default to preserve
    # the original sensitivity-filtering behavior. When on, the design is
    # density-filtered then projected toward 0/1, giving crisp, low-grey
    # structures with smooth (non-jagged) boundaries.
    use_projection: bool = False
    beta_max: float = 16.0      # final projection sharpness
    beta_iters: int = 40        # double beta every this many iterations
    eta: float = 0.5            # projection threshold


class SIMPSolver:
    """Vectorized SIMP. Caches FE topology; `optimize` returns full history."""

    def __init__(self, cfg: SIMPConfig, bc: BoundaryConditions):
        self.cfg = cfg
        self.bc = bc
        self.KE = _element_stiffness(cfg.nu)
        self._build_topology()
        self._build_filter()

    # -- FE bookkeeping ---------------------------------------------------- #
    def _build_topology(self):
        nelx, nely = self.cfg.nelx, self.cfg.nely
        self.ndof = 2 * (nelx + 1) * (nely + 1)
        # edofMat: (nel, 8) global dofs per element, element order col-major (88-line)
        edof = np.zeros((nelx * nely, 8), dtype=int)
        el = 0
        for ix in range(nelx):
            for iy in range(nely):
                n1 = _node(ix, iy, nely)
                n2 = _node(ix + 1, iy, nely)
                n3 = _node(ix + 1, iy + 1, nely)
                n4 = _node(ix, iy + 1, nely)
                edof[el] = [2*n1, 2*n1+1, 2*n2, 2*n2+1,
                            2*n3, 2*n3+1, 2*n4, 2*n4+1]
                el += 1
        self.edofMat = edof
        self.iK = np.kron(edof, np.ones((8, 1))).flatten().astype(int)
        self.jK = np.kron(edof, np.ones((1, 8))).flatten().astype(int)
        # element (ix,iy) -> row-major (iy, ix) image position
        self.el_iy = np.repeat(np.arange(nelx), nely) * 0 + \
            np.tile(np.arange(nely), nelx)
        self.el_ix = np.repeat(np.arange(nelx), nely)
        # loads / supports
        self.F = np.zeros(self.ndof)
        self.F[self.bc.load_dofs] = self.bc.load_vals
        self.free = np.setdiff1d(np.arange(self.ndof), self.bc.fixed_dofs)

    def _build_filter(self):
        """Density/sensitivity filter weights (Sigmund mesh-independence)."""
        nelx, nely, rmin = self.cfg.nelx, self.cfg.nely, self.cfg.rmin
        nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
        iH = np.zeros(nfilter, dtype=int)
        jH = np.zeros(nfilter, dtype=int)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row = i * nely + j
                kk1 = int(max(i - (np.ceil(rmin) - 1), 0))
                kk2 = int(min(i + np.ceil(rmin), nelx))
                ll1 = int(max(j - (np.ceil(rmin) - 1), 0))
                ll2 = int(min(j + np.ceil(rmin), nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * nely + l
                        fac = rmin - np.sqrt((i - k)**2 + (j - l)**2)
                        if fac > 0:
                            iH[cc] = row; jH[cc] = col; sH[cc] = fac
                            cc += 1
        H = coo_matrix((sH[:cc], (iH[:cc], jH[:cc])),
                       shape=(nelx*nely, nelx*nely)).tocsc()
        self.H = H
        self.Hs = np.asarray(H.sum(1)).flatten()

    # -- core mechanics ---------------------------------------------------- #
    def _solve_u(self, xPhys_flat: np.ndarray) -> np.ndarray:
        """Solve K(x) U = F. xPhys_flat in element order (col-major, len nel).
        Raises SingularBCError if the stiffness matrix is singular (under-
        constrained random BCs), so the caller can resample immediately."""
        c = self.cfg
        E = c.Emin + xPhys_flat**c.penal * (c.E0 - c.Emin)
        sK = (self.KE.flatten()[np.newaxis]).T * E
        sK = sK.flatten(order='F')
        K = coo_matrix((sK, (self.iK, self.jK)),
                       shape=(self.ndof, self.ndof)).tocsc()
        K = K[self.free, :][:, self.free]
        U = np.zeros(self.ndof)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # silence MatrixRankWarning
            sol = spsolve(K, self.F[self.free])
        if not np.all(np.isfinite(sol)):
            raise SingularBCError("singular stiffness matrix (under-constrained BCs)")
        U[self.free] = sol
        return U

    def ce_field(self, x_img: np.ndarray) -> np.ndarray:
        """Strain-energy density ce = u_e^T k0 u_e for every element.

        Input : x_img density as an image (nely, nelx), values in [0,1].
        Output: ce as an image (nely, nelx), non-negative. This is the
                rho-INDEPENDENT core; the signed sensitivity is obtained via
                `sensitivity_from_ce`.
        """
        x_flat = self._img_to_elem(x_img)
        U = self._solve_u(x_flat)
        ue = U[self.edofMat]                                  # (nel, 8)
        ce = np.einsum('ij,jk,ik->i', ue, self.KE, ue)        # (nel,)
        return self._elem_to_img(ce)

    def sensitivity_from_ce(self, ce_img: np.ndarray, rho_img: np.ndarray
                            ) -> np.ndarray:
        """Closed-form signed compliance sensitivity dc/drho (image).

        dc/drho = -p * rho^(p-1) * (E0 - Emin) * ce      (<= 0 everywhere)
        """
        c = self.cfg
        return -c.penal * np.clip(rho_img, 1e-6, 1.0)**(c.penal - 1) \
            * (c.E0 - c.Emin) * ce_img

    def sensitivity(self, rho_img: np.ndarray, filtered: bool = False
                    ) -> np.ndarray:
        """Full FE solve at `rho_img` then signed dc/drho (image).

        filtered=False : raw closed-form dc/drho (clean physical reference).
        filtered=True  : the SENSITIVITY-FILTERED gradient actually used to
                         update the densities in the OC step (Sigmund 88-line):
                             dc_f = H @ (rho * dc) / Hs / max(1e-3, rho)
                         This is "the gradient that performs the density update".
        """
        dc = self.sensitivity_from_ce(self.ce_field(rho_img), rho_img)
        if not filtered:
            return dc
        x = self._img_to_elem(rho_img)
        dcf = (self.H @ (x * self._img_to_elem(dc))) / self.Hs \
            / np.maximum(1e-3, x)
        return self._elem_to_img(dcf)

    # -- optimization loop ------------------------------------------------- #
    @staticmethod
    def _project(xt, beta, eta):
        """Smooth Heaviside projection of filtered density xt toward 0/1."""
        num = np.tanh(beta * eta) + np.tanh(beta * (xt - eta))
        den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        return num / den

    @staticmethod
    def _dproject(xt, beta, eta):
        """d(projected)/d(filtered): the projection chain-rule factor."""
        den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        return beta * (1.0 - np.tanh(beta * (xt - eta)) ** 2) / den

    def optimize(self, x0_img: np.ndarray | None = None, seed: int | None = None,
                 record_every: int = 1) -> dict:
        """Run SIMP. Returns dict with density history + final sensitivity.

        If cfg.use_projection, uses density filtering + Heaviside projection
        (three-field) with beta continuation, yielding crisp 0/1 designs."""
        c = self.cfg
        nel = c.nelx * c.nely
        rng = np.random.default_rng(seed)
        if x0_img is None:
            x = np.full(nel, c.volfrac)
            x = np.clip(x + 0.03 * rng.standard_normal(nel), 0.05, 0.95)
        else:
            x = self._img_to_elem(x0_img).copy()

        proj = c.use_projection
        beta = 1.0 if proj else None
        eta = c.eta

        def phys_of(xv):
            xt = (self.H @ xv) / self.Hs
            return self._project(xt, beta, eta) if proj else xt

        xPhys = phys_of(x)
        history, compliance_hist = [], []
        change, it = 1.0, 0
        while change > c.tol and it < c.max_iter:
            U = self._solve_u(xPhys)
            ue = U[self.edofMat]
            ce = np.einsum('ij,jk,ik->i', ue, self.KE, ue)
            E = c.Emin + xPhys**c.penal * (c.E0 - c.Emin)
            comp = float((E * ce).sum())
            dc_phys = -c.penal * xPhys**(c.penal - 1) * (c.E0 - c.Emin) * ce
            dv_phys = np.ones(nel)

            if proj:
                xt = (self.H @ x) / self.Hs
                dpr = self._dproject(xt, beta, eta)
                # chain rule: d/dx = H( dPhys * dproj ) / Hs   (density filter)
                dc = (self.H @ (dc_phys * dpr)) / self.Hs
                dv = (self.H @ (dv_phys * dpr)) / self.Hs
                x_new = self._oc_update(x, dc, dv, beta=beta, eta=eta)
            else:
                # original sensitivity filtering
                dc = (self.H @ (x * dc_phys)) / self.Hs / np.maximum(1e-3, x)
                dv = dv_phys
                x_new = self._oc_update(x, dc, dv)

            xPhys = phys_of(x_new)
            change = float(np.max(np.abs(x_new - x)))
            x = x_new
            if it % record_every == 0:
                history.append(self._elem_to_img(xPhys.copy()))
                compliance_hist.append(comp)
            it += 1
            # beta continuation: sharpen projection gradually
            if proj and beta < c.beta_max and (it % c.beta_iters == 0):
                beta = min(beta * 2.0, c.beta_max)
                change = 1.0  # force a few more iterations after sharpening

        history.append(self._elem_to_img(xPhys.copy()))
        compliance_hist.append(comp)

        return {
            "x_final": self._elem_to_img(xPhys),
            "history": np.stack(history),
            "compliance": np.array(compliance_hist),
            "sensitivity_final": self.sensitivity(self._elem_to_img(xPhys)),
            "iters": it,
            "bc": self.bc,
        }

    def _oc_update(self, x, dc, dv, beta=None, eta=0.5):
        c = self.cfg
        l1, l2 = 0.0, 1e9
        nel = x.size
        target_v = c.volfrac * nel
        while (l2 - l1) / max(1e-12, (l1 + l2)) > 1e-3:
            lmid = 0.5 * (l1 + l2)
            xnew = np.clip(
                x * np.sqrt(np.maximum(0.0, -dc / dv / lmid)),
                np.maximum(0.0, x - c.move),
                np.minimum(1.0, x + c.move),
            )
            xt = (self.H @ xnew) / self.Hs
            xPhys_try = self._project(xt, beta, c.eta) if beta is not None else xt
            if xPhys_try.sum() > target_v:
                l1 = lmid
            else:
                l2 = lmid
        return xnew

    # -- element<->image conversions (col-major element order) ------------- #
    def _elem_to_img(self, v: np.ndarray) -> np.ndarray:
        return v.reshape(self.cfg.nelx, self.cfg.nely).T  # (nely, nelx)

    def _img_to_elem(self, img: np.ndarray) -> np.ndarray:
        return img.T.reshape(-1)
