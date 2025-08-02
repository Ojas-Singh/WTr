Below is a **detailed implementation plan** for a Python package named **`WTr`** that (1) builds an *N*-water amorphous cluster/surface, (2) generates *physics‑guided* surface microstates that lower a reaction’s transition‑state (TS) barrier, and (3) evaluates/returns the best water configurations for any reaction you supply. It uses **ASE** calculators and optimizers throughout (no CLI wrappers), and it includes the **math you’ll need** to implement scoring, constraints, rates, tunneling, and optimization.

You can hand this plan directly to an AI agent (or engineer) to implement.

---

## 0) Scope & design targets

* **Compute stack:** Python 3.11+, **ASE** (for structures, calculators, NEB, Dimer, Vibrations, constraints), NumPy, SciPy, scikit‑learn (optional for GP), NetworkX (for proton‑wire graph), pydantic (for typed I/O), typer (CLI).
* **Calculator:** Use `ase.calculators.xtb.XTB` (GFN2‑xTB) by default; package *must* accept any ASE calculator (e.g., DFTB+, ORCA via ASE, etc.).
* **System:** clusters/surfaces of \~12–60 H₂O. “Core” waters fixed or softly restrained; “surface” waters mobile.
* **Reactions:** Generic—user provides **Reactant complex (RC)**, **Product (P)**, optional **TS seed**; mapping of **reactive atom indices** defines “reaction axis”.
* **Temperatures:** focus on **10–20 K** (low‑T thermochemistry + tunneling).

---

## 1) Package layout

```
WTr/
  __init__.py
  cli.py
  config.py
  utils/units.py
  io/
    geometry.py         # XYZ/Traj I/O, fragment ops, alignment
    ase_helpers.py      # Atoms<->Geometry adapters, calculators factory
  geom/
    build.py            # N-water cluster builder
    constraints.py      # core selection + ASE constraints (FixAtoms, Hookean)
    transforms.py       # rigid rotations, proton flips, H-bond rewiring
  sampling/
    ts_template.py      # TS-templated surface design
    mc_field.py         # field-targeted Monte Carlo (surrogate scoring)
    docking.py          # RC placement via heuristic + local opt
  calc/
    descriptors.py      # E-field, donor/acceptor, proton-wire, strain
    path.py             # NEB (IDPP) + Climbing, Dimer TS refine
    vibthermo.py        # Vibrations, quasi-RRHO, ΔG(T)
    kinetics.py         # Eyring + tunneling (Wigner, Eckart option)
  orchestrators/
    pipeline.py         # end-to-end search & evaluate
    runners.py          # local parallel execution (futures)
  models/
    datatypes.py        # Pydantic models for configs & results
  optimize/
    gp_surrogate.py     # optional Bayesian optimization on descriptors
  reports/
    summarize.py        # CSV/JSON exports; small plots
tests/
  ...
```

---

## 2) Data models (pydantic)

```python
# models/datatypes.py
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict

class Geometry(BaseModel):
    symbols: List[str]
    coords: List[Tuple[float,float,float]]  # Å
    comment: str = ""

class ReactionSpec(BaseModel):
    name: str
    reactant: Geometry              # RC complex
    product: Geometry               # P
    ts_seed: Optional[Geometry] = None
    reactive_indices: Dict[str, List[int]]  # {"rc":[iA,iB,...], "p":[...], "ts":[...]}
    reaction_axis_atoms: Tuple[int,int]     # (donor_idx, acceptor_idx) or (A,B) that define axis

class SurfaceSpec(BaseModel):
    waters_n: int
    radius: float                   # spherical envelope radius (Å)
    core_fraction: float = 0.5      # fraction of waters in “core”
    random_seed: int = 0
    harmonic_k: Optional[float] = None  # eV/Å^2 for soft restraint (None => hard FixAtoms)

class CalcSpec(BaseModel):
    ase_calculator: str = "xtb"     # "xtb", "dftb", "orca", etc.
    calc_kwargs: Dict = Field(default_factory=lambda: {"method": "GFN2-xTB", "accuracy": 1.0})
    charge: int = 0
    spin_multiplicity: int = 1

class DescriptorSet(BaseModel):
    efield: Tuple[float,float,float]
    efield_proj: float
    donor_count: int
    acceptor_count: int
    wire_length: float
    wire_bend_max: float
    strain_penalty: float

class EvalResult(BaseModel):
    surface_id: str
    deltaE_dagger: float            # eV
    deltaG_dagger_10K: float        # eV
    deltaG_dagger_20K: float        # eV
    rate_10K: float                 # s^-1
    rate_20K: float                 # s^-1
    descriptors: DescriptorSet
    paths: Dict[str,str]            # filepaths for RC/TS/P, vib, etc.
```

---

## 3) Geometry & ASE adapters

```python
# io/ase_helpers.py
from ase import Atoms
from ase.calculators.xtb import XTB

def geometry_to_atoms(g: Geometry) -> Atoms: ...
def atoms_to_geometry(a: Atoms, comment: str="") -> Geometry: ...

def make_calculator(spec: CalcSpec):
    """
    Return an ASE calculator instance.
    Default: XTB(method="GFN2-xTB", accuracy=..., charge=..., uhf=spin_multiplicity-1)
    """

# io/geometry.py
def load_xyz(path: str) -> Geometry: ...
def save_xyz(g: Geometry, path: str) -> None: ...
def merge(ga: Geometry, gb: Geometry) -> Geometry: ...
def rigid_place(fragment: Geometry, origin: Tuple[float,float,float],
                axis: Tuple[float,float,float], angle: float) -> Geometry: ...
```

---

## 4) Building the water cluster (geom/build.py)

### Algorithm

1. **O placement:** Blue‑noise (Poisson‑disc) sampling inside sphere radius $R$ with minimum O···O distance $d_\text{min}\approx 2.6$ Å.
2. **H placement:** For each O, create ideal H₂O geometry (r$_{\text{OH}}$=0.9572 Å, $\angle$HOH=104.5°) with random rotation.
3. **Quick relaxation:** One or two steps of steepest descent using a **toy H‑bond potential** to remove clashes:

   $$
   E_\text{toy} = \sum_{\text{O–O}} k_\text{rep} \,\sigma(r) + \sum_{\text{O–H}} -\epsilon \, f(r,\theta)
   $$

   with a repulsive switch $\sigma(r)$ (e.g., $(r_0/r)^{12}$) and H‑bond shape
   $f(r,\theta)=s(r)\,(\cos\theta)^{2}_+$, $s(r)$ a smooth cutoff, $\theta$ angle O–H···O; $(\cdot)_+$ zeroes negatives.
4. **Return** Geometry.

### Functions

```python
def build_water_cluster(spec: SurfaceSpec) -> Geometry: ...
```

---

## 5) Constraints (geom/constraints.py)

* **Core selection:** Let $r_i$ be O positions. Compute distances from cluster centroid; sort waters by radius; pick innermost fraction $f=$ `core_fraction`.
* **Hard freeze:** ASE `FixAtoms(indices=core_indices)`.
* **Soft restraint (Hookean):** If `harmonic_k` provided, store equilibrium positions $\mathbf{r}_i^0$ (core atoms) and add **Hookean** constraints:

  $$
  E_\text{rest} = \frac{1}{2}k \sum_{i\in \text{core}} \lVert \mathbf{r}_i - \mathbf{r}_i^0 \rVert^2 \quad (\text{eV})
  $$

```python
def select_core_atoms(cluster: Geometry, spec: SurfaceSpec) -> list[int]: ...
def make_constraints(atoms, core_indices: list[int], k: float|None): ...
```

---

## 6) Transforms (geom/transforms.py)

* **Rigid rotation (quaternion):** Given unit axis $\hat{\mathbf{u}}$ and angle $\theta$, rotation matrix:

  $$
  R = I\cos\theta + (1-\cos\theta)\,\hat{\mathbf{u}}\hat{\mathbf{u}}^\top + [\hat{\mathbf{u}}]_\times \sin\theta
  $$

  where $[\hat{\mathbf{u}}]_\times$ is the cross‑product matrix.
* **Water proton flip:** For a water molecule with oxygen O and hydrogens H1, H2, define the **lone‑pair bisector plane**; reflect $\vec{\text{OH}}$ directions across that plane to change donor orientation while keeping O fixed; renormalize OH distances.
* **H‑bond rewiring helper:** Rotate two neighboring waters by small angles $\theta\sim \mathcal{N}(0,\sigma^2)$ to exchange donor/acceptor roles according to geometric criteria below.

```python
def rotate_water(atoms, water_index: int, axis, angle): ...
def flip_protons(atoms, water_index: int): ...
def rewire_hbond_pair(atoms, water_i: int, water_j: int): ...
```

---

## 7) Docking RC to the surface (sampling/docking.py)

Heuristic “aISS‑like” placement without external tools:

1. **Surface points:** Sample candidate anchor points on the cluster’s **convex hull** or on a sphere of radius $R+\delta$.
2. **Orient RC:** For each anchor, align the reaction axis $\hat{\mathbf{u}}$ to the **local surface normal** $\hat{\mathbf{n}}$ (or to maximize preliminary score, see §9).
3. **Steric filter:** Reject placements where any interatomic distance $< d_\text{vdW, sum} - \delta$.
4. **Local optimization:** Keep surface fixed initially, relax RC with ASE **BFGS** (few steps), then allow selected surface waters to relax.

```python
def dock_reactant(surface: Geometry, rc: Geometry, axis: Tuple[int,int],
                  ntries: int, calc_spec: CalcSpec, workdir: str) -> list[Geometry]:
    """Return a list of RC-on-surface candidates (as merged Geometries)."""
```

---

## 8) Descriptors & physics (calc/descriptors.py)

Let **S** be the set of surface atoms; **Q$_i$** their partial charges (from calculator if available; fallback to model charges). Let **r** be a point characterizing the reactive region (e.g., midpoint of forming bond).

### 8.1 Electric field

* **Definition (SI):**

  $$
  \mathbf{E}(\mathbf{r}) = \frac{1}{4\pi\epsilon_0} \sum_{i\in S} q_i \frac{\mathbf{r}-\mathbf{R}_i}{\lVert \mathbf{r}-\mathbf{R}_i \rVert^3}
  $$
* In code, prefer **atomic units** (set $4\pi\epsilon_0=1$), then convert to V/Å as needed.
* **Projection along reaction axis $\hat{\mathbf{u}}$:**

  $$
  E_\parallel = \mathbf{E}\cdot\hat{\mathbf{u}}
  $$

  A **negative** $E_\parallel$ value stabilizes a TS with dipole aligned **with** $\hat{\mathbf{u}}$ if charges are defined consistently—sign convention documented in code.

### 8.2 Donor/acceptor counts

* Hydrogen bond **geometry test** between water A (donor) and water B (acceptor):

  * Distance: $r_{\text{O}_\text{A}\cdots\text{O}_\text{B}} \le r_c$ (default $3.3$ Å).
  * Angle: $\angle \text{H–O}_\text{A}\cdots\text{O}_\text{B} \ge \theta_c$ (default $140^\circ$).
* Count donors/acceptors **within a cutoff** of the **reactive site** (e.g., within 3.2 Å of the TS heteroatoms).

### 8.3 Proton‑wire metrics (graph‑based)

* Build a **directed graph** $G=(V,E)$ where nodes $V$ are waters; add **edge** $i\to j$ if water $i$ can **donate** an H‑bond to $j$ (passes geometry test).
* For a target **donor node** $s$ near the transferring H and **acceptor node** $t$, compute:

  * **Shortest path length** $L = \sum_{(i\to j)\in P} d(\text{O}_i,\text{O}_j)$.
  * **Max bend** $\phi_\text{max}$ across consecutive edges; each bend is angle between vectors $\text{O}_i\to \text{O}_j$ and $\text{O}_j\to \text{O}_k$.
* Paths with $L$ too long or $\phi_\text{max} > \phi_c$ (e.g., $50^\circ$) are penalized.

### 8.4 Strain penalty

* For all **H‑bonded** O···O pairs (as detected above), penalize deviation from reference $d_0\approx 2.8$ Å:

  $$
  S_\text{strain} = \sum_{\langle i,j\rangle} \left(\frac{d_{ij}-d_0}{\sigma_d}\right)^2, \quad \sigma_d\approx 0.15\ \text{Å}
  $$

### 8.5 Descriptor bundle and normalization

* **Descriptor vector:** $\mathbf{x}=[E_\parallel,\ \#\text{don},\ \#\text{acc},\ L,\ \phi_\text{max},\ S_\text{strain}]$.
* Normalize for scoring via running **z‑scores** over the current candidate pool (or pre‑set scales).

```python
def compute_descriptors(surface_atoms, rc_atoms, reaction_axis: np.ndarray,
                        reactive_point: np.ndarray, charges: np.ndarray) -> DescriptorSet: ...
```

---

## 9) Surrogate score & Monte‑Carlo (sampling/mc\_field.py)

### 9.1 Surrogate objective

We **minimize**:

$$
J(\mathbf{x}) = w_E\,\underbrace{(-E_\parallel)}_{\text{prefer large negative}} + 
w_D\,(\alpha_D\,n_\text{don} + \alpha_A\,n_\text{acc}) + 
w_W\,(a_L L + a_\phi \phi_\text{max}) + 
w_S\,S_\text{strain}
$$

where signs/coefficients ensure **lower is better**. Recommended defaults (dimensionless after normalization): $w_E=1.0$, $w_D=0.5$, $w_W=0.7$, $w_S=0.3$; $\alpha_D=-1, \alpha_A=-0.5, a_L=1, a_\phi=1$.

### 9.2 Moves

* `RotateWater(i, θ~N(0,σ²))`
* `FlipProtons(i)`
* `RewirePair(i, j)`
* Optional small **rigid shift** of a surface water.

### 9.3 Acceptance (Metropolis + Simulated Annealing)

Given current $J$ and proposed $J'$ at **annealing temperature** $T_\text{sa}$:

$$
p_\text{acc} = \min\left(1,\ \exp\left(-\frac{J' - J}{T_\text{sa}}\right)\right)
$$

**Schedule:** geometric $T_\text{sa}(k)=T_0\alpha^k$ with $T_0\sim 1.0$ and $\alpha\in[0.95,0.99]$ per step or per sweep.

### 9.4 Periodic evaluator

Every $M$ accepted moves (e.g., $M=50$), run a **quick ASE optimization** (few BFGS steps; core fixed; calculator on) to avoid drift; recompute descriptors on the relaxed structure; keep **top‑K** surfaces by $J$.

```python
class MCMove: ...
def mc_field_sampler(surface: Geometry, rc: Geometry, reaction_axis, temps_sa: list[float],
                     moves: list[MCMove], steps: int, checkpoint_every: int, top_k: int,
                     calc_spec: CalcSpec, workdir: str) -> list[Geometry]:
    ...
```

---

## 10) TS‑templated surface design (sampling/ts\_template.py)

Place the **TS seed** near the surface, **freeze/soft‑restrain core**, **lightly restrain TS internals**, and **relax waters**:

* **TS restraints:** For a set of key distances $d_k$ (e.g., forming/breaking bonds), add harmonic penalties:

  $$
  E_\text{TS-rest} = \tfrac{1}{2}\sum_k k_k \left(d_k(\mathbf{R})-d_k^\text{seed}\right)^2
  $$
* Optimization target: energy of **\[surface + TS]** with TS held by weak springs, until forces < small threshold. Then **remove TS**; keep the reoriented waters (this is your **TS‑stabilized surface**).

```python
def ts_templated_surface(surface: Geometry, ts_seed: Geometry,
                         calc_spec: CalcSpec, core_indices: list[int],
                         k_internal: float = 0.5, workdir: str = ".") -> Geometry:
    ...
```

---

## 11) Reaction path & TS with ASE (calc/path.py)

### 11.1 NEB initialization

* Build images from **RC → P** by linear interpolation on **reactive atoms** and rigidly carrying waters; then run **IDPP** pre‑optimization (ASE’s `interpolate` with `mic=False` and `idpp=True`).
* Run **NEB** with $N$ images (7–11), **climbing image** after initial convergence.

### 11.2 TS refinement: **Dimer method**

* Take the highest‑energy NEB image as a TS guess; apply ASE **Dimer** to find a first‑order saddle (converge on gradient along lowest mode).

### 11.3 TS verification & connection

* **Vibrations** at TS; require exactly **one** imaginary frequency ($\nu_i$).
* Displace ± along the imaginary mode vector $\mathbf{q}_i$ (small $s$), then **optimize** both to reach RC and P basins—this brackets the TS connects the intended minima.

```python
def neb_ts(rc_atoms, p_atoms, calc_spec: CalcSpec, n_images=9, fmax=0.05, workdir="."):
    """Return ts_atoms, images, energies"""
```

---

## 12) Vibrations & thermochemistry (calc/vibthermo.py)

### 12.1 Finite‑difference Hessian (ASE Vibrations)

* Use ASE `Vibrations` to compute normal modes $\{\nu_j\}$ (cm⁻¹), including TS with one imaginary.

### 12.2 Quasi‑RRHO (qRRHO) at low T

At 10–20 K, the vibrational contributions are small but **low‑frequency modes** are problematic. Use **quasi‑harmonic** damping:

* Set a **frequency cutoff** $\nu_c \sim 100\ \text{cm}^{-1}$.
* For modes $\nu_j < \nu_c$, replace contributions by a **modified** partition factor:

  $$
  S_j^\text{qRRHO}(T) = f(\nu_j,\nu_c)\,S_j^\text{HO}(T), \quad
  f = \frac{\nu_j}{\nu_c}
  $$

  or clamp $\nu_j \to \nu_c$ in the harmonic formulas (choose one policy and document).

### 12.3 Thermo formulas (harmonic oscillator)

* Angular frequency $\omega_j = 2\pi c\,\nu_j$ (rad/s); $\beta=1/k_B T$.
* **Zero‑point energy:** $\text{ZPE} = \tfrac{1}{2}\sum_j \hbar \omega_j$.
* **Vibrational contributions:**

  $$
  E_\text{vib}(T) = \sum_j \left( \tfrac{1}{2}\hbar\omega_j + \frac{\hbar\omega_j}{e^{\beta\hbar\omega_j}-1} \right),
  \quad
  S_\text{vib}(T) = k_B \sum_j \left[ \frac{\beta\hbar\omega_j}{e^{\beta\hbar\omega_j}-1} - \ln\left(1-e^{-\beta\hbar\omega_j}\right) \right].
  $$
* **Gibbs free energy:** $ G(T) = E_\text{elec} + E_\text{trans+rot} + E_\text{vib}(T) - T\,S_\text{vib}(T)$.

  * For **clusters**, **trans/rot** should be treated consistently across RC/TS/P (they mostly cancel); you may omit or keep ideal‑gas formulas at very low T and comment this choice.
* **Activation free energy:** $\Delta G^\ddagger(T)=G_\text{TS}(T)-G_\text{RC}(T)$.

```python
def vibrational_analysis(atoms_ts, atoms_rc, atoms_p, calc_spec: CalcSpec, T_list: list[float],
                         nu_cutoff_cm1: float = 100.0) -> dict:
    """Return frequencies, ZPEs, G(T) for RC/TS/P and ΔG‡(T)."""
```

---

## 13) Kinetics & tunneling (calc/kinetics.py)

### 13.1 Eyring TST

$$
k_\text{TST}(T) = \kappa(T)\,\frac{k_B T}{h}\,\exp\!\left(-\frac{\Delta G^\ddagger(T)}{k_B T}\right)
$$

### 13.2 Wigner tunneling correction (simple, uses $|\omega_i|$)

Let $\omega_i$ be the magnitude of the **imaginary angular frequency** of the TS (rad/s):

$$
\kappa_\text{Wigner}(T) = 1 + \frac{1}{24}\left(\frac{\hbar \omega_i}{k_B T}\right)^2
$$

Convert from **cm⁻¹** to $\omega$: $\omega = 2\pi c\,\nu$, $c=2.9979\times10^{10}\ \text{cm s}^{-1}$.

### 13.3 Eckart tunneling (optional)

Given forward/backward barrier heights $V_f, V_b$ and **imaginary frequency** $\nu_i$, define an **Eckart potential** and compute the **transmission coefficient** $\kappa_\text{Eckart}(T)$ by thermal averaging:

$$
\kappa_\text{Eckart}(T) = \frac{\int_0^\infty P(E)\,e^{-E/k_B T}\,dE}{\int_0^\infty e^{-E/k_B T}\,dE}
$$

where $P(E)$ is the analytical transmission probability for the Eckart barrier (closed‑form; implement in a helper, or use quadrature).

```python
def eyring_rate(deltaG_eV: float, T: float, kappa: float) -> float: ...
def wigner_kappa(nu_imag_cm1: float, T: float) -> float: ...
def eckart_kappa(deltaE_fwd_eV: float, deltaE_rev_eV: float, nu_imag_cm1: float, T: float) -> float:
    """Optional; implement via standard Eckart formulas."""
```

---

## 14) Evaluating a surface (orchestrators/pipeline.py)

```python
def evaluate_surface(surface: Geometry, reaction: ReactionSpec, calc_spec: CalcSpec,
                     core_indices: list[int], temps: list[float], workdir: str) -> EvalResult:
    """
    Steps:
    1) Build RC-on-surface via docking; apply constraints; local opt (BFGS until fmax ~ 0.05 eV/Å).
    2) Path: NEB(IDPP) + Cl climbing; extract highest image; Dimer refine to TS.
    3) Vibrations on RC, TS, P; quasi-RRHO; ΔG‡(T) for temps; Wigner κ(T).
    4) Eyring k(T) with κ(T).
    5) Descriptors @ RC: E-field, donors/acceptors, proton-wire, strain.
    6) Save geometries and JSON; return EvalResult.
    """
```

---

## 15) End‑to‑end search (orchestrators/pipeline.py)

```python
def search_best_surfaces(reaction: ReactionSpec, surface_spec: SurfaceSpec,
                         calc_spec: CalcSpec, temps: list[float],
                         n_ts_templates: int, n_mc_rounds: int, max_evals: int,
                         workdir: str) -> list[EvalResult]:
    """
    A) Build base N-water cluster; select core; make constraints.
    B) Generate n_ts_templates TS-templated surfaces (vary placement/orientation).
    C) For each, run MC field sampler (few hundred moves) to diversify.
    D) Evaluate up to max_evals candidates fully (evaluate_surface).
    E) Rank by ΔG‡(T_min) and k(T); return top results.
    """
```

---

## 16) Optional Bayesian optimization (optimize/gp\_surrogate.py)

* **Inputs:** Descriptor vectors $\mathbf{x}_i$ and measured targets $y_i=\Delta G_i^\ddagger$ (eV).
* **GP model:** Zero‑mean, kernel $k(\mathbf{x},\mathbf{x}')=\sigma_f^2 \exp(-\tfrac{1}{2}\sum_j ((x_j-x_j')/\ell_j)^2)+\sigma_n^2\delta$.
* **Acquisition:** **Expected Improvement** at current best $y^*$:

  $$
  \text{EI}(\mathbf{x})=
  \begin{cases}
    (y^*-\mu)\Phi(z)+\sigma\phi(z), & \sigma>0\\
    0, & \sigma=0
  \end{cases}
  \quad z=\frac{y^*-\mu}{\sigma}
  $$

  where $\mu,\sigma$ are GP predictive mean/std; $\Phi,\phi$ CDF/PDF of $\mathcal{N}(0,1)$.
* **Loop:** Propose descriptor targets → **invert** to surface edits using MC initial seeds (bias towards moves that improve the closest descriptors).

```python
def gp_fit(X: np.ndarray, y: np.ndarray) -> Any: ...
def gp_suggest(gp, bounds: list[tuple[float,float]], n: int) -> np.ndarray: ...
```

---

## 17) CLI & config (cli.py, config.py)

* **YAML** config keys: `reaction`, `surface`, `calc`, `temps`, `search`.
* **Commands:**

  * `wtr init --config config.yml`
  * `wtr run  --config config.yml`
  * `wtr report --run runs/run001 --top 10`

All CLI commands produce machine‑readable JSON artifacts for agent tools.

---

## 18) Agent tool specs (JSON‑friendly functions)

* `create_cluster(surface_spec) -> {"cluster_id": ..., "path": ...}`
* `generate_ts_surfaces(cluster_id, reaction_id, k) -> [surface_id]`
* `mc_refine_surfaces(surface_ids, nsteps, ...) -> [surface_id]`
* `evaluate_surface(surface_id, reaction_id, temps) -> EvalResult`
* `rank_surfaces(run_id, temperature, top) -> [EvalResult]`

Each tool accepts/returns **pure JSON** and logs a stable `run_id`.

---

## 19) Units & constants (utils/units.py)

* eV↔J: $1\ \text{eV}=1.602176634\times10^{-19}\ \text{J}$.
* cm⁻¹ → angular frequency: $\omega=2\pi c\,\nu$.
* Boltzmann: $k_B=8.617333262\times10^{-5}\ \text{eV/K}$.
* Planck: $h=4.135667696\times10^{-15}\ \text{eV·s}$, $\hbar=h/2\pi$.
* Speed of light: $c=2.99792458\times10^{10}\ \text{cm/s}$.
* Coulomb prefactor for E‑field **if** using SI; else do A.U. and convert at output.

---

## 20) Testing strategy

* **Unit:** rotations, proton flips preserve OH lengths; descriptor values on toy geometries; Wigner κ vs analytical values; Eyring rates.
* **Integration:** tiny 6‑water toy reaction (or formal RC/TS/P mini set) → NEB → Dimer → 1 imaginary frequency; ensure ΔG‡ ordering stable across seeds.
* **Determinism:** capture RNG seeds for moves; record ASE + calculator versions.

---

## 21) Minimal pseudocode wiring (for the agent)

```python
from WTr.models.datatypes import ReactionSpec, SurfaceSpec, CalcSpec
from WTr.geom.build import build_water_cluster
from WTr.geom.constraints import select_core_atoms
from WTr.sampling.ts_template import ts_templated_surface
from WTr.sampling.mc_field import mc_field_sampler
from WTr.orchestrators.pipeline import evaluate_surface

# Load reaction from files (RC.xyz, P.xyz, optional TS_seed.xyz) -> ReactionSpec
reaction = ReactionSpec(...)

surface_spec = SurfaceSpec(waters_n=20, radius=8.0, core_fraction=0.5, random_seed=42)
calc_spec = CalcSpec(ase_calculator="xtb", calc_kwargs={"method":"GFN2-xTB","accuracy":1.0})

cluster = build_water_cluster(surface_spec)
core = select_core_atoms(cluster, surface_spec)

# Generate TS-templated surfaces
ts_surfaces = [ts_templated_surface(cluster, reaction.ts_seed, calc_spec, core, k_internal=0.5,
                                    workdir=f"runs/ts{i}") for i in range(40)]

# MC refinement on a few
refined = []
for i,s in enumerate(ts_surfaces[:10]):
    refined += mc_field_sampler(s, reaction.reactant, reaction.reaction_axis_atoms,
                                temps_sa=[1.0,0.9,0.81,0.73], moves=[...], steps=500,
                                checkpoint_every=50, top_k=5, calc_spec=calc_spec,
                                workdir=f"runs/mc{i}")

# Evaluate candidates
temps = [10.0, 20.0]
results = [evaluate_surface(surf, reaction, calc_spec, core, temps, workdir=f"runs/eval{k}")
           for k,surf in enumerate(refined)]

# Rank & export
best = sorted(results, key=lambda r: r.deltaG_dagger_10K)[:10]
```

---

## 22) Practical defaults & notes

* **Calculator default:** `XTB(method="GFN2-xTB", accuracy=1.0)`. Expose `charge`/`uhf` for radicals (OH).
* **Fixing the core:** Start with **hard `FixAtoms`**; enable **Hookean** only if you see edge artifacts.
* **NEB:** start with 9 images, force convergence `fmax=0.05 eV/Å`; tighten for final runs.
* **Vibrations:** finite differences ±0.01 Å; confirm force consistency; reuse Hessians for thermo.
* **Low‑T:** include Wigner κ; at 10 K it matters a lot for H‑transfer.

---
