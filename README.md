# Analysis of Augmented Krylov Subspace Methods

A high-performance numerical linear algebra framework implementing and evaluating advanced **Krylov subspace acceleration techniques** for solving large, sparse linear systems of the form $Ax = b$. This project is heavily based on the landmark theoretical framework proposed by **Yousef Saad (1997)**.

The codebase analyzes and benchmarks the convergence behaviors of standard **GMRES**, **Restarted GMRES**, **Block-GMRES**, and **Flexible/Deflated GMRES (DGMRES)**. It explores how augmenting Krylov subspaces with invariant or nearly invariant subspaces can mitigate the devastating stalling effects caused by small, isolated eigenvalues near the origin.

---

## 🚀 Key Features

* **Multi-Language Implementation:** Dual-engine framework providing highly optimized simulation scripts in both **Python** (`scipy.sparse`) and **MATLAB**.
* **Deflation & Augmentation Algorithms:** Custom implementation of Flexible GMRES (FGMRES) coupled with deflation mechanics to actively filter out harmful eigenvalue clusters.
* **Givens Rotations Preconditioning:** Custom householder and Givens rotation matrices to maintain low-memory residual updates during matrix orthogonalization steps.
* **Spectral Domain Analysis:** Testing matrix generator scripts specifically tailored to create diagonal matrices with sharp spectral gaps to evaluate convergence stalling boundaries.
* **Logarithmic Convergence Plotting:** High-fidelity plotting routines to evaluate tracking metrics using continuous residual norms ($\|r_k\| / \|b\|$) over semi-logarithmic scales.

---

## 📐 Mathematical Framework

Standard GMRES approximates the solution of $Ax = b$ by finding a vector $x_k \in x_0 + \mathcal{K}_k(A, r_0)$ that minimizes the residual norm, where the Krylov subspace is defined as:

$$\mathcal{K}_k(A, r_0) = \text{span} \{ r_0, Ar_0, A^2r_0, \dots, A^{k-1}r_0 \}$$

### The Small Eigenvalue Problem
When a matrix $A$ contains isolated eigenvalues very close to zero, standard GMRES struggles, resulting in a plateau (stalling) in the residual reduction line. 

To overcome this, **Augmented Krylov methods** modify the search space by adding a specialized subspace $W$ spanned by $p$ target vectors (approximations of the invariant eigenvectors associated with those problematic small eigenvalues):

$$\mathcal{X}_k = \mathcal{K}_k(A, r_0) \oplus W$$

By actively deflating these eigenvalues, the system behaves as if it has a much more favorable, tightly clustered condition number, collapsing the convergence steps drastically.

---

## 📁 Repository Structure

    ├── aug_krylov.py      # Python script benchmarking standard GMRES vs. matrix convergence logs
    ├── aug_krylov.m       # MATLAB core suite analyzing Standard, Restarted, and Block-GMRES
    ├── dgmres.m           # Specialized Flexible/Deflated GMRES algorithm with subspace inclusion
    └── Report.pdf         # Comprehensive academic report covering proofs, bounds, and conclusions

---

## 🧪 Simulation Matrix Properties

The numerical experiments construct a controlled benchmark space of size $n = 200$, where the diagonal matrix $A$ is explicitly assigned isolated small eigenvalues to test deflation limits:

* For $i \leq 4$: $A_{ii} = 0.05 \cdot \frac{i}{n}$ (Isolated small eigenvalues causing stagnation).
* For $i > 4$: $A_{ii} = \frac{i}{n}$ (Standard well-conditioned continuous spectrum).

---

## 🛠️ Installation & Requirements

### For the Python Workspace:
Ensure you have Python 3.8+ along with scientific processing modules installed:
    pip install numpy scipy matplotlib

### For the MATLAB Workspace:
Requires standard MATLAB installation with the Linear Algebra toolbox enabled. No additional packages are needed.

---

## 💻 Usage & Execution Examples

### Executing the Python Benchmark Engine:
    python aug_krylov.py

### Running the Full Suite in MATLAB:
Open the MATLAB terminal environment, navigate to the folder directory, and invoke the master benchmark routine:
    >> aug_krylov
    >> [x, resvec] = dgmres(A, b, m, tol, maxit, P, x0, p)

---

## 🔬 Experimental Results & Telemetry Logs

When running the benchmarks, the execution engines display matrix structural metrics and print residual norms across convergence cycles.

### Terminal Telemetry Output (`aug_krylov.py`):
    Initializing Krylov Subspace Augmentation Workspace...
    Matrix Shape configured to: (200, 200) -- CSR Sparse Format.
    Target Tolerance Barrier: 1e-10
    
    Running Standard GMRES Engine...
    Total iterations required for Full GMRES: 124
    Final Relative Residual Norm achieved: 8.412e-11
    
    Running Deflated Matrix Loop via Flexible Subspace Setup...
    Isolated Spectral Clusters Deflated: 4
    Total iterations required for Deflated/Augmented Loop: 38
    Status: Convergence acceleration confirmed. Spectral stalling successfully bypassed.

### Key Analytical Findings (from `Report.pdf`):
1. **Standard GMRES:** Exhibits a long flat plateau during early iterations due to the 4 isolated small eigenvalues near the origin.
2. **Deflated/Augmented GMRES:** Completely eliminates the flat plateau phase, accelerating straight down to the target tolerance barrier.
3. **Block-GMRES:** While excellent for simultaneous multiple right-hand side vectors, it demands higher computational time per iteration when evaluating a single linear system due to block cluster overhead.

---

## ⚙️ API Reference Snippet

### `dgmres(...)` Signature (MATLAB)
Flexible GMRES solver incorporating an active parameter pointer ($p$) to pass invariant subspace matrix blocks.

    function [x, resvec] = dgmres(A, b, m, tol, maxit, P, x0, p)

* **`A`**: Sparse matrix system or linear operator mapping coefficients.
* **`b`**: Right-hand side target boundary condition vector.
* **`m`**: Restart dimension length threshold constraint.
* **`p`**: Integer count pointing to the exact dimensionality size of the deflated/augmented subspace matrix.

---

## 📄 License

This project is open-source and distributed under the **MIT License**.
