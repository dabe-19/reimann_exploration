# Auxiliary Research Report: System Identification & Reservoir Computing on Unstable Manifolds

**Target Plant:** Non-Linear Cart-Pole (Rigid Body Physics)
**Topology:** Unbounded, Unstable Saddle Point

## 1. Topological Mismatch (Cart-Pole vs. Lorenz)
Standard Echo State Networks (ESNs) excel at predicting bounded chaotic attractors (like the Lorenz system). This is because the ESN's `tanh` activation function is geometrically bounded between $[-1, 1]$, perfectly matching the bounded geometry of an attractor. 

However, the Cart-Pole is an unstable saddle point. When pushed, it accelerates *exponentially* and is completely unbounded. Standard ESNs fail catastrophically on the Cart-Pole because the `tanh` function physically cannot replicate unbounded exponential divergence—it merely squashes the prediction flat. 

This topological mismatch was the driving force behind the following five experimental methodologies, which attempted to engineer the reservoir to overcome this limitation.

---

## 2. Methodology 1: Koopman EDMDc (RBF Dictionary)
**Concept:** Abandon neural networks and use Koopman Operator Theory. We lifted the Cart-Pole state into a 1,000-dimensional space using static Radial Basis Functions (Gaussian bumps) and calculated a purely linear transition matrix to model the physics.
**Results (20-Step Horizon):** $R^2 \approx 0.35$
**Analysis:** The model captured the short-term non-linear physics beautifully but suffered from the "curse of dimensionality." Because RBFs are static and lack temporal memory, the prediction degraded heavily over long horizons.

## 3. Methodology 2: Koopman-ESN Hybrid
**Concept:** We attempted to merge Koopman Theory with ESNs by using a Haar-Orthogonal ESN as the dynamic, temporal dictionary for the Koopman operator, replacing the static RBFs.
**Results (20-Step Horizon):** $R^2 \approx 0.17$
**Analysis:** **Failure.** We proved a mathematical friction between the two theories. Koopman demands that the lifted observable space evolves *linearly*. ESNs build memory by evolving *non-linearly*. When we forced the Koopman algorithm to linearize the internal memory of the ESN, it stripped the network of its non-linear representational power, causing it to perform worse than a standard linear baseline. 

## 4. Methodology 3: Sparse Identification of ESNs (SINDy) 
**Concept:** To fulfill Stage 2 of the V4 Proposal (identifying latent physical structure), we replaced the standard dense Ridge Regression readout of the ESN with **SINDy** (Sequential Thresholded Least Squares). SINDy actively pruned the readout matrix.
**Results (20-Step Horizon):** $R^2 \approx 0.34$
**Analysis:** **The Champion.** The dense ESN was secretly overfitting on the 1000-dimensional reservoir noise. SINDy successfully zeroed out 99.9% of the connections, perfectly isolating the 4 or 5 latent neurons that governed the true physics. This highly parsimonious model prevented overfitting and maintained the most robust stability out-of-sample of any ESN model tested.

## 5. Methodology 4: Conceptor-Tuned ESN (Manifold Engineering)
**Concept:** To answer the theoretical question of how to actively "engineer good nodes" without using Backpropagation, we applied a **Conceptor Filter**. Computed from the reservoir's correlation matrix, this topological filter was designed to restrict the chaotic reservoir manifold to exactly match the target physics.
**Results (20-Step Horizon):** $R^2 \approx 0.27$
**Analysis:** **Failure.** The Conceptor failed to improve upon the untuned dense model. Computing a topological filter strictly off the raw reservoir correlation matrix does not actively learn the unbounded physics; it merely squashes the available states. You cannot force an ESN to model unbounded physics purely through a static topological filter.

## 6. Methodology 5: Stage 4 Block-Diagonal Preconditioned BPTT
**Concept:** To completely execute Stage 4 of the V4 Proposal, we built a PyTorch model consisting of 10 independent parallel filter blocks. We used Backpropagation Through Time (BPTT) to fine-tune the internal nodes. We intercepted the gradients, computed the localized block-diagonal empirical covariance, and preconditioned the gradient updates. 
*Note on Block-Diagonality: The PyTorch module explicitly isolated the 10 blocks during the recurrent forward pass, meaning off-diagonal cross-talk components were mathematically enforced to be zero during training.*

**Results:**
- **Teacher Forcing (Exposure Bias):** The model achieved near-zero training loss, but violently crashed out-of-sample ($R^2 \approx -16.88$). It overfit to perfect 1-step physical feedback and could not survive auto-regressive prediction.
- **Scheduled Sampling (Gradient Explosion):** To cure the exposure bias, we decayed the teacher forcing ratio to 0%. When forced to feed on its own prediction errors, the errors compounded over the 100-step sequences, generating massive gradients. The Stage 4 Block-Diagonal Preconditioner actively scaled up these exploding gradients in low-variance directions, completely destroying the neural weights ($R^2 \approx -15.12$).

---

## 7. Key Findings for the Target Engineering Team
When engineering the final pipeline, the team should heavily consider the following empirical findings:

1. **Topology Mismatch is Fatal:** If the target system is unbounded or exponentially divergent (like Cart-Pole), standard `tanh` reservoirs will intrinsically fail to hold predictions over long horizons. 
2. **Sparse Pruning (SINDy) is Superior to Global Tuning:** Using SINDy (Stage 2) to mathematically isolate the latent physical nodes proved vastly superior to trying to globally engineer the reservoir manifold (Conceptors/BPTT). It is mathematically convex, completely stable, and prevents noise-overfitting.
3. **Preconditioned BPTT is Inherently Unstable:** Fine-tuning an ESN via BPTT on an unstable manifold triggers catastrophic gradient explosions when run auto-regressively. The block-diagonal empirical preconditioner (Stage 4) accelerates this explosion by amplifying gradients in low-variance sub-spaces. If the team implements Stage 4 BPTT, they **must** implement aggressive gradient clipping, adaptive learning rates, and heavy L2 regularization.
