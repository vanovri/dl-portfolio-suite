"""Exact complex-valued differentiable PyTorch Variational Quantum Circuit (VQC) statevector simulator."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class VariationalQuantumLayer(nn.Module):
    """Exact 4-qubit complex64 statevector VQC with Angle Encoding, RY/RZ ansatz, ring CNOT, and Pauli-Z readout.

    Statevector space dimension is 2^n_qubits (default 4 qubits -> dim=16).
    All operations are differentiable via native PyTorch complex autograd on CPU or CUDA.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        if n_qubits < 2 or n_qubits > 8:
            raise ValueError("n_qubits must be in [2, 8] for fast exact statevector simulation.")
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.hilbert_dim = 1 << self.n_qubits

        # Trainable variational rotation parameters: RY(theta) and RZ(phi) per layer and qubit.
        self.theta = nn.Parameter(torch.empty(self.n_layers, self.n_qubits))
        self.phi = nn.Parameter(torch.empty(self.n_layers, self.n_qubits))
        nn.init.uniform_(self.theta, -0.5, 0.5)
        nn.init.uniform_(self.phi, -0.5, 0.5)

        # Precompute exact CNOT ring permutation indices: CNOT(0->1, 1->2, ..., n-1->0).
        cnot_perms: list[list[int]] = []
        for q in range(self.n_qubits):
            ctrl = q
            targ = (q + 1) % self.n_qubits
            cnot_perms.append(self._build_cnot_permutation(self.n_qubits, ctrl, targ))
        self.register_buffer("cnot_perms", torch.tensor(cnot_perms, dtype=torch.long))

        # Precompute exact Pauli-Z diagonal eigenvalues (+1 if qubit bit is 0, -1 if bit is 1).
        z_diag = torch.zeros(self.n_qubits, self.hilbert_dim, dtype=torch.float32)
        for q in range(self.n_qubits):
            shift = self.n_qubits - 1 - q
            for basis_idx in range(self.hilbert_dim):
                bit = (basis_idx >> shift) & 1
                z_diag[q, basis_idx] = 1.0 if bit == 0 else -1.0
        self.register_buffer("z_diag", z_diag)

    @staticmethod
    def _build_cnot_permutation(n_qubits: int, control: int, target: int) -> list[int]:
        ctrl_bit = 1 << (n_qubits - 1 - control)
        targ_bit = 1 << (n_qubits - 1 - target)
        perm: list[int] = []
        for idx in range(1 << n_qubits):
            if idx & ctrl_bit:
                perm.append(idx ^ targ_bit)
            else:
                perm.append(idx)
        return perm

    def _apply_ry(self, state: torch.Tensor, qubit: int, angle: torch.Tensor) -> torch.Tensor:
        """Apply single-qubit RY(angle) to batched statevector (B, 2^n).

        RY(a) = [[cos(a/2), -sin(a/2)], [sin(a/2), cos(a/2)]].
        """
        bsz = state.size(0)
        half = 0.5 * angle
        c = torch.cos(half).to(torch.complex64)  # (B,)
        s = torch.sin(half).to(torch.complex64)  # (B,)

        shape = [bsz] + [2] * self.n_qubits
        psi = state.view(*shape)
        axis = qubit + 1
        # Move target qubit to last axis for vectorized 2-component linear combination.
        psi_t = psi.movedim(axis, -1)  # (B, ..., 2)
        a0 = psi_t[..., 0]
        a1 = psi_t[..., 1]

        view_dims = [bsz] + [1] * (self.n_qubits - 1)
        c_v = c.view(*view_dims)
        s_v = s.view(*view_dims)

        new0 = c_v * a0 - s_v * a1
        new1 = s_v * a0 + c_v * a1
        out_t = torch.stack([new0, new1], dim=-1)
        return out_t.movedim(-1, axis).reshape(bsz, self.hilbert_dim)

    def _apply_rz(self, state: torch.Tensor, qubit: int, angle: torch.Tensor) -> torch.Tensor:
        """Apply single-qubit RZ(angle) = diag(exp(-i*a/2), exp(+i*a/2)) to batched statevector."""
        bsz = state.size(0)
        half = 0.5 * angle
        e_neg = torch.complex(torch.cos(half), -torch.sin(half))
        e_pos = torch.complex(torch.cos(half), torch.sin(half))

        shape = [bsz] + [2] * self.n_qubits
        psi = state.view(*shape)
        axis = qubit + 1
        psi_t = psi.movedim(axis, -1)

        view_dims = [bsz] + [1] * (self.n_qubits - 1)
        new0 = e_neg.view(*view_dims) * psi_t[..., 0]
        new1 = e_pos.view(*view_dims) * psi_t[..., 1]
        out_t = torch.stack([new0, new1], dim=-1)
        return out_t.movedim(-1, axis).reshape(bsz, self.hilbert_dim)

    def _apply_cnot_ring(self, state: torch.Tensor) -> torch.Tensor:
        for q in range(self.n_qubits):
            state = state.index_select(dim=1, index=self.cnot_perms[q])
        return state

    def evolve_state(self, angles: torch.Tensor, init_state: torch.Tensor | None = None) -> torch.Tensor:
        """Evolve |0...0> (or provided complex statevector) through encoding + L variational layers."""
        # Complex64 ops must run outside FP16 AMP autocast for numerical precision and CUDA support.
        with torch.amp.autocast(device_type=angles.device.type, enabled=False):
            angles_f32 = angles.float()
            bsz = angles_f32.size(0)
            if angles_f32.size(1) != self.n_qubits:
                raise ValueError(f"Expected angles shape (B, {self.n_qubits}), got {tuple(angles_f32.shape)}")

            if init_state is None:
                state = torch.zeros(
                    (bsz, self.hilbert_dim), dtype=torch.complex64, device=angles_f32.device
                )
                state[:, 0] = 1.0 + 0.0j
            else:
                state = init_state.to(device=angles_f32.device, dtype=torch.complex64)

            # 1. Angle Encoding layer: RY(x_q) on each qubit q.
            for q in range(self.n_qubits):
                state = self._apply_ry(state, q, angles_f32[:, q])

            # 2. L Variational layers: RY(theta) -> RZ(phi) -> Entangling Ring CNOT.
            for layer in range(self.n_layers):
                for q in range(self.n_qubits):
                    th = self.theta[layer, q].expand(bsz)
                    ph = self.phi[layer, q].expand(bsz)
                    state = self._apply_ry(state, q, th)
                    state = self._apply_rz(state, q, ph)
                state = self._apply_cnot_ring(state)

            return state

    def measure_pauli_z(self, state: torch.Tensor) -> torch.Tensor:
        """Compute exact expectation values <psi | Z_k | psi> in [-1, 1] for k=0..n_qubits-1."""
        probs = state.real.pow(2) + state.imag.pow(2)  # (B, 2^n)
        expectations = probs @ self.z_diag.t()  # (B, n_qubits)
        return expectations.clamp(-1.0, 1.0)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        state = self.evolve_state(angles)
        return self.measure_pauli_z(state)

    @torch.no_grad()
    def build_circuit_unitary(self, single_angles: torch.Tensor) -> torch.Tensor:
        """Construct the exact (2^n, 2^n) complex unitary operator U(x, theta, phi)."""
        x = single_angles.detach().float().view(1, self.n_qubits)
        x_rep = x.expand(self.hilbert_dim, self.n_qubits)
        basis = torch.eye(self.hilbert_dim, dtype=torch.complex64, device=x.device)
        # Column j of U is the evolved j-th computational basis vector.
        evolved = self.evolve_state(x_rep, init_state=basis)  # (dim, dim) where row j is U|j>
        return evolved.t().contiguous()

    @torch.no_grad()
    def verify_unitary_and_norm(self, sample_angles: torch.Tensor | None = None) -> dict[str, Any]:
        """Verify U^dagger U == I and statevector normalization sum_j |alpha_j|^2 == 1.0."""
        device = self.theta.device
        if sample_angles is None:
            sample_angles = torch.linspace(-1.2, 1.2, self.n_qubits, device=device).unsqueeze(0)
        else:
            sample_angles = sample_angles.to(device=device, dtype=torch.float32)
            if sample_angles.ndim == 1:
                sample_angles = sample_angles.unsqueeze(0)

        u_mat = self.build_circuit_unitary(sample_angles[0])
        ident = torch.eye(self.hilbert_dim, dtype=torch.complex64, device=device)
        prod = u_mat.conj().t() @ u_mat
        unitary_max_error = float(torch.abs(prod - ident).max().item())

        states = self.evolve_state(sample_angles)
        norms = (states.real.pow(2) + states.imag.pow(2)).sum(dim=-1)
        norm_max_error = float(torch.abs(norms - 1.0).max().item())

        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "hilbert_dim": self.hilbert_dim,
            "unitary_max_error": unitary_max_error,
            "norm_max_error": norm_max_error,
            "is_unitary": bool(unitary_max_error < 1e-4),
            "is_normalized": bool(norm_max_error < 1e-4),
        }
