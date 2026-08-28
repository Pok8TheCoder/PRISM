"""
PRISM - Latent Dynamics World Model (VAE-based, inspired by PlaNet/Dreamer)
Encodes observed state into a stochastic latent space, learns transition
dynamics in latent space, and predicts infiltration from latent state.

Flow:
  Encode: S_t -> q(z_t | S_t)  (variational encoder, reparameterised)
  Dynamics: z_{t+1} = f_theta(z_t)  (deterministic GRU transition)
  Decode:  z_t -> S_t_hat  (reconstruction for self-supervision)
  Predict: z_t -> (attack_prob, mitre_stage)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import ClassificationHead
from src.utils.constants import NUM_MITRE_STAGES

logger = logging.getLogger("prism.models.latent_dynamics")


class VariationalEncoder(nn.Module):
    """Encode S_t -> (mu_z, logvar_z) in latent space."""

    def __init__(self, d_state: int, d_latent: int, dropout: float = 0.1):
        super().__init__()
        d_hidden = max(d_state, d_latent * 2)
        self.net = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(d_hidden // 2, d_latent)
        self.logvar_head = nn.Linear(d_hidden // 2, d_latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (..., d_state) -> (mu, logvar), each (..., d_latent)"""
        h = self.net(x)
        return self.mu_head(h), self.logvar_head(h)


class StateDecoder(nn.Module):
    """Decode z_t -> S_t (reconstruction head)."""

    def __init__(self, d_latent: int, d_state: int, dropout: float = 0.1):
        super().__init__()
        d_hidden = max(d_state, d_latent * 2)
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_state),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GRUDynamicsCore(nn.Module):
    """
    Deterministic state transition in latent space using a GRU.
    Takes z_t and the GRU hidden state h_{t-1}, returns z_{t+1} deterministic
    prior and updated hidden state.
    """

    def __init__(self, d_latent: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_latent,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        # Project GRU hidden state to latent prior
        self.prior_mu = nn.Linear(d_model, d_latent)
        self.prior_logvar = nn.Linear(d_model, d_latent)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z_seq: torch.Tensor,
        h_prev: torch.Tensor | None = None,
    ) -> dict:
        """
        z_seq  : (B, L, d_latent) — sequence of latent states
        h_prev : (1, B, d_model)  — initial GRU hidden state (optional)

        Returns dict:
            h_seq       : (B, L, d_model) — GRU outputs
            h_last      : (1, B, d_model) — final GRU hidden state
            prior_mu    : (B, L, d_latent) — prior mean per step
            prior_logvar: (B, L, d_latent) — prior log-variance per step
        """
        h_seq, h_last = self.gru(z_seq, h_prev)   # (B, L, d_model), (1, B, d_model)
        h_seq = self.dropout(h_seq)
        prior_mu = self.prior_mu(h_seq)
        prior_logvar = self.prior_logvar(h_seq)
        return {
            "h_seq": h_seq,
            "h_last": h_last,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
        }


def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterisation trick: z = mu + eps * sigma."""
    if not torch.is_grad_enabled():
        return mu
    std = (0.5 * logvar.clamp(-10.0, 2.0)).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


class LatentDynamicsWorldModel(nn.Module):
    """
    VAE-based world model with GRU latent transition dynamics.

    Training losses:
      1. Reconstruction loss  : MSE(decoded S_t, actual S_t)
      2. KL divergence        : KL(q(z_t|S_t) || p(z_t|z_{t-1}))
      3. Infiltration CE loss
      4. MITRE stage CE loss
    """

    def __init__(
        self,
        d_state: int = 110,
        d_latent: int = 64,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        num_mitre_stages: int = NUM_MITRE_STAGES,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_latent = d_latent
        self.d_model = d_model

        self.encoder = VariationalEncoder(d_state, d_latent, dropout)
        self.decoder = StateDecoder(d_latent, d_state, dropout)
        self.dynamics = GRUDynamicsCore(d_latent, d_model, dropout)

        # Classification heads operate on GRU hidden state
        self.infiltration_head = ClassificationHead(d_model, 2, head_dropout)
        self.mitre_head = ClassificationHead(d_model, num_mitre_stages, head_dropout)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "LatentDynamicsWorldModel: d_state=%d, d_latent=%d, "
            "d_model=%d, params=%.2fM",
            d_state, d_latent, d_model, n_params / 1e6,
        )

    def forward(
        self,
        state_seq: torch.Tensor,
        h_prev: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        state_seq : (B, L, d_state)
        h_prev    : (1, B, d_model) optional prior GRU hidden state

        Returns
        -------
        dict:
            pred_state_mean  : (B, d_state) — predicted next state mean
            pred_state_logvar: (B, d_state) — predicted next state logvar
            pred_binary      : (B, 2)
            pred_mitre       : (B, N_stages)
            hidden           : (B, d_model)
            recon_seq        : (B, L, d_state) — reconstructed input
            posterior_mu     : (B, L, d_latent)
            posterior_logvar : (B, L, d_latent)
            prior_mu         : (B, L, d_latent)
            prior_logvar     : (B, L, d_latent)
            h_last           : (1, B, d_model)
        """
        B, L, _ = state_seq.shape

        # --- Encode all steps ---
        post_mu, post_logvar = self.encoder(state_seq)   # (B, L, d_latent)

        # --- Sample posterior latents ---
        z_seq = reparameterise(post_mu, post_logvar)      # (B, L, d_latent)

        # --- Decode (reconstruction) ---
        recon_seq = self.decoder(z_seq)                   # (B, L, d_state)

        # --- GRU dynamics (shift by 1: predict z_t from z_{t-1}) ---
        # Input to GRU is z_{0..L-1}, we get prior for z_{1..L}
        dyn_out = self.dynamics(z_seq, h_prev)

        h_seq = dyn_out["h_seq"]                          # (B, L, d_model)
        h_t = h_seq[:, -1, :]                             # (B, d_model)

        # Next state prediction: use prior from last dynamics step
        pred_mean = dyn_out["prior_mu"][:, -1, :]         # (B, d_latent)
        pred_logvar = dyn_out["prior_logvar"][:, -1, :]   # (B, d_latent)

        # Decode predicted latent to state space
        pred_state_mean = self.decoder(pred_mean)
        pred_state_logvar = pred_logvar.mean(-1, keepdim=True).expand(
            -1, self.d_state
        )

        pred_binary = self.infiltration_head(h_t)
        pred_mitre = self.mitre_head(h_t)

        return {
            "pred_state_mean": pred_state_mean,
            "pred_state_logvar": pred_state_logvar,
            "pred_binary": pred_binary,
            "pred_mitre": pred_mitre,
            "hidden": h_t,
            "hidden_seq": h_seq,
            "recon_seq": recon_seq,
            "posterior_mu": post_mu,
            "posterior_logvar": post_logvar,
            "prior_mu": dyn_out["prior_mu"],
            "prior_logvar": dyn_out["prior_logvar"],
            "h_last": dyn_out["h_last"],
            "attention_weights": None,
        }

    def compute_elbo_loss(self, out: dict, state_seq: torch.Tensor) -> dict:
        """
        Compute Evidence Lower BOund (ELBO) loss.

        L_elbo = -E[log p(S|z)] + beta * KL(q(z|S) || p(z|z_{prev}))
        """
        beta = 1.0  # Free-ELBO; tune this

        # Reconstruction loss
        recon_loss = F.mse_loss(out["recon_seq"], state_seq)

        # KL divergence: q(z|S) || p(z|z_prev)
        # q = posterior (encoder output)
        # p = prior (GRU dynamics prediction, shifted by 1)
        # Align: posterior[1:] vs prior[:-1]  (KL at each future step)
        post_mu = out["posterior_mu"][:, 1:, :]       # (B, L-1, d_latent)
        post_lv = out["posterior_logvar"][:, 1:, :]
        prior_mu = out["prior_mu"][:, :-1, :]          # (B, L-1, d_latent)
        prior_lv = out["prior_logvar"][:, :-1, :]

        kl = self._kl_gaussian(post_mu, post_lv, prior_mu, prior_lv)

        return {
            "recon": recon_loss,
            "kl": kl,
            "elbo": recon_loss + beta * kl,
        }

    @staticmethod
    def _kl_gaussian(
        mu1: torch.Tensor, lv1: torch.Tensor,
        mu2: torch.Tensor, lv2: torch.Tensor,
    ) -> torch.Tensor:
        """Analytical KL(N(mu1,sigma1^2) || N(mu2,sigma2^2))."""
        lv1 = lv1.clamp(-10, 2)
        lv2 = lv2.clamp(-10, 2)
        kl = 0.5 * (
            lv2 - lv1
            + (lv1.exp() + (mu1 - mu2) ** 2) / (lv2.exp() + 1e-8)
            - 1.0
        )
        return kl.mean()

    def predict_infiltration_prob(self, state_seq: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(state_seq)
        return torch.softmax(out["pred_binary"], dim=-1)[:, 1]

    def sample_next_state(
        self, state_seq: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(state_seq)
        mean = out["pred_state_mean"]
        if deterministic:
            return mean
        logvar = out["pred_state_logvar"].clamp(-10.0, 2.0)
        std = (0.5 * logvar).exp()
        return mean + torch.randn_like(std) * std
