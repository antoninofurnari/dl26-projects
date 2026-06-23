"""
Multi-Source Domain Adaptation model (Track 9, objective 2 & 3).

Components
----------
- SharedEncoder E:        feature_dim -> embed_dim, shared across all domains.
- Per-source classifiers: one head per source domain (HMDB, UCF), each over the
                          shared closed-set label space.
- DomainDiscriminator:    predicts which domain an embedding came from, trained
                          adversarially through a GRL -> domain alignment.
- Weighted ensemble:      at inference on the target, each source classifier
                          produces logits; we combine them with per-batch weights
                          derived from how close the target batch is to each
                          source's embedding centroid. This yields BOTH the target
                          prediction and the Source-1 vs Source-2 influence ratio
                          required by objective 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.grl import grad_reverse


class SharedEncoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 256, hidden: int = 512, p: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.net(x)


class ClassifierHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, z):
        return self.fc(z)


class DomainDiscriminator(nn.Module):
    """Predicts domain id from an embedding. Used adversarially via GRL."""

    def __init__(self, embed_dim: int, num_domains: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, z, lambd: float = 1.0):
        return self.net(grad_reverse(z, lambd))


class MultiSourceDA(nn.Module):
    def __init__(self, in_dim, num_classes, source_domains, embed_dim=256, num_domains=3):
        super().__init__()
        self.source_domains = list(source_domains)
        self.encoder = SharedEncoder(in_dim, embed_dim)
        self.classifiers = nn.ModuleDict(
            {dom: ClassifierHead(embed_dim, num_classes) for dom in self.source_domains}
        )
        self.discriminator = DomainDiscriminator(embed_dim, num_domains)
        # Running centroids of each source in embedding space (for ensemble weights).
        for dom in self.source_domains:
            self.register_buffer(f"centroid_{dom}", torch.zeros(embed_dim))
        self.register_buffer("centroid_count", torch.zeros(len(self.source_domains)))
        self.embed_dim = embed_dim

    # ---- encoding ----
    def encode(self, x):
        return self.encoder(x)

    # ---- source supervised path ----
    def classify_source(self, z, domain):
        return self.classifiers[domain](z)

    # ---- adversarial path ----
    def discriminate(self, z, lambd=1.0):
        return self.discriminator(z, lambd)

    # ---- centroid tracking (EMA), used for ensemble weighting ----
    @torch.no_grad()
    def update_centroid(self, domain, z, momentum=0.9):
        c = getattr(self, f"centroid_{domain}")
        batch_mean = z.mean(0)
        new = momentum * c + (1 - momentum) * batch_mean
        setattr(self, f"centroid_{domain}", new)

    # ---- weighted ensemble on target (objective 3 & 4) ----
    def ensemble_predict(self, x, temperature=0.1, return_weights=False):
        """Combine source classifiers with per-batch confidence weights.

        Weighting rationale: a source classifier that produces CONFIDENT (low
        entropy) predictions on the target batch is more trustworthy for that
        batch than one that is uncertain. We therefore weight each source by the
        negative mean prediction entropy of its classifier on the target batch,
        passed through a softmax with `temperature`.

        This replaces the earlier centroid-cosine weighting, which collapsed to
        ~uniform (50/50): after the encoder's ReLU all embeddings live in the same
        positive cone, so source centroids became near-identical (cosine ~0.999)
        and carried no discriminative signal. Confidence is computed per batch and
        stays informative regardless of that geometry.

        Returns combined logits and, optionally, the normalized source weights
        (the Source-1 vs Source-2 influence ratio).
        """
        z = self.encode(x)

        per_source_logits = []
        scores = []  # higher = more confident = should get more weight
        for dom in self.source_domains:
            logits = self.classifiers[dom](z)            # (B, C)
            per_source_logits.append(logits)
            probs = F.softmax(logits, dim=1)             # (B, C)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(1)  # (B,)
            scores.append(-entropy.mean())               # scalar: -mean entropy

        scores = torch.stack(scores)                     # (S,)
        weights = F.softmax(scores / temperature, dim=0) # (S,)

        stacked = torch.stack(per_source_logits, dim=0)  # (S, B, C)
        combined = (weights.view(-1, 1, 1) * stacked).sum(0)  # (B, C)

        if return_weights:
            return combined, weights
        return combined
