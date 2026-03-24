"""
train.py — Pipeline d'entraînement GLoW
=========================================
Entraînement en 2 phases :

  Phase 1 (Pretrain) : Chaque domaine s'entraîne comme autoencoder
    indépendant. Projections gelées.
    Loss = reconstruction par domaine.

  Phase 2 (Align) : On entraîne les projections + le workspace.
    Loss = contrastive + translation + cycle consistency.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

from models.workspace import GlobalWorkspace
from losses.contrastive import ContrastiveLoss
from losses.translation import TranslationLoss, TextTranslationLoss
from losses.cycle import CycleConsistencyLoss, compute_cycle


def _freeze_projections(workspace: GlobalWorkspace):
    """Gèle les projections pour Phase 1."""
    for module in workspace.domain_modules.values():
        for p in module.projection.parameters():
            p.requires_grad = False
        for p in module.inv_projection.parameters():
            p.requires_grad = False


def _unfreeze_projections(workspace: GlobalWorkspace):
    """Dégèle les projections pour Phase 2."""
    for module in workspace.domain_modules.values():
        for p in module.projection.parameters():
            p.requires_grad = True
        for p in module.inv_projection.parameters():
            p.requires_grad = True


def pretrain_epoch(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Phase 1 : Entraîne chaque domaine comme autoencoder.

    Loss = Σ_d reconstruction_loss(x_d, decode_d(encode_d(x_d)))
    """
    workspace.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)

        for name, module in workspace.domain_modules.items():
            if name not in batch:
                continue

            x = batch[name].to(device)

            if name == "text":
                z = module.encode(x)
                logits = module.decode(z)  # (B, seq_len, vocab_size)
                B, S, V = logits.shape
                loss += F.cross_entropy(
                    logits.reshape(B * S, V), x.reshape(B * S), ignore_index=0
                )
            else:
                # Vision : MSE sur les features ResNet-50
                x_hat = module.reconstruct(x)
                loss += F.mse_loss(x_hat, x)

        loss.backward()
        nn.utils.clip_grad_norm_(workspace.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def align_epoch(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    domain_pairs: List[Tuple[str, str]],
    loss_weights: Dict[str, float],
    contrastive_loss_fn: ContrastiveLoss,
    translation_loss_fn: TranslationLoss,
    text_translation_loss_fn: TextTranslationLoss,
    cycle_loss_fn: CycleConsistencyLoss,
) -> Dict[str, float]:
    """Phase 2 : Aligne les domaines dans le workspace.

    Pour chaque paire (A, B) :
      1. Contrastive   : w_a proche de w_b (même sample)
      2. Translation   : x_a → workspace → x̂_b ≈ x_b (et inversement)
      3. Cycle         : z_a → w → z_b → w → ẑ_a ≈ z_a
    """
    workspace.train()
    metrics = {"contrastive": 0.0, "translation": 0.0, "cycle": 0.0, "total": 0.0}

    for batch in dataloader:
        optimizer.zero_grad()

        # Encode tout et projette dans le workspace
        inputs = {name: batch[name].to(device) for name in batch if name in workspace.domain_modules}
        workspace_reprs = workspace.encode_domains(inputs)

        # Aussi besoin des latents (avant projection) pour le cycle
        latents = {}
        for name in inputs:
            latents[name] = workspace.domain_modules[name].encode(inputs[name])

        loss_contrastive = torch.tensor(0.0, device=device)
        loss_translation = torch.tensor(0.0, device=device)
        loss_cycle = torch.tensor(0.0, device=device)

        for (name_a, name_b) in domain_pairs:
            if name_a not in workspace_reprs or name_b not in workspace_reprs:
                continue

            w_a = workspace_reprs[name_a]
            w_b = workspace_reprs[name_b]
            mod_a = workspace.domain_modules[name_a]
            mod_b = workspace.domain_modules[name_b]

            # --- 1. Contrastive ---
            loss_contrastive += contrastive_loss_fn(w_a, w_b)

            # --- 2. Translation (A → B et B → A) ---
            # A → workspace → B
            z_b_hat = mod_b.from_workspace(w_a)
            x_b_hat = mod_b.decode(z_b_hat)

            if name_b == "text":
                loss_translation += text_translation_loss_fn(inputs[name_b], x_b_hat)
            else:
                loss_translation += translation_loss_fn(inputs[name_b], x_b_hat)

            # B → workspace → A
            z_a_hat = mod_a.from_workspace(w_b)
            x_a_hat = mod_a.decode(z_a_hat)

            if name_a == "text":
                loss_translation += text_translation_loss_fn(inputs[name_a], x_a_hat)
            else:
                loss_translation += translation_loss_fn(inputs[name_a], x_a_hat)

            # --- 3. Cycle consistency ---
            z_a_cycle = compute_cycle(mod_a, mod_b, latents[name_a])
            loss_cycle += cycle_loss_fn(latents[name_a].detach(), z_a_cycle)

            z_b_cycle = compute_cycle(mod_b, mod_a, latents[name_b])
            loss_cycle += cycle_loss_fn(latents[name_b].detach(), z_b_cycle)

        # Pondération
        total_loss = (
            loss_weights["contrastive"] * loss_contrastive
            + loss_weights["translation"] * loss_translation
            + loss_weights["cycle"] * loss_cycle
        )

        total_loss.backward()
        nn.utils.clip_grad_norm_(workspace.parameters(), max_norm=1.0)
        optimizer.step()

        metrics["contrastive"] += loss_contrastive.item()
        metrics["translation"] += loss_translation.item()
        metrics["cycle"] += loss_cycle.item()
        metrics["total"] += total_loss.item()

    n = len(dataloader)
    return {k: v / n for k, v in metrics.items()}


def _save_log(log, run_dir, filename):
    """Sauvegarde le training log en JSON (append-safe: écrit à chaque epoch)."""
    path = os.path.join(run_dir, filename)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def train_pretrain(
    workspace: GlobalWorkspace,
    train_loader: DataLoader,
    config: dict,
    device: torch.device,
    run_dir: str = ".",
):
    """Phase 1 seule : Pretrain autoencoders."""
    tc = config["training"]

    print("=" * 60)
    print("PHASE 1 — Pretrain autoencoders")
    print("=" * 60)

    _freeze_projections(workspace)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, workspace.parameters()),
        lr=tc["pretrain_lr"],
        weight_decay=tc["weight_decay"],
    )

    log = []
    for epoch in range(tc["pretrain_epochs"]):
        loss = pretrain_epoch(workspace, train_loader, optimizer, device)
        print(f"  Epoch {epoch+1:3d}/{tc['pretrain_epochs']} — recon loss: {loss:.4f}")
        log.append({"epoch": epoch + 1, "recon_loss": loss})
        _save_log(log, run_dir, "log_pretrain.json")

    pretrain_path = os.path.join(run_dir, "checkpoint_pretrain.pt")
    torch.save(workspace.state_dict(), pretrain_path)
    print(f"  Checkpoint Phase 1 sauvegardé: {pretrain_path}")

    _unfreeze_projections(workspace)


def train_align(
    workspace: GlobalWorkspace,
    train_loader: DataLoader,
    config: dict,
    device: torch.device,
    run_dir: str = ".",
):
    """Phase 2 seule : Workspace alignment."""
    tc = config["training"]
    lc = config["losses"]

    print("=" * 60)
    print("PHASE 2 — Workspace alignment")
    print("=" * 60)

    _unfreeze_projections(workspace)

    contrastive_fn = ContrastiveLoss(temperature=lc["contrastive"]["temperature"])
    translation_fn = TranslationLoss()
    text_translation_fn = TextTranslationLoss()
    cycle_fn = CycleConsistencyLoss()

    loss_weights = {
        "contrastive": lc["contrastive"]["weight"],
        "translation": lc["translation"]["weight"],
        "cycle": lc["cycle"]["weight"],
    }

    domain_pairs = [tuple(p) for p in tc["domain_pairs"]]

    optimizer = torch.optim.AdamW(
        workspace.parameters(), lr=tc["align_lr"], weight_decay=tc["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc["align_epochs"]
    )

    log = []
    for epoch in range(tc["align_epochs"]):
        metrics = align_epoch(
            workspace=workspace,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            domain_pairs=domain_pairs,
            loss_weights=loss_weights,
            contrastive_loss_fn=contrastive_fn,
            translation_loss_fn=translation_fn,
            text_translation_loss_fn=text_translation_fn,
            cycle_loss_fn=cycle_fn,
        )
        scheduler.step()

        print(
            f"  Epoch {epoch+1:3d}/{tc['align_epochs']} — "
            f"total: {metrics['total']:.4f}  "
            f"contr: {metrics['contrastive']:.4f}  "
            f"trans: {metrics['translation']:.4f}  "
            f"cycle: {metrics['cycle']:.4f}"
        )
        log.append({"epoch": epoch + 1, **metrics})
        _save_log(log, run_dir, "log_align.json")

    print()
    print("Training terminé.")


def train(
    workspace: GlobalWorkspace,
    train_loader: DataLoader,
    config: dict,
    device: torch.device,
    run_dir: str = ".",
):
    """Boucle d'entraînement complète : Phase 1 + Phase 2."""
    tc = config["training"]

    if tc["pretrain_epochs"] > 0:
        train_pretrain(workspace, train_loader, config, device, run_dir)
    else:
        print("Phase 1 skippée (pretrain checkpoint chargé)")
        _unfreeze_projections(workspace)

    print()
    train_align(workspace, train_loader, config, device, run_dir)
