"""
train.py — Pipeline d'entraînement GLoW
=========================================
Entraînement en 2 phases :

  Phase 1 (Pretrain) : Chaque domaine s'entraîne comme autoencoder
    indépendant. Projections gelées.
    Loss = MSE reconstruction par domaine.

  Phase 2 (Align) : On entraîne les projections.
    Losses (terminologie shimmer) :
      - Lcont : contrastive (InfoNCE) sur les w — PAIRED ONLY
      - Ltr   : translation — cross-modale via le décodeur (MSE) — PAIRED ONLY
      - Ldcy  : demi-cycle — same-domain sur les latents — ALL DATA
      - Lcy   : cycle — round-trip z_a → w → z_b → w → z_a — ALL DATA
      - Lfusion: weighted-sum + Tanh (anti-dégénérescence) — ALL DATA
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
from losses.cycle import CycleConsistencyLoss, compute_cycle
from pipeline.diagnostics import compute_diagnostics, DiagnosticsLogger


def _freeze_domains(workspace: GlobalWorkspace, domain_names: list):
    """Gèle entièrement les domaines spécifiés (encoder + decoder + projections).

    Utilisé pour Phase 3 : freeze vision+text, entraîner seulement brain.
    """
    for name in domain_names:
        if name in workspace.domain_modules:
            for p in workspace.domain_modules[name].parameters():
                p.requires_grad = False
            print(f"    Frozen domain: {name}")


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

    Loss = Σ_d MSE(x_d, decode_d(encode_d(x_d)))
    Les deux modalités utilisent MSE (vision sur features, texte sur E5 embeddings).
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
            x_hat = module.reconstruct(x)
            loss += F.mse_loss(x_hat, x)

        loss.backward()
        nn.utils.clip_grad_norm_(workspace.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def _freeze_encoders_decoders(workspace: GlobalWorkspace):
    """Gèle les encodeurs et décodeurs pour Phase 2."""
    for module in workspace.domain_modules.values():
        for p in module.encoder.parameters():
            p.requires_grad = False
        for p in module.decoder.parameters():
            p.requires_grad = False


def _unfreeze_encoders_decoders(workspace: GlobalWorkspace):
    """Dégèle les encodeurs et décodeurs."""
    for module in workspace.domain_modules.values():
        for p in module.encoder.parameters():
            p.requires_grad = True
        for p in module.decoder.parameters():
            p.requires_grad = True


def align_epoch(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    domain_pairs: List[Tuple[str, str]],
    loss_weights: Dict[str, float],
    contrastive_loss_fn: ContrastiveLoss,
    cycle_loss_fn: CycleConsistencyLoss,
) -> Dict[str, float]:
    """Phase 2 : Aligne les domaines dans le workspace.

    Encodeurs/décodeurs sont frozen. Seules les projections s'entraînent.

    Semi-supervised (20% paired) :
      - Lcont et Ltr : seulement sur les samples paired
      - Ldcy, Lcy, Lfusion : sur TOUS les samples

    Fusion : weighted-sum + Tanh (comme Multimodal Dreaming).
    """
    workspace.train()
    beta_dist = torch.distributions.Beta(5.0, 5.0)
    metrics = {
        "contrastive": 0.0, "translation": 0.0,
        "demi_cycle": 0.0, "cycle": 0.0, "fusion": 0.0, "total": 0.0,
        "mean_pos_per_anchor": 0.0,
    }

    for batch in dataloader:
        optimizer.zero_grad()

        # Encode tout et projette dans le workspace
        inputs = {name: batch[name].to(device) for name in batch if name in workspace.domain_modules}
        image_ids = batch.get("image_id", None)
        if image_ids is not None:
            image_ids = image_ids.to(device)
        is_paired = batch.get("is_paired", None)
        if is_paired is not None:
            is_paired = is_paired.to(device)

        workspace_reprs = workspace.encode_domains(inputs)

        # Latents (avant projection) — detach car encodeurs frozen
        latents = {}
        for name in inputs:
            latents[name] = workspace.domain_modules[name].encode(inputs[name]).detach()

        loss_contrastive = torch.tensor(0.0, device=device)
        loss_translation = torch.tensor(0.0, device=device)
        loss_demi_cycle = torch.tensor(0.0, device=device)
        loss_cycle = torch.tensor(0.0, device=device)
        loss_fusion = torch.tensor(0.0, device=device)

        for (name_a, name_b) in domain_pairs:
            if name_a not in workspace_reprs or name_b not in workspace_reprs:
                continue

            w_a = workspace_reprs[name_a]
            w_b = workspace_reprs[name_b]
            mod_a = workspace.domain_modules[name_a]
            mod_b = workspace.domain_modules[name_b]

            # === SUPERVISED LOSSES (paired only) ===
            paired_mask = is_paired if is_paired is not None else None
            has_paired = paired_mask is not None and paired_mask.any()

            if has_paired:
                w_a_p = w_a[paired_mask]
                w_b_p = w_b[paired_mask]
                ids_p = image_ids[paired_mask] if image_ids is not None else None

                # --- 1. Contrastive (Lcont) — multi-positifs, paired only ---
                loss_contrastive += contrastive_loss_fn(w_a_p, w_b_p, image_ids=ids_p)

                # --- 2. Translation (Ltr) — MSE through frozen decoder, paired only ---
                z_b_hat_p = mod_b.from_workspace(w_a_p)
                x_b_hat_p = mod_b.decode(z_b_hat_p)
                loss_translation += F.mse_loss(x_b_hat_p, inputs[name_b][paired_mask])

                z_a_hat_p = mod_a.from_workspace(w_b_p)
                x_a_hat_p = mod_a.decode(z_a_hat_p)
                loss_translation += F.mse_loss(x_a_hat_p, inputs[name_a][paired_mask])

            # === UNSUPERVISED LOSSES (all data) ===

            # --- 3. Demi-cycle (Ldcy) — same-domain reconstruction ---
            # z_a → proj_a → w_a → inv_proj_a → z_a_hat (unsupervised)
            z_a_dcy = mod_a.from_workspace(w_a)
            loss_demi_cycle += F.mse_loss(z_a_dcy, latents[name_a])
            z_b_dcy = mod_b.from_workspace(w_b)
            loss_demi_cycle += F.mse_loss(z_b_dcy, latents[name_b])

            # --- 4. Cycle (Lcy) — round-trip, only needs z_a as target ---
            z_a_cycle = compute_cycle(mod_a, mod_b, latents[name_a])
            loss_cycle += cycle_loss_fn(latents[name_a], z_a_cycle)

            z_b_cycle = compute_cycle(mod_b, mod_a, latents[name_b])
            loss_cycle += cycle_loss_fn(latents[name_b], z_b_cycle)

            # --- 5. Fusion (Lfusion) — weighted-sum + Tanh ---
            alpha = beta_dist.sample().to(device)
            w_fused = torch.tanh(alpha * w_a + (1 - alpha) * w_b)
            z_a_from_fused = mod_a.from_workspace(w_fused)
            z_b_from_fused = mod_b.from_workspace(w_fused)
            loss_fusion += F.mse_loss(z_a_from_fused, latents[name_a])
            loss_fusion += F.mse_loss(z_b_from_fused, latents[name_b])

        # Pondération
        total_loss = (
            loss_weights["contrastive"] * loss_contrastive
            + loss_weights["translation"] * loss_translation
            + loss_weights["demi_cycle"] * loss_demi_cycle
            + loss_weights["cycle"] * loss_cycle
            + loss_weights["fusion"] * loss_fusion
        )

        total_loss.backward()
        nn.utils.clip_grad_norm_(workspace.parameters(), max_norm=1.0)
        optimizer.step()

        metrics["contrastive"] += loss_contrastive.item()
        metrics["translation"] += loss_translation.item()
        metrics["demi_cycle"] += loss_demi_cycle.item()
        metrics["cycle"] += loss_cycle.item()
        metrics["fusion"] += loss_fusion.item()
        metrics["total"] += total_loss.item()

        # Log mean positives per anchor (excluding self-match)
        if has_paired and ids_p is not None:
            pos_mask = ids_p.unsqueeze(1) == ids_p.unsqueeze(0)
            pos_mask.fill_diagonal_(False)
            metrics["mean_pos_per_anchor"] += pos_mask.float().sum(dim=1).mean().item()

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

    # Freeze entire domains if configured (e.g. vision+text for Phase 3)
    frozen_domains = tc.get("frozen_domains", [])
    if frozen_domains:
        _freeze_domains(workspace, frozen_domains)

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
    val_loader: DataLoader = None,
):
    """Phase 2 seule : Workspace alignment."""
    tc = config["training"]
    lc = config["losses"]

    print("=" * 60)
    print("PHASE 2 — Workspace alignment")
    print("=" * 60)

    # Freeze encodeurs/décodeurs, unfreeze projections (standard Phase 2)
    _freeze_encoders_decoders(workspace)
    _unfreeze_projections(workspace)

    # Freeze entire domains if configured (e.g. vision+text for Phase 3)
    frozen_domains = tc.get("frozen_domains", [])
    if frozen_domains:
        _freeze_domains(workspace, frozen_domains)
        # Unfreeze encoder/decoder for non-frozen domains — they need
        # end-to-end training (e.g. brain encoder was just pretrained)
        for name, mod in workspace.domain_modules.items():
            if name not in frozen_domains:
                for p in mod.encoder.parameters():
                    p.requires_grad = True
                for p in mod.decoder.parameters():
                    p.requires_grad = True
                print(f"    Unfrozen encoder+decoder: {name}")

    trainable = sum(p.numel() for p in workspace.parameters() if p.requires_grad)
    total = sum(p.numel() for p in workspace.parameters())
    print(f"  Params trainables: {trainable:,} / {total:,}")

    contrastive_fn = ContrastiveLoss(temperature=lc["contrastive"]["temperature"])
    cycle_fn = CycleConsistencyLoss()

    loss_weights = {
        "contrastive": lc["contrastive"]["weight"],
        "translation": lc["translation"]["weight"],
        "demi_cycle": lc["demi_cycle"]["weight"],
        "cycle": lc["cycle"]["weight"],
        "fusion": lc["fusion"]["weight"],
    }

    domain_pairs = [tuple(p) for p in tc["domain_pairs"]]

    # Optimizer only on trainable params (projections)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, workspace.parameters()),
        lr=tc["align_lr"], weight_decay=tc["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc["align_epochs"]
    )

    # wandb
    use_wandb = config.get("logging", {}).get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=config["logging"]["project"],
            name=os.path.basename(run_dir),
            config=config,
            dir=run_dir,
        )

    # Diagnostics logger
    diag_logger = DiagnosticsLogger(run_dir, use_wandb=use_wandb) if val_loader is not None else None

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
            cycle_loss_fn=cycle_fn,
        )
        scheduler.step()

        # Diagnostics on fixed val subset
        diag_str = ""
        if diag_logger is not None:
            diag = compute_diagnostics(workspace, val_loader, device, contrastive_fn)
            diag_logger.log(epoch + 1, diag)
            # Show brain metrics if brain domain exists, otherwise vision↔text
            if "brain" in workspace.domain_modules:
                diag_str = (
                    f"  val: centr_vb={diag.get('centroid_cos_vision_brain', 0):.3f} "
                    f"centr_tb={diag.get('centroid_cos_text_brain', 0):.3f} "
                    f"R@1_v2b={diag.get('retrieval_vision2brain_R@1', 0):.3f} "
                    f"R@1_b2v={diag.get('retrieval_brain2vision_R@1', 0):.3f} "
                    f"R@1_t2b={diag.get('retrieval_text2brain_R@1', 0):.3f} "
                    f"R@1_b2t={diag.get('retrieval_brain2text_R@1', 0):.3f}"
                )
            else:
                diag_str = (
                    f"  val: cos_gap={diag['cos_gap']:.3f} "
                    f"centr={diag['centroid_cos']:.3f} "
                    f"R@1={diag['retrieval_i2t_R@1']:.3f}"
                )

        print(
            f"  Epoch {epoch+1:3d}/{tc['align_epochs']} — "
            f"total: {metrics['total']:.4f}  "
            f"contr: {metrics['contrastive']:.4f}  "
            f"trans: {metrics['translation']:.4f}  "
            f"dcy: {metrics['demi_cycle']:.4f}  "
            f"cy: {metrics['cycle']:.4f}  "
            f"fus: {metrics['fusion']:.4f}"
            f"{diag_str}"
        )
        # wandb train losses
        if use_wandb:
            wandb.log({
                "train/total": metrics["total"],
                "train/contrastive": metrics["contrastive"],
                "train/translation": metrics["translation"],
                "train/demi_cycle": metrics["demi_cycle"],
                "train/cycle": metrics["cycle"],
                "train/fusion": metrics["fusion"],
                "train/mean_pos_per_anchor": metrics["mean_pos_per_anchor"],
                "train/lr": optimizer.param_groups[0]["lr"],
            }, step=epoch + 1)

        log.append({"epoch": epoch + 1, **metrics})
        _save_log(log, run_dir, "log_align.json")

    # Unfreeze tout pour l'eval/inference
    _unfreeze_encoders_decoders(workspace)

    if use_wandb:
        wandb.finish()

    print()
    print("Training terminé.")


def train(
    workspace: GlobalWorkspace,
    train_loader: DataLoader,
    config: dict,
    device: torch.device,
    run_dir: str = ".",
    val_loader: DataLoader = None,
):
    """Boucle d'entraînement complète : Phase 1 + Phase 2."""
    tc = config["training"]

    if tc["pretrain_epochs"] > 0:
        train_pretrain(workspace, train_loader, config, device, run_dir)
    else:
        print("Phase 1 skippée (pretrain checkpoint chargé)")
        _unfreeze_projections(workspace)

    print()
    train_align(workspace, train_loader, config, device, run_dir, val_loader=val_loader)
