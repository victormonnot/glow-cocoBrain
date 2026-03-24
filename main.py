"""
main.py — Point d'entrée GLoW cocoBrain
=========================================
Global Workspace avec COCO Captions (vision + texte).

Usage:
    python main.py                          # Config par défaut
    python main.py --config path/to/cfg.yaml  # Config custom
"""

import argparse
import json
import os
import shutil
from datetime import datetime

import yaml
import torch
from torch.utils.data import DataLoader

from models.encoders import VisionEncoder, TextEncoder
from models.decoders import VisionDecoder, TextDecoder
from models.domain_module import DomainModule
from models.workspace import GlobalWorkspace
from data.dataset import get_dataset
from data.tokenizer import SimpleTokenizer
from pipeline.train import train
from pipeline.eval import evaluate


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_domain_modules(config: dict) -> dict:
    """Construit les domain modules vision + text."""
    cfg_proj = config["projection"]
    ws_dim = config["workspace_dim"]

    modules = {}

    # --- Vision (features ResNet-50 pré-extraites) ---
    cfg_v = config["vision"]
    modules["vision"] = DomainModule(
        name="vision",
        encoder=VisionEncoder(
            feature_dim=cfg_v["feature_dim"],
            latent_dim=config["latent_dims"]["vision"],
        ),
        decoder=VisionDecoder(
            latent_dim=config["latent_dims"]["vision"],
            feature_dim=cfg_v["feature_dim"],
        ),
        latent_dim=config["latent_dims"]["vision"],
        workspace_dim=ws_dim,
        proj_hidden_dim=cfg_proj["hidden_dim"],
        proj_num_layers=cfg_proj["num_layers"],
        proj_dropout=cfg_proj["dropout"],
    )

    # --- Text (COCO Captions) ---
    cfg_t = config["text"]
    modules["text"] = DomainModule(
        name="text",
        encoder=TextEncoder(
            vocab_size=cfg_t["vocab_size"],
            embed_dim=cfg_t["embed_dim"],
            latent_dim=config["latent_dims"]["text"],
            max_seq_len=cfg_t["max_seq_len"],
            num_heads=cfg_t["num_heads"],
            num_layers=cfg_t["num_layers"],
        ),
        decoder=TextDecoder(
            latent_dim=config["latent_dims"]["text"],
            embed_dim=cfg_t["embed_dim"],
            vocab_size=cfg_t["vocab_size"],
            max_seq_len=cfg_t["max_seq_len"],
            num_heads=cfg_t["num_heads"],
            num_layers=cfg_t["num_layers"],
        ),
        latent_dim=config["latent_dims"]["text"],
        workspace_dim=ws_dim,
        proj_hidden_dim=cfg_proj["hidden_dim"],
        proj_num_layers=cfg_proj["num_layers"],
        proj_dropout=cfg_proj["dropout"],
    )

    return modules


def create_run_dir(name=None):
    """Crée un dossier checkpoints/run_XXX/ pour cette run."""
    os.makedirs("checkpoints", exist_ok=True)
    existing = [d for d in os.listdir("checkpoints") if d.startswith("run_")]
    next_num = max([int(d.split("_")[1]) for d in existing], default=0) + 1
    suffix = f"_{name}" if name else ""
    run_dir = f"checkpoints/run_{next_num:03d}{suffix}"
    os.makedirs(run_dir)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="GLoW cocoBrain Training")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config"
    )
    parser.add_argument(
        "--resume_pretrain", type=str, default=None,
        help="Path to pretrain checkpoint (skip Phase 1)",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Optional run name (e.g. --name baseline)",
    )
    args = parser.parse_args()

    # --- Run directory ---
    run_dir = create_run_dir(args.name)
    print(f"Run directory: {run_dir}")

    # --- Config ---
    config = load_config(args.config)
    shutil.copy2(args.config, os.path.join(run_dir, "config.yaml"))
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset (splits officiels COCO) ---
    train_dataset = get_dataset(config, split="train")
    val_dataset = get_dataset(config, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    # --- Modèle ---
    domain_modules = build_domain_modules(config)
    workspace = GlobalWorkspace(
        domain_modules=domain_modules,
        workspace_dim=config["workspace_dim"],
        fusion=config["workspace"]["fusion"],
        num_heads=config["workspace"]["num_heads"],
        num_layers=config["workspace"]["num_layers"],
    ).to(device)

    # Compte les params
    total_params = sum(p.numel() for p in workspace.parameters())
    trainable_params = sum(p.numel() for p in workspace.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable:    {trainable_params:,}")
    print()

    # --- Train ---
    if args.resume_pretrain:
        print(f"Loading pretrain checkpoint: {args.resume_pretrain}")
        workspace.load_state_dict(torch.load(args.resume_pretrain, map_location=device))
        config["training"]["pretrain_epochs"] = 0  # Skip Phase 1

    train(workspace, train_loader, config, device, run_dir=run_dir)

    # --- Eval ---
    tokenizer = SimpleTokenizer.load(
        f"{config['data']['data_root']}/vocab.json"
    )
    results = evaluate(workspace, val_loader, config, device, tokenizer)

    # --- Save ---
    save_path = os.path.join(run_dir, "checkpoint.pt")
    torch.save({
        "model_state_dict": workspace.state_dict(),
        "config": config,
        "eval_results": results,
    }, save_path)
    print(f"\nCheckpoint sauvegardé: {save_path}")

    # Save eval results as JSON for easy comparison
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "name": args.name,
            "metrics": {k: float(v) if hasattr(v, '__float__') else v
                        for k, v in results.items()},
        }, f, indent=2)
    print(f"Résultats sauvegardés: {results_path}")


if __name__ == "__main__":
    main()
