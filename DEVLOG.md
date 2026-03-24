# DEVLOG — glow-cocoBrain

Journal de bord du projet. Chaque entrée documente ce qu'on a fait, pourquoi, les problèmes rencontrés et les résultats.

---

## 17 mars 2026 — Setup initial

**Ce qu'on a fait :**
- Créé le repo à partir d'une copie de `glow-minimal` (qui avait 3 domaines dummy : vision, text, audio)
- Initial commit avec la structure de base

---

## 21 mars 2026 — Etape 1 complète : COCO vision + texte

### Préparation des données

**Ce qu'on a fait :**
- Supprimé le domaine audio (on garde que vision + texte)
- Créé `scripts/download_coco.sh` pour télécharger COCO 2017 (~19 GB : train2017, val2017, annotations)
- Créé `scripts/extract_features.py` pour pré-extraire les features ResNet-50 (frozen, couche fc remplacée par Identity) → vecteurs de 2048 dims par image, sauvés en `.npy`
- Créé `scripts/build_vocab.py` pour construire un SimpleTokenizer (word-level, top 10k mots, tokens spéciaux: pad=0, unk=1, bos=2, eos=3)
- Créé `data/tokenizer.py` avec la classe SimpleTokenizer

**Pourquoi ResNet-50 frozen :** pragmatique — rentre en 12 GB sur une 4070, rapide à entraîner. On entraîne juste une tête linéaire par-dessus.

**Problème :** `unzip: command not found` sur WSL. Fix: `sudo apt install unzip`.

### Architecture

**Encodeurs :**
- Vision : Linear(2048 → 512) + LayerNorm + GELU (sur features pré-extraites)
- Texte : Embedding(10k, 256, padding_idx=0) + positional + TransformerEncoder (2 layers, 4 heads) + masked mean pooling → 512 dims

**Décodeurs :**
- Vision : MLP (512 → 2048 features). Note: on décode vers des features ResNet, PAS vers des pixels
- Texte : Transformer non-autoregressif → logits (B, seq_len, vocab_size)

**Workspace :** dim=256, fusion attention (8 heads, 2 layers)
**Projections :** MLP 2 couches avec hidden_dim=512, dropout=0.1
**3 losses :** Contrastive (InfoNCE, temp=0.07), Translation, Cycle consistency — toutes poids 1.0

**Training 2 phases :**
1. Pretrain autoencoders (20 epochs, lr=3e-4) — projections gelées
2. Workspace alignment (50 epochs, lr=5e-4) — tout entraîné

### Training run 1 — NaN à l'epoch 2

**Problème :** NaN dans les losses dès l'epoch 2, venant du TextDecoder (gradients instables).

**Fix :**
- Ajout de gradient clipping (max_norm=1.0) en Phase 1
- Ajout de `ignore_index=0` dans la cross-entropy du texte (ignorer le padding)
- Baissé pretrain_lr de 1e-3 à 3e-4

### Training run 2 — Cycle loss qui monte

**Problème :** Après ~2h de training, Phase 1 (pretrain) OK mais en Phase 2 : contrastive descend, translation stagne, cycle MONTE. Les losses sont en conflit.

**Diagnostic :** Poids cycle trop faible (0.5) par rapport au contrastive (1.0) — le contrastive dominait et cassait la cycle consistency.

**Fix :**
- Cycle weight : 0.5 → 1.0
- Batch size : 256 → 512 (plus de négatifs pour le contrastive)
- Ajout de sauvegarde `checkpoint_pretrain.pt` après Phase 1 + flag `--resume_pretrain` pour ne pas refaire la Phase 1

### Training run 3 — Convergence

**Résultat :** Les 3 losses descendent ensemble à partir de l'epoch ~30-35.

**Métriques finales (val set) :**
- R@1 = 4.08%, R@5 = 16.6%, R@10 = 26.2%
- Pour référence : le random donnerait R@1 = 0.004% (1/25014), donc on est ~1000x au-dessus
- centroid_cos = 0.077 (cosine similarity entre centroids vision et texte dans le workspace — bas, marge d'amélioration)

**Observations qualitatives :**
- La traduction image → texte capte les concepts (motorcycle → "black motorcycle parked", toilet → "toilet toilet")
- Mais dégénère souvent en répétitions ("a a a a")
- Le telephone arabe montre que le texte change beaucoup à la première passe puis se stabilise (attracteurs dans le workspace)

### Notebook d'inference

**Ce qu'on a fait :** Créé `notebooks/inference_viz.ipynb` avec :
1. Chargement du modèle et des données
2. Image → Text translation avec affichage des vraies images COCO
3. Retrieval (sous-ensemble de 5000 samples pour éviter OOM — la matrice 25014x25014 crashait le kernel)
4. Exemples de retrieval image → text (top-5 captions) et text → image (top-5 images)
5. Visualisation des losses (contrastive bar chart, translation text, cycle heatmap)
6. "Telephone arabe" — passages multiples dans le workspace avec nearest-neighbor image retrieval pour visualiser les images à chaque passe

**Problème OOM :** sim_matrix (25014x25014) = 2.5 GB crashait le kernel Jupyter. Fix: sous-ensemble de 5000 samples.

### Améliorations identifiées (pas encore faites)

Pour le prochain training run :
- Normaliser les features ResNet avant de les donner au modèle (L2 norm)
- Ajouter `ignore_index=0` aussi dans la translation loss (pas juste la reconstruction)
- Warm-up progressif du poids contrastive (commencer bas, monter)
- Explorer workspace_dim=384 (le bottleneck 256 perd peut-être trop d'info)

---

## Réflexions architecture (pas encore implémenté)

### GW dynamique vs statique
Notre implémentation actuelle est **statique** : un seul forward pass (encode → fuse → decode). La vraie théorie de Baars prévoit un cycle dynamique (compétition → broadcast → réaction → boucle). Deux pistes discutées :

1. **Workspace itératif avec gating** — N étapes de fusion où chaque module a un gate (sigmoid) pour décider s'il réagit au broadcast
2. **Active inference** — chaque module minimise sa surprise (free energy) face au broadcast, et peut "repousser" vers le workspace

Ces deux approches deviennent pertinentes quand on aura 3+ modalités (avec le module fMRI). Avec 2 modalités toujours présentes, le mécanisme dynamique n'apporte pas grand-chose de mesurable.
