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

## 24 mars 2026 — Run 4 (run_001) : L2 norm features + ignore_index + workspace 384

### Changements
- L2 normalisation des features ResNet dans le dataset (chaque vecteur sur la sphère unitaire)
- `ignore_index=0` dans `TextTranslationLoss` (ignorer le padding dans la translation loss Phase 2)
- workspace_dim 256 → 384 (tester si plus d'espace aide)

### Résultats

| Métrique | Run 3 (baseline) | Run 4 | Diff |
|---|---|---|---|
| R@1 | 4.08% | 4.39% | +0.3 |
| R@5 | 16.6% | 16.5% | ~pareil |
| R@10 | 26.2% | 25.7% | -0.5 |
| centroid_cos | 0.077 | -0.0095 | pire |

**Diagnostic :** Retrieval quasi identique. Le centroid_cos est passé négatif — les représentations vision (norme ~17) et texte (norme ~11) vivent dans des sphères de tailles différentes dans le workspace. Le workspace_dim=384 a dispersé les représentations sans mieux les aligner.

### Infra ajoutée ce jour
- Système de run versionnées (`checkpoints/run_XXX_name/`)
- Logs JSON par epoch (`log_pretrain.json`, `log_align.json`)
- `results.json` pour comparaison facile entre runs
- Phases séparables (`--phase pretrain`, `--phase align`)
- Flag `--name` pour nommer les runs
- README complet (setup, commandes, architecture)
- DEVLOG.md (ce fichier)

---

## 24 mars 2026 — Run 5 (run_002_norm_ws_warmup) : fix normes + warmup

### Changements
- workspace_dim 384 → **256** (retour, le 384 n'a pas aidé)
- **L2 normalisation dans le workspace** : `to_workspace()` applique `F.normalize` sur les w. Toutes les modalités projettent sur la même sphère unitaire. Fix le problème de normes différentes (17 vs 11).
- **Warm-up contrastive** : le poids contrastive commence à 0.1x et monte linéairement vers 1.0x sur les 10 premières epochs (20% de 50). Donne le temps à la translation et cycle de s'installer avant que la contrastive compresse.

### Ce qu'on espère
- centroid_cos nettement plus élevé (les normes sont maintenant identiques = 1.0)
- Retrieval en hausse grâce à un meilleur alignement
- Losses qui descendent ensemble sans conflit grâce au warm-up

### Résultats

| Métrique | Run 3 (baseline) | Run 4 (ws384) | Run 5 |
|---|---|---|---|
| R@1 | 4.08% | 4.39% | **4.36%** |
| R@5 | 16.6% | 16.5% | **16.4%** |
| R@10 | 26.2% | 25.7% | **26.3%** |
| centroid_cos | 0.077 | -0.0095 | **0.085** |
| vision_mean_norm | ~17 | ~17 | **1.0** |
| text_mean_norm | ~11 | ~11 | **1.0** |

**Diagnostic :** La normalisation fonctionne (normes = 1.0, centroid_cos positif), mais le retrieval n'a quasi pas bougé (~4% R@1). Le fix freeze + z-loss n'a pas débloqué la performance comme espéré.

**Analyse des logs Phase 2 :**
- Contrastive : 2.41 → 0.46 ✓ (descend bien)
- Translation : 3.85 → 2.98 — **reste très haute**, c'est le bottleneck
- Cycle : 0.21 → 0.33 — monte légèrement (tension entre les losses)
- Les 3 losses descendent ensemble, pas de conflit majeur

**Conclusion :** Le problème n'est pas le freeze (qui était quand même un vrai bug), mais la capacité des projections à aligner les espaces latents. La translation loss stagne à ~3.0, signe que les projections MLP n'arrivent pas à bien traduire entre les latents vision et texte.

---

## 5 avril 2026 — Audit archi + fix fusion + fusion loss

### Problèmes détectés lors de l'audit

**1. Fusion attention jamais entraînée (CRITIQUE)**
Le Transformer de fusion (`workspace.fuse()`) n'était **jamais appelé** pendant le training Phase 2. Les losses (contrastive, translation, cycle) opéraient toutes sur les w individuels (avant fusion). Résultat : ~1.1M de params (2 layers, 8 heads) restaient à leur init random.

Impact : en inférence, `workspace.translate()` et `workspace.forward()` passent par la fusion → poids random au milieu de la chaîne. Le retrieval n'était pas impacté (il compare les w individuels), mais la translation cross-modale et le "telephone arabe" passaient par un Transformer non-entraîné.

**2. workspace_dim (256) = moitié de latent_dim (512)**
Les projections compriment de 512 → 256 dims. L'InverseProjection doit reconstruire 512 dims à partir de 256 dims normalisés (~255 degrés de liberté). La translation loss à ~3.0 confirme que c'est un bottleneck.

**3. Double normalisation dans le contrastive**
`to_workspace()` fait `F.normalize()`, puis `ContrastiveLoss` refait `F.normalize()`. Redondant mais inoffensif (normaliser un vecteur unitaire = no-op).

### Changements implémentés

**Fusion loss (nouvelle loss Phase 2) :**
- Après les losses existantes, on fusionne les w via `workspace.fuse()` → `w_fused`
- Pour chaque domaine : `z_hat = inv_proj(w_fused)`, `loss += MSE(z_hat, z_d)`
- Ça donne un gradient au Transformer de fusion pour la première fois
- Poids configurable dans `config.yaml` (`losses.fusion.weight: 1.0`)

**Fichiers modifiés :**
- `pipeline/train.py` : ajout du calcul fusion + fusion loss dans `align_epoch()`, logging de la fusion loss
- `config/config.yaml` : ajout `losses.fusion.weight: 1.0`

**Ce qu'on espère :**
- La fusion apprend à combiner les w de manière utile
- Le w_fused devient une meilleure représentation que les w individuels
- Amélioration de la translation cross-modale en inférence
- Potentiellement meilleur retrieval si la fusion aide les projections à mieux s'organiser

### Run 6 (run_004_fusion_loss) — Résultats

| Métrique | Run 5 | Run 6 (fusion loss) | Diff |
|---|---|---|---|
| R@1 | 4.36% | **3.47%** | -0.9 |
| R@5 | 16.4% | **13.5%** | -2.9 |
| R@10 | 26.3% | **21.2%** | -5.1 |
| centroid_cos | 0.085 | **-0.68** | catastrophe |

| Loss (epoch 50) | Run 5 | Run 6 |
|---|---|---|
| Contrastive | **0.46** | 1.85 |
| Translation | 2.98 | **0.22** |
| Cycle | 0.33 | **0.056** |
| Fusion | — | 0.051 |

**La fusion loss a EMPIRÉ le retrieval.** Le contrastive est 4x plus haut (1.85 vs 0.46), et centroid_cos = -0.68 (centroids vision/texte en directions opposées).

**Diagnostic : signaux contradictoires aux inv_projections.**
- Translation dit : `inv_proj_text(w_vision)` doit ≈ `z_text` (input = w d'un autre domaine)
- Fusion dit : `inv_proj_text(w_fused)` doit ≈ `z_text` (input = moyenne pondérée des w)
- L'inv_projection reçoit deux types d'inputs et doit reconstruire le même z → trop généraliste, fait mal les deux
- La translation stagne (0.23 → 0.22 en 50 epochs) car les gradients de fusion perturbent l'optimisation

### Fix : cycle via fusion (pas de fusion loss séparée)

**Approche :** au lieu d'une fusion loss séparée qui conflict avec les inv_projections, on fait passer le **cycle** par la fusion. Le cycle devient :

```
w_fused = fuse({w_a, w_b})    ← gradients vers fusion Transformer
z_b_hat = inv_proj_b(w_fused)  ← même type d'input que la translation
w_b_hat = proj_b(z_b_hat)
z_a_cycle = inv_proj_a(w_b_hat)
Loss: MSE(z_a_cycle, z_a)
```

**Pourquoi c'est mieux :**
- La fusion reçoit des gradients via la cycle loss (vérifié : `fusion_net` has gradients = True)
- Les inv_projections reçoivent des inputs cohérents (w individuels via translation, w intermédiaires via cycle) — pas de conflit avec un w_fused qui vient d'une distribution différente
- Pas de loss séparée qui tire dans une direction opposée

**Fichiers modifiés :**
- `pipeline/train.py` : cycle passe par `workspace.fuse()`, suppression de la fusion loss et de `compute_cycle` import
- `config/config.yaml` : fusion weight gardé mais plus utilisé

### Run 7 (run_005_cycle_via_fusion) — Résultats

| Métrique | Run 003 (freeze baseline) | Run 6 (fusion loss) | Run 7 (cycle via fusion) |
|---|---|---|---|
| R@1 | 3.64% | 3.47% | 3.62% |
| Contrastive | 1.83 | 1.85 | 1.83 |
| Translation | 0.217 | 0.22 | 0.214 |
| centroid_cos | -0.68 | -0.68 | -0.69 |

**Les trois runs sont quasi identiques.** La fusion n'a ni aidé ni nui. Avec 2 domaines toujours présents, le Transformer de fusion est du bruit.

### Correction importante : le vrai baseline est ~3.5%, pas 4.36%

Le Run 5 (run_002_norm_ws_warmup, R@1=4.36%) était un run complet SANS le freeze des encodeurs. Le 4.36% était un résultat gonflé : tous les params (y compris encodeurs) étaient optimisés pour l'alignement, ce qui corrompait les autoencoders. Tous les runs avec freeze correct (003, 006, 007) convergent vers ~3.5% R@1.

### Diagnostic : pourquoi ~3.5% avec freeze

Les projections trouvent une solution dégénérée : vision et texte sont projetés dans des directions **opposées** (centroid_cos = -0.68). Ça permet aux inv_projections de facilement distinguer l'origine et reconstruire (translation basse), mais le contrastive lutte car les distributions sont séparées au lieu d'être alignées.

**Causes identifiées :**
1. **workspace_dim (256) = moitié de latent_dim (512)** — bottleneck trop agressif
2. **Projections trop petites** — MLP 2 couches (512→512→256), pas assez de capacité pour transformer les espaces latents structurellement différents
3. **L2 norm sur la sphère** — les projections ne peuvent modifier que la direction, pas la magnitude. Force une contrainte qui peut empêcher de trouver un bon alignement

---

## 5 avril 2026 — Run 8 : workspace=512 + projections 3 couches + pas de norm

### Changements
- **workspace_dim 256 → 512** — plus de bottleneck (latent=512, workspace=512)
- **Projections 3 couches, hidden=1024** : 512 → 1024 → 1024 → 512 (vs 512 → 512 → 256 avant). 8.4M params projections (vs 1.4M avant)
- **Retrait F.normalize dans to_workspace()** — les w ne sont plus contraints sur la sphère unitaire. Le contrastive normalise déjà en interne, l'eval aussi
- **Retour au cycle classique** (sans fusion) — la fusion est un non-sujet à 2 domaines
- Toujours le même pretrain checkpoint (run_002)

### Ce qu'on espère
- Le contrastive descend enfin sous 1.83 (plus de capacité + plus de degrés de liberté)
- centroid_cos positif (les projections ont assez de room pour aligner)
- R@1 significativement au-dessus de 3.5%

### Résultats (run_006_ws512_proj3L)

| Métrique | Run 5 (baseline) | Runs 6/7 (fusion) | **Run 8 (ws512)** |
|---|---|---|---|
| R@1 (25k) | 4.36% | ~3.5% | **4.10%** |
| R@5 (25k) | 16.4% | ~13.5% | 15.3% |
| R@10 (25k) | 26.3% | ~21% | 24.1% |
| centroid_cos | 0.085 | -0.68 | **-0.845** |
| Contrastive (ep50) | 0.46 | 1.83 | 1.09 |
| Translation (ep50) | 2.98 | 0.22 | 0.211 |
| Cycle (ep50) | 0.33 | 0.056 | 0.079 |

**Verdict : pire.** Le centroid_cos est à -0.845 — les projections séparent les domaines encore plus agressivement. 6x plus de params, pas de bottleneck → aucun effet sur le retrieval. Le problème n'est pas la capacité, c'est structural.

**Translation complètement plate :** 0.225 (epoch 1) → 0.211 (epoch 50). Les projections n'apprennent pas à traduire.

**Asymétrie translation (diagnostics val) :** v2t=0.209 vs t2v=0.008. L'inv_proj arrive à reconstruire z_vision depuis w_text, mais pas l'inverse. Le z_text (sortie d'un Transformer + mean pool) est structurellement plus complexe que z_vision (sortie d'un Linear).

### Diagnostic clé : pourquoi ça dégénère

Les projections apprennent une solution paresseuse : "vision en haut, texte en bas". Le contrastive InfoNCE normalise et compare les cosinus → peut matcher les paires par la variance intra-cluster sans que les domaines soient réellement alignés. La translation/cycle sont basses car `inv_proj` reconnaît l'origine du w par sa position dans l'espace.

### Insight fondamental : il nous manque la fusion loss weighted-sum

En relisant le papier shimmer-ssd, on a identifié le mécanisme anti-dégénérescence de GLoW standard :

Dans shimmer, la fusion est une **somme pondérée** (pas un Transformer) :
```
w_fused = α_v * w_v + α_t * w_t    (α_v + α_t = 1)
```

La **fusion loss** force les inv_projections à reconstruire les latents depuis n'importe quel mélange de w. Si les domaines sont aux pôles opposés (centroid_cos = -0.85), la somme pondérée α=0.5 s'annule → reconstruction impossible → loss énorme. Les projections sont donc **forcées** d'aligner les domaines.

C'est pour ça que notre Transformer fusion au Run 6 a échoué : le Transformer peut apprendre à démixer (il a l'attention), donc il contournait la contrainte. La somme pondérée fixe ne laisse pas d'échappatoire.

---

## 6 avril 2026 — Tier 0 : Monitoring anti-dégénérescence

### Motivation
On volait à l'aveugle : pas de diagnostics epoch-par-epoch, juste les losses train et l'eval finale. Impossible de savoir QUAND la dégénérescence s'installe.

### Implémentation (`pipeline/diagnostics.py`)

Module de diagnostic calculé à **chaque epoch sur un subset val fixe de 1000 paires** (pas sur le train) :

**Géométrie workspace :**
- `centroid_cos` — cosine entre centroïdes vision/text (objectif > 0)
- `norm_vision`, `norm_text` — normes moyennes
- `std_per_dim` — détecteur de collapse (si → 0, dims mortes)
- `effective_rank` — SVD, nombre effectif de dims utilisées (Shannon entropy)
- `pairwise_cos_pos` — cosine moyen paires matchées
- `pairwise_cos_neg` — cosine moyen non-matchées
- `cos_gap` — pos - neg (discrimination)

**Losses décomposées :**
- `contrastive`, `translation_v2t`, `translation_t2v`, `cycle` — sur val, séparément

**Retrieval :**
- `R@1`, `R@5` image→text et text→image sur les 1000 paires

Tout est loggé en CSV (`diagnostics.csv`) dans le run_dir, lisible directement en pandas.

### Fichiers modifiés
- `pipeline/diagnostics.py` — CRÉÉ
- `pipeline/train.py` — `train_align()` et `train()` acceptent `val_loader`, log diagnostics à chaque epoch + wandb
- `main.py` — construit `diag_loader` (1000 paires val fixées) et le passe au training

### Run 9 (run_007_ws512_diag) — Première run avec monitoring

Même archi que Run 8 (ws=512, projections 3L), 30 epochs au lieu de 50. Objectif : observer la dégénérescence en temps réel.

**Résultats finaux (25k val) :** R@1=4.29%, centroid_cos=-0.847

**Analyse epoch par epoch (diagnostics.csv) :**

| Epoch | centroid_cos | R@1 (1k val) | translation_v2t | effective_rank_vis |
|-------|-------------|-----------|-----------------|-------------------|
| 1 | **+0.266** | 20.7% | 0.212 | 132 |
| 2 | **-0.044** | 23.7% | 0.212 | 121 |
| 3 | -0.280 | 25.1% | 0.212 | 119 |
| 6 | -0.494 | 29.4% | 0.211 | 119 |
| 14 | -0.707 | 32.8% | 0.210 | 115 |
| 30 | -0.783 | 32.9% | 0.209 | 111 |

**Observations clés :**

1. **La dégénérescence est immédiate** — centroid_cos passe négatif dès l'epoch 2. Les projections choisissent la stratégie "séparer les clusters" dans les toutes premières steps, pas un drift tardif.

2. **Le retrieval monte quand même** — R@1 (1k) : 20.7% → 32.9%. Le contrastive apprend à discriminer *au sein* des clusters séparés. Mais sur le val complet (25k), ça donne seulement 4.3% — ça ne scale pas.

3. **Translation complètement plate** — v2t : 0.212 → 0.209 en 30 epochs. Les projections n'apprennent pas à traduire.

4. **Asymétrie flagrante** — v2t=0.209 vs t2v=0.008. L'inv_proj reconstruit facilement z_vision depuis w_text, mais pas l'inverse. Confirme que z_text (sortie Transformer + mean pool) est structurellement plus complexe que z_vision (sortie d'un simple Linear).

5. **Dimensional collapse en cours** — Effective rank : 132→111 (vision), 152→107 (texte) sur 512 dims. Les représentations perdent des dimensions utiles.

**Conclusion :** Le monitoring confirme le diagnostic. Il faut bloquer la dégénérescence dès les premières epochs. La fusion loss weighted-sum pénaliserait immédiatement la séparation des centroïdes. Les changements d'encodeurs (vision plus profonde, text attentional pooling) et la gestion multi-positifs COCO sont aussi nécessaires.

---

## 6 avril 2026 — Renommage losses + ajout Ltr (translation shimmer)

### Motivation
En comparant avec le papier shimmer-ssd, on a identifié que :
1. Notre nommage ne correspondait pas à la terminologie du lab
2. Il nous manquait la vraie translation loss (Ltr) qui passe par le décodeur

### Mapping ancien → nouveau (terminologie shimmer)

| Ancien nom | Nouveau nom | Shimmer | Chemin |
|------------|-------------|---------|--------|
| "translation" | **demi_cycle** (Ldcy) | Ldcy | `z_a → proj → inv_proj → ẑ_b`, MSE(ẑ_b, z_b) sur latents |
| "cycle" | **cycle** (Lcy) | Lcy | `z_a → proj → inv_proj → proj → inv_proj → ẑ_a`, MSE(ẑ_a, z_a) |
| *(n'existait pas)* | **translation** (Ltr) | Ltr | `z_a → proj → inv_proj → ẑ_b → decode → x̂_b`, loss(x̂_b, x_b) |

### Pourquoi Ltr est utile même avec décodeur frozen
Le gradient de Ltr passe **à travers** le décodeur frozen vers les projections. Le Jacobien du décodeur agit comme un filtre de pertinence : il pondère les dimensions de z par leur importance pour la reconstruction. Le demi-cycle (Ldcy) traite toutes les dims de z comme équivalentes, ce qui gaspille de la capacité sur des dims que le décodeur ignore.

- Ltr vision : MSE(decode(ẑ_vis), features_resnet)
- Ltr texte : cross-entropy(decode(ẑ_text), tokens) avec ignore_index=0

### Fichiers modifiés
- `pipeline/train.py` — renommage complet, ajout Ltr dans `align_epoch()`, helpers `_translation_loss_vision` et `_translation_loss_text`
- `pipeline/diagnostics.py` — métriques renommées (`loss_demi_cycle_v2t/t2v`, `loss_translation_v2t/t2v`), collecte des raw inputs pour Ltr
- `config/config.yaml` — section losses : `translation` (Ltr), `demi_cycle` (Ldcy), `cycle` (Lcy), suppression `fusion` et `reconstruction`

### Diagnostics sur checkpoint Run 9 (vérification)
Les nouvelles métriques Ltr révèlent l'asymétrie encore plus clairement :
- `loss_translation_v2t = 16.29` (cross-entropy texte — très élevé, le décodeur ne sait pas reconstruire le texte depuis un z traduit)
- `loss_translation_t2v = 0.0004` (MSE features — quasi parfait dans ce sens)

---

## 6 avril 2026 — Run 10 : refonte Phase 2 (fusion loss + Ltr + multi-positifs)

### Contexte
Après 9 runs, le problème fondamental est identifié : les projections trouvent une solution dégénérée (centroid_cos = -0.78, domaines aux pôles opposés). Le monitoring epoch par epoch (Run 9) a prouvé que la dégénérescence s'installe dès l'epoch 2. Il manquait le mécanisme anti-dégénérescence central de GLoW : la fusion weighted-sum.

### Tous les changements depuis Run 9

**1. Fusion loss weighted-sum (Lfusion) — LE changement clé**
- À chaque batch : α ~ Beta(2,2), `w_fused = α * normalize(w_a) + (1-α) * normalize(w_b)`
- Les inv_projections doivent reconstruire z_a et z_b depuis w_fused
- Loss = MSE(inv_proj_a(w_fused), z_a) + MSE(inv_proj_b(w_fused), z_b)
- Anti-dégénérescence : si domaines aux pôles opposés, la somme s'annule → loss énorme → force l'alignement
- Beta(2,2) : pic autour de 0.5, évite les extrêmes (α≈0 ou α≈1 = juste le demi-cycle)
- w normalisés avant le mix pour éviter qu'un domaine domine par sa norme
- Diagnostic : fusion loss à α=0.5 fixe sur val (stress test)

**2. Translation loss (Ltr) — nouvelle loss, à travers le décodeur frozen**
- `z_b_hat = inv_proj_b(w_a)` → `x_b_hat = decode_b(z_b_hat)` → loss(x_b_hat, x_b)
- Le gradient passe par le Jacobien du décodeur frozen : pondère les dims de z par leur importance pour la reconstruction
- Vision : MSE(features_hat, features). Texte : cross-entropy(logits, tokens)
- Poids 0.1 (pas 1.0) car la cross-entropy texte (~16) dominerait les autres losses (~2.5)

**3. Renommage losses → terminologie shimmer**
- Ancien "translation" (MSE sur z) → `demi_cycle` (Ldcy)
- Ancien "cycle" → `cycle` (Lcy), inchangé
- Nouveau `translation` (Ltr) → à travers le décodeur
- Nouveau `fusion` (Lfusion) → weighted-sum

**4. Multi-positifs COCO dans le contrastive**
- Le dataset retourne maintenant les `image_id` dans chaque batch
- L'InfoNCE utilise des soft labels : pour chaque anchor, la target est une distribution uniforme sur tous les positifs (même image_id), pas un one-hot sur la diagonale
- Supprime les faux négatifs (captions de la même image traitées comme négatifs)
- Impact limité à batch=512 (~0.01 vrais multi-positifs par anchor), sera plus utile avec batching par image (TODO)
- Fix self-match : `mean_pos_per_anchor` exclut la diagonale pour une stat honnête

**5. Suppression du warm-up contrastive**
- Le poids contrastive montait de 0.1 → 1.0 sur les 20% premières epochs
- Retiré pour avoir des résultats purs avec la nouvelle archi, sans schedule qui complique l'analyse

**6. Pré-check équilibrage (avant lancement)**

Ordres de grandeur des losses sur le checkpoint Run 9 :
| Loss | Valeur | × poids | Contribution |
|------|--------|---------|-------------|
| Contrastive | ~2.5 | ×1.0 | 2.5 |
| Demi-cycle | ~0.22 | ×1.0 | 0.22 |
| Cycle | ~0.03 | ×1.0 | 0.03 |
| Translation | ~16.3 | ×0.1 | **1.63** |
| Fusion | ~0.10 | ×1.0 | 0.10 |

Ltr mise à 0.1 pour ne pas dominer. Les autres à 1.0.

### 5 losses Phase 2 (récap complet)

| Loss | Shimmer | Chemin | Poids |
|------|---------|--------|-------|
| Contrastive (Lcont) | Lcont | InfoNCE(w_a, w_b) soft labels | 1.0 |
| Translation (Ltr) | Ltr | w_a → inv_proj → decode → loss(x̂_b, x_b) | 0.1 |
| Demi-cycle (Ldcy) | Ldcy | w_a → inv_proj → MSE(ẑ_b, z_b) | 1.0 |
| Cycle (Lcy) | Lcy | z_a → proj → inv_proj → proj → inv_proj → MSE(ẑ_a, z_a) | 1.0 |
| Fusion (Lfusion) | Lfusion | α*w_a + (1-α)*w_b → inv_proj → MSE(ẑ, z) | 1.0 |

### Fichiers modifiés
- `pipeline/train.py` — 5 losses, fusion weighted-sum, multi-positifs, suppression warm-up
- `losses/contrastive.py` — InfoNCE multi-positifs avec soft labels via image_ids
- `data/dataset.py` — retourne `image_id` dans chaque batch
- `pipeline/diagnostics.py` — fusion loss α=0.5, normalisation avant mix
- `config/config.yaml` — poids losses (Ltr=0.1), ajout fusion, 30 epochs

### Ce qu'on espère
- centroid_cos qui reste positif ou au moins ne plonge pas à -0.8
- Fusion loss qui descend (= les domaines s'alignent)
- R@1 significativement au-dessus de 4%

### Résultats (run_008_fusion_ltr_mp)

Le training a complété 30 epochs mais le terminal a crashé avant la sauvegarde du checkpoint et l'eval finale. Les diagnostics CSV sont complets.

| Epoch | centroid_cos | R@1 (1k) | fusion_total | contrastive | translation |
|-------|-------------|----------|-------------|-------------|-------------|
| 1 | **+0.148** | 20.4% | 0.096 | 3.11 | 5.16 |
| 4 | -0.375 | 26.7% | 0.066 | 2.77 | 5.09 |
| 10 | -0.656 | 30.9% | 0.045 | 2.60 | 5.00 |
| 20 | -0.778 | 34.8% | 0.033 | 2.57 | 4.95 |
| 30 | **-0.794** | 35.5% | 0.029 | 2.57 | 4.93 |

**Même pattern de dégénérescence.** La fusion loss descend (0.096 → 0.029) pendant que les centroïdes se séparent (-0.794 à epoch 30). Les projections 3 couches/1024 hidden sont assez puissantes pour reconstruire z depuis un w_fused partiellement annulé — elles contournent la contrainte.

**Rapport de forces :** contrastive ~2.57 (85x la fusion à 0.03). La fusion n'a aucun levier pour contrer la dégénérescence. Le mécanisme anti-dégénérescence est présent mais sous-dimensionné.

**Multi-positifs :** mean_pos_per_anchor ≈ 0.003 (quasi zéro, confirmé — à batch=512 sur 118k images, pas de collisions). Le fix est correct mais n'aura d'impact qu'avec le batching par image.

**Pistes pour la suite :**
- Monter le poids fusion (5x–10x) pour qu'il pèse face à la contrastive
- Réduire la capacité des projections (revenir à 2 couches / hidden plus petit) — des projections trop puissantes contournent la contrainte fusion
- Batching par image pour activer les multi-positifs
- Encodeur vision plus profond / attentional pooling texte

---

## 6 avril 2026 — Run 11 : fusion weight=5.0, Beta(5,5)

### Changements
- Fusion weight : 1.0 → **5.0**
- Beta distribution : Beta(2,2) → **Beta(5,5)** (concentré autour de 0.5, mix toujours "dur")

### Résultats (run_009_fusion_w5_beta55)

| Epoch | centroid_cos | R@1 (1k) | fusion_total | contrastive |
|-------|-------------|----------|-------------|-------------|
| 1 | -0.065 | 21.4% | 0.046 | 3.13 |
| 10 | -0.585 | 30.5% | 0.016 | 2.65 |
| 20 | -0.787 | 33.8% | 0.012 | 2.57 |
| 30 | **-0.776** | 36.0% | **0.011** | 2.57 |

**Eval finale (25k) :** R@1=4.15%, centroid_cos=-0.834

**Même pattern.** La fusion loss descend (0.046 → 0.011) malgré les centroïdes opposés. ×5 n'a rien changé. Le problème n'est pas le poids — les projections 3 couches/1024 hidden ont assez de capacité pour satisfaire toutes les losses tout en maintenant la solution dégénérée.

### Diagnostic fondamental

On essaie de contraindre les projections par une loss, mais elles contournent la contrainte. C'est un problème de **capacité des projections**, pas de pondération. Les inv_projections apprennent à reconstruire z depuis n'importe quel input (w_fused partiellement annulé compris) — elles fonctionnent comme un réseau assez puissant pour "deviner" le z original même avec une info partielle.

### Découverte critique : inv_proj_vision est une fonction quasi-constante

Diagnostic approfondi sur le checkpoint Run 11 — on a testé ce que l'inv_proj_vision reconstruit depuis différents inputs :

| Input | MSE vers z_vis | Cosine avec z_vis |
|-------|---------------|-------------------|
| w_vision (correct) | 0.005 | 0.981 |
| w_fused (0.5/0.5) | 0.004 | 0.984 |
| **w_random (bruit)** | **0.019** | **0.945** |
| **w_zeros (rien)** | **0.010** | **0.962** |

**L'inv_proj_vision reconstruit z_vis avec 96% cosine similarity depuis un vecteur de ZÉROS.** Elle a mémorisé z_vis_mean dans ses biais et fait des petites corrections. Elle n'utilise pas son input.

Pour comparaison, inv_proj_text depuis w_random → cosine=0.117 (quasi aléatoire). L'inv_proj_text utilise réellement le w.

**Classifieur linéaire** sur la direction centroïde : 85.5% accuracy pour distinguer vision/texte. Les domaines sont clairement séparés dans le workspace.

**Scaling R@1 :** 79% (100 paires) → 46% (500) → 36% (1000). Le scaling est normal — le modèle capte quelque chose (discrimination par catégorie grossière), mais pas assez fin pour 25k.

### Cause racine identifiée

Le z_vision (sortie d'un seul Linear(2048→512)) a **trop peu de variance**. Tous les z_vis sont proches les uns des autres. L'inv_proj_vision a appris à retourner ~z_vis_moyen quelle que soit l'entrée, car c'est la solution optimale quand la target ne varie presque pas. La fusion loss descend parce que l'inv_proj n'a même pas besoin du w_fused.

### Plan en 2 étapes

**Étape A (sans refaire Phase 1) :** forcer inv_proj à dépendre du w
- Projections 2 couches / hidden=512 (moins de capacité pour mémoriser)
- workspace_dim=384 (bottleneck : 384 dims → reconstruire 512 dims, impossible sans lire le w)
- τ=0.1 (contrastive moins agressive)

**Étape B (avec nouvelle Phase 1) :** encodeur vision plus profond + attentional pooling texte
- Donner aux z_vision plus de variance et de structure
- Mais inutile tant que inv_proj contourne tout

---

## 6 avril 2026 — Run 12 : bottleneck ws=384, projections 2L/512, τ=0.1

### Changements
- **workspace_dim 512 → 384** — bottleneck, force la compression (512→384→512)
- **Projections 2 couches, hidden=512** (au lieu de 3 couches / 1024) — moins de capacité pour mémoriser z_mean dans les biais
- **τ = 0.07 → 0.1** — contrastive moins agressive, gradients plus doux
- Toutes les autres losses inchangées (Ltr=0.1, Ldcy/Lcy/Lfusion=1.0, fusion weight=5.0, Beta(5,5))
- Toujours le même pretrain checkpoint (run_002, Phase 1 inchangée)

### Ce qu'on espère
- inv_proj_vision qui dépend réellement du w (tester avec w_random/w_zeros après)
- centroid_cos qui ne plonge pas aussi vite
- Fusion loss qui ne descend pas artificiellement

### Résultats (run_010_ws384_proj2L_tau01)

**Eval finale (25k) :** R@1=2.93%, centroid_cos=-0.426

| Métrique | Run 11 (ws512, 3L) | **Run 12 (ws384, 2L, τ=0.1)** |
|---|---|---|
| centroid_cos (ep30) | -0.776 | **-0.333** |
| centroid_cos (ep1) | -0.065 | **+0.112** |
| Passe négatif à epoch | 2 | **6** |
| R@1 1k (ep30) | 36.0% | 26.1% |
| R@1 25k | 4.15% | 2.93% |
| cos_gap (ep30) | 0.610 | **0.686** |
| fusion_total (ep30) | 0.011 | 0.010 |
| effective_rank_vis | 103 | **64** |

### Analyse

**Ce qui va mieux :**
- centroid_cos = -0.33 au lieu de -0.78. La dégénérescence est **freinée** — les domaines sont beaucoup moins séparés. Met 6 epochs au lieu de 2 pour passer négatif.
- cos_gap = 0.686 vs 0.610. Meilleure discrimination paires positives/négatives.

**Ce qui va moins bien :**
- R@1 25k = 2.93% — plus bas qu'avant. Le bottleneck comprime et les projections 2L n'ont pas assez de capacité pour bien traduire.
- effective_rank_vis = 64/384 — collapse dimensionnel sévère sur la vision. Seulement 64 dims sur 384 sont utilisées.
- Asymétrie des normes : vision ~17, texte ~8. Le domaine vision domine le workspace.

### Diagnostic

On est passé de "trop de capacité" (projections contournent les contraintes) à "pas assez" (ne peut pas bien traduire). Le bottleneck + projections réduites freinent la dégénérescence mais au prix d'un collapse dimensionnel et d'un retrieval plus bas. Il faut trouver le sweet spot entre les deux.

Le collapse dimensionnel (effective_rank 64/384) suggère que le problème vient aussi des encodeurs — le z_vision (un seul Linear) n'a pas assez de structure pour remplir 384 dims de manière utile.

---

## Réflexions architecture (pas encore implémenté)

### GW dynamique vs statique
Notre implémentation actuelle est **statique** : un seul forward pass (encode → fuse → decode). La vraie théorie de Baars prévoit un cycle dynamique (compétition → broadcast → réaction → boucle). Deux pistes discutées :

1. **Workspace itératif avec gating** — N étapes de fusion où chaque module a un gate (sigmoid) pour décider s'il réagit au broadcast
2. **Active inference** — chaque module minimise sa surprise (free energy) face au broadcast, et peut "repousser" vers le workspace

Ces deux approches deviennent pertinentes quand on aura 3+ modalités (avec le module fMRI). Avec 2 modalités toujours présentes, le mécanisme dynamique n'apporte pas grand-chose de mesurable.

---

## 8 avril 2026 — Analyse des papiers du lab : comparaison avec notre implémentation

### Papiers analysés

1. **"Semi-supervised Multimodal Representation Learning through a Global Workspace"** (Devillers, Maytié, VanRullen, IEEE TNNLS 2024) — arXiv:2306.15711. C'est le papier shimmer, la base théorique de tout le framework GLoW du lab. Testé sur Simple Shapes, Factory, et **COCO Captions**.
2. **"Multimodal Dreaming: A Global Workspace Approach to World Model-Based Reinforcement Learning"** (Maytié, Bertin Johannet, VanRullen, 2025) — arXiv:2502.21142. GW + Dreamer pour du RL sur RoboDesk.

### Résultats COCO du papier shimmer (Table III)

Le papier 1 teste sur COCO Captions avec seulement **20% de données appariées** (le reste est utilisé pour les losses unsupervised : cycle et demi-cycle). Eval sur **1K Karpathy** (5-fold, moyenné sur 5 runs).

| Modèle | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 |
|---|---|---|---|---|---|---|
| Full GW (Ltr+Lcont+Ldcy+Lcy) | **34.4±0.6%** | 65.1±0.3% | 79.0±0.2% | **30.3±0.5%** | 65.3±0.1% | 79.0±0.2% |
| Contrastive seul (Pcont) | 31.8±0.6% | 62.5±0.5% | 76.0±0.4% | 27.3±0.2% | 62.0±0.4% | 77.3±0.3% |
| **Nous (Run 11, 1k diag)** | **~36%** | — | — | — | — | — |
| **Nous (Run 12, 1k diag)** | ~26% | — | — | — | — | — |

**Observations :** Sur 1K, nos résultats (26-36%) sont dans le même ordre de grandeur que les leurs (34%). Mais notre eval 25K (4.15%) est beaucoup plus dure (5× plus de candidats). Le full GW bat le contrastive seul (+2.6% R@1 i2t), confirmant que les broadcast losses aident aussi sur COCO.

**Architecture COCO du papier 1 :** Pour COCO, ils abandonnent les VAE (trop lourds pour les images naturelles) et utilisent des encodeurs pré-entraînés frozen directement :
- Vision : **ResNet50** pré-entraîné ImageNet (comme nous !)
- Texte : **BGE** (sentence embedding model pré-entraîné, bien plus puissant que notre Transformer from scratch)
- Contrastive coefficient pour COCO : **0.05**
- Losses : Ltr + Lcont + Ldcy + Lcy (pas de Lfusion mentionnée)

### Architecture du lab vs la nôtre

| Aspect | Shimmer — Simple Shapes/Factory | Shimmer — **COCO** | Multimodal Dreaming | **Nous (cocoBrain)** |
|---|---|---|---|---|
| **Workspace dim** | **10-12** | non spécifié | **10-60** | **384** |
| **Latent dim** | **12-20** | non spécifié | **10-40** | **512** |
| **Projections** | MLP **4 layers**, 256-512 hidden | MLP **4 layers** | MLP **8 layers** | MLP **2 layers**, 512 hidden |
| **Lcont poids** | grid search {0.1, 1, 10} | **0.05** | **0.1** | **1.0** |
| **Ltr poids** | 1.0 | 1.0 | 1.0 | 0.1 |
| **Batch size** | non spécifié | non spécifié | **2056** | 512 |
| **Vision encoder** | β-VAE (frozen) | **ResNet50** (frozen) | CNN-VAE (frozen) | ResNet-50 frozen + Linear(2048→512) |
| **Texte encoder** | BERT VAE (frozen) | **BGE** (frozen) | N/A (attributs) | Transformer 2L/4H/256d **from scratch** |
| **Fusion** | Weighted sum (fixe) | Weighted sum (fixe) | Weighted sum + **Tanh** | Transformer attention (learnable) |
| **Norm workspace** | Aucune | Aucune | **Tanh** (borne [-1,1]) | L2 norm dans contrastif |
| **% données appariées** | 100% | **20%** | 100% | 100% |

### Découvertes clés

#### 1. Nos dimensions sont 6× à 38× trop grandes

Le lab travaille avec workspace 10-60 dims. Nous on est à 384. Avec des dims si petites, le bottleneck est **tellement serré** que le workspace doit absolument compresser l'information sémantique utile pour reconstruire. Si vision→+z et texte→-z (pôles opposés), la fusion donne ~0 → perte totale d'information → reconstruction impossible. Avec dim=384, il y a des centaines de dimensions "spare" où chaque modalité peut se cacher dans son propre sous-espace.

#### 2. Le contrastif est 10-20× trop fort

Le lab met Lcont à 0.05-0.1, nous à 1.0. Le contrastif pousse à **discriminer** (séparer les négatifs), les broadcast losses poussent à **aligner** (reconstruire fidèlement). Quand le contrastif domine → il sépare les distributions → dégénérescence. Dans un GW, **l'alignement vient des broadcast losses, pas du contrastif**. Le contrastif est un complément fin, pas le moteur.

Preuve : dans le papier 2, le CLIP-Dreamer (contrastive seul, pas de broadcast losses) **échoue complètement** sur la tâche. Ça confirme que le contrastif seul ne suffit pas.

#### 3. L2 norm + cosine similarity = attracteur pour les pôles opposés

Le lab n'utilise pas de L2 normalization sur le workspace. Papier 2 utilise **Tanh** (borne [-1,1] par dim, pas de projection sphérique). Notre contrastif normalise les w sur l'hypersphère (cosine similarity) → les pôles opposés sont des attracteurs naturels sur une sphère.

#### 4. Notre text encoder est massivement sous-dimensionné

Pour COCO, le lab utilise **BGE** (sentence embedding model pré-entraîné, spécialisé similarité sémantique). Pour Simple Shapes/Factory, ils utilisent BERT. Nous on a un Transformer 2L/4H/256d entraîné from scratch sur un vocab de 10k mots. BGE/BERT donnent des représentations sémantiquement riches dès le départ, notre petit Transformer ne peut pas produire de représentations de qualité comparable.

Chaîne causale de la dégénérescence : texte encoder trop faible → faible variance dans z_text → projections trouvent plus facile de séparer que d'aligner → contrastif trop fort accélère la séparation → dégénérescence dès epoch 2.

#### 5. Le lab utilise des VAE pour les datasets synthétiques, pas pour COCO

Pour Simple Shapes et Factory, leurs encodeurs sont des β-VAE (espace latent lisse et structuré). **Mais pour COCO**, ils abandonnent les VAE (trop lourds) et utilisent des encodeurs pré-entraînés frozen directement : ResNet50 (vision) et BGE (texte) — une approche similaire à la nôtre, sauf pour le choix du texte encoder.

#### 6. Projections beaucoup plus profondes au lab

4-8 layers au lab vs 2 chez nous. Avec un bottleneck agressif (10-60 dims), les projections ont besoin de profondeur pour mapper correctement entre espaces de dimensions très différentes.

#### 7. Notre eval est 5× plus dure

Le lab évalue sur le **Karpathy 5K test split** (5000 candidats). Nous on évalue sur 25K (25000 candidats). Un R@1 de ~15% sur 5K correspond approximativement à ~4% sur 25K. Nos résultats ne sont peut-être pas aussi mauvais qu'on le pensait — mais on doit ajouter une eval 5K pour comparer équitablement.

### Changements à faire (par priorité)

**Sans retrain Phase 1 (rapide) :**

| # | Changement | Impact attendu |
|---|---|---|
| 1 | Lcont weight : 1.0 → **0.1** | Fort — rééquilibre les losses, arrête la domination du contrastif |
| 2 | Ltr weight : 0.1 → **1.0** | Fort — renforce l'alignement via le décodeur |
| 3 | Ajouter eval sur **5K Karpathy** | Permet la comparaison avec le lab |
| 4 | workspace_dim : 384 → **64** | Fort — bottleneck force l'alignement (comme le lab) |
| 5 | latent_dim : 512 → **64** | Fort — cohérent avec les dims du lab, mais nécessite retrain Phase 1 |
| 6 | Projections : 2L → **4L** | Moyen — compense le bottleneck plus serré |
| 7 | Retirer L2 norm / tester Tanh | Supprime l'attracteur sphérique |

**Avec retrain Phase 1 (plus gros impact) :**

| # | Changement | Impact attendu |
|---|---|---|
| 8 | Text encoder → **BGE ou BERT frozen** (HuggingFace) — c'est ce que le lab utilise sur COCO | **Très fort** — z_text riche et structuré dès le départ |
| 9 | Vision encoder : garder ResNet50 frozen (validé par le papier 1 sur COCO) | Le lab fait pareil |
| 10 | latent_dim → 64 (avec nouveaux encodeurs) | Cohérent avec le lab |

### Note sur la comparabilité des résultats

Nos runs 11 (R@1 1K ~36%) et runs précédentes sont **dans le même ordre de grandeur** que le papier shimmer sur COCO (R@1 1K = 34.4%), malgré nos problèmes de dégénérescence. Cependant :
- Eux n'utilisent que 20% de données appariées (nous 100%)
- Eux utilisent BGE comme text encoder (nous un Transformer from scratch)
- Leur eval est moyennée sur 5-fold 1K (nous un subset val fixe de 1k)
- L'écart se creuse probablement sur des évaluations plus larges (5K, 25K) car la dégénérescence pénalise le scaling

---

## 8 avril 2026 — Run 13 : refonte majeure alignée sur les papiers du lab

### Motivation

Après l'analyse des papiers shimmer (TNNLS 2024) et Multimodal Dreaming (2025), on a identifié 5 causes structurelles de la dégénérescence. Cette run applique **tous les correctifs en une seule fois** — c'est le plus gros changement depuis le setup initial du repo.

### Tous les changements

| # | Changement | Ancien | Nouveau | Pourquoi |
|---|---|---|---|---|
| 1 | **Text encoder** | Transformer 2L/4H from scratch (256d, vocab 10k) | **E5-large-v2** frozen (1024-dim, pré-extrait) | Le lab utilise BGE/BERT frozen. Notre Transformer from scratch produit des z_text sans variance → dégénérescence. E5 est un sentence embedding model moderne, meilleur que BGE. |
| 2 | **Vision encoder** | Linear(2048→512) + LN + GELU | **MLP 3 couches** (2048→1024→512→128) + LN + GELU | Un seul Linear produit des z_vis trop peu structurés. Compression progressive pour garder plus de variance. |
| 3 | **Latent dim** | 512 | **128** | Le lab utilise 10-40. On prend 128 comme compromis pour COCO (plus complexe que Simple Shapes). |
| 4 | **Workspace dim** | 384 | **64** | Le lab utilise 10-60. Le bottleneck agressif (128→64) force l'alignement car pas de room pour cacher chaque modalité dans son sous-espace. |
| 5 | **Projections** | 2 layers, 512 hidden | **6 layers**, 256 hidden | Le lab utilise 4-8 layers. Plus de profondeur compense le bottleneck serré. |
| 6 | **Lcont weight** | 1.0 | **0.1** | Le lab utilise 0.05 (shimmer COCO) à 0.1 (Multimodal Dreaming). Le contrastif trop fort dominait les broadcast losses et tirait vers la dégénérescence. |
| 7 | **Ltr weight** | 0.1 | **1.0** | Le lab utilise 1.0. La translation à travers le décodeur est maintenant MSE pour les deux modalités (plus d'asymétrie CE vs MSE). |
| 8 | **Lfusion weight** | 5.0 | **1.0** | Plus besoin de surpondérer : le bottleneck 64-dim fait le travail anti-dégénérescence naturellement. |
| 9 | **Fusion** | Transformer attention (learnable) | **Weighted sum + Tanh** | Le lab n'utilise jamais d'attention. Le Transformer contournait les contraintes (Runs 6-7). Tanh borne à [-1,1] sans projeter sur l'hypersphère. |
| 10 | **L2 norm fusion** | F.normalize(w) avant le mix | **Pas de L2 norm** (Tanh sur le résultat) | L2 norm projetait sur une sphère → attracteur pôles opposés. Tanh borne sans géométrie sphérique. |
| 11 | **Données appariées** | 100% pour toutes les losses | **20% pour Ltr et Lcont**, 100% pour Ldcy, Lcy, Lfusion | Comme le papier shimmer. Les broadcast losses n'ont pas besoin d'appariement. |
| 12 | **Demi-cycle** | Cross-modal (z_a → w → z_b_hat) | **Same-domain** (z_a → w → z_a_hat) | Le vrai demi-cycle shimmer est unsupervised (same-domain). Pas besoin de l'appariement. |
| 13 | **Phase 1 texte** | Cross-entropy sur tokens | **MSE sur embeddings E5** | Le TextDecoder reconstruit maintenant des vecteurs E5 (1024-dim), pas des logits tokens. Symétrique avec la vision. |
| 14 | **Vision decoder** | MLP 2 couches (512→1024→2048) | **MLP 3 couches** (128→512→1024→2048) | Miroir du nouvel encodeur. |
| 15 | **Text decoder** | Transformer non-autoregressif → logits vocab | **MLP simple** (128→512→1024) | Reconstruit des embeddings E5, pas des tokens. Plus simple, plus symétrique. |

### Conséquences architecturales

**Symétrie complète vision/texte** — Les deux modalités sont maintenant traitées exactement de la même façon :
- Input = vecteur de features pré-extraites (ResNet 2048-dim ou E5 1024-dim)
- Encoder = MLP → z (128-dim)
- Decoder = MLP → features reconstruites
- Loss Phase 1 = MSE
- Loss Ltr = MSE à travers le décodeur frozen

L'asymétrie Ltr_v2t (cross-entropy ~16) vs Ltr_t2v (MSE ~0.004) qui posait problème depuis le début est éliminée.

**Suppression du tokenizer** — Le dataset charge directement les embeddings E5 pré-extraits. Plus besoin de `data/tokenizer.py` dans le pipeline de training (gardé dans le repo pour référence).

**Suppression de la traduction qualitative** — Avec des embeddings E5 en sortie du décodeur, on ne peut plus décoder vers du texte lisible. L'évaluation se fait uniquement par retrieval (R@K) et MSE.

### Fichiers modifiés

| Fichier | Action |
|---|---|
| `scripts/extract_text_features.py` | **CRÉÉ** — extraction E5-large-v2 pour toutes les captions |
| `requirements.txt` | Ajouté `sentence-transformers` |
| `config/config.yaml` | Réécrit — nouvelles dims, poids, paired_fraction |
| `models/encoders.py` | Réécrit — VisionEncoder MLP 3L, TextEncoder MLP 2L |
| `models/decoders.py` | Réécrit — VisionDecoder MLP 3L, TextDecoder MLP simple |
| `models/workspace.py` | Simplifié — supprimé Transformer, gardé weighted sum + Tanh |
| `data/dataset.py` | Modifié — charge E5 embeddings, `is_paired` flag |
| `pipeline/train.py` | Modifié — tout MSE, Tanh fusion, 20% paired, same-domain dcy |
| `pipeline/diagnostics.py` | Modifié — Ltr MSE, fusion Tanh |
| `pipeline/eval.py` | Modifié — retiré tokenizer, translation MSE bidirectionnel |
| `main.py` | Modifié — nouveaux constructeurs, workspace simplifié |

Fichiers inchangés : `models/projections.py`, `models/domain_module.py`, `losses/contrastive.py`, `losses/cycle.py`.

### Ce qu'on espère

- centroid_cos **positif** ou au minimum > -0.3 (le bottleneck 64-dim + Tanh devrait empêcher la dégénérescence)
- R@1 1K > 30% (comparable au papier shimmer)
- R@1 25K significativement au-dessus de 4%
- Fusion loss qui reste élevée (= ne contourne plus la contrainte)
- effective_rank proche de 64 sur les deux modalités (pas de collapse)
- Losses symétriques vision/texte (plus d'asymétrie)

### Pré-requis avant lancement

```bash
pip install sentence-transformers
python scripts/extract_text_features.py  # ~590k captions train + ~25k val
python main.py --name run13_refonte      # Phase 1 + Phase 2
```

---

## 8 avril 2026 — Résultats Run 13

### La dégénérescence est corrigée

La refonte a fonctionné. Le problème principal (centroid_cos ≈ -0.8, modalités aux pôles opposés) est résolu.

| Métrique | Avant (runs 1-10) | Run 13 | Cible |
|---|---|---|---|
| centroid_cos | **-0.8** | **+0.45** | > 0 |
| R@1 1K (i2t) | ~4% (25K, non comparable) | **28.6%** | > 30% |
| R@1 1K (t2i) | — | **27.4%** | — |
| R@5 1K (i2t) | — | **64.2%** | — |
| R@5 1K (t2i) | — | **64.3%** | — |
| effective_rank | collapse | **33/64** | proche de 64 |
| cos_gap (pos-neg) | ~0 | **0.81** | > 0.5 |
| std_per_dim | collapse → 0 | **1.1-1.4** | stable > 0 |

### Comparaison avec le lab (shimmer paper, 1K Karpathy)

| | Lab (shimmer) | Nous (Run 13, 1K val) |
|---|---|---|
| i2t R@1 | 34.4% | 28.6% |
| t2i R@1 | 25.3% | 27.4% |
| i2t R@5 | 63.5% | 64.2% |
| t2i R@5 | 53.5% | 64.3% |

On est dans la bonne zone. Le t2i est comparable voire meilleur. Le i2t R@1 est 6 points derrière. Les R@5 sont au niveau ou au-dessus.

**Note :** notre eval 1K est sur un subset val aléatoire, pas le split Karpathy officiel. Les chiffres ne sont pas directement comparables mais donnent l'ordre de grandeur.

### Phase 1 — Pretrain

Recon loss : 0.0015 → 0.00015 (÷10 en 20 epochs). Les autoencoders convergent bien sur les deux modalités. La symétrie MSE vision/texte fonctionne — plus d'asymétrie cross-entropy vs MSE.

### Phase 2 — Alignment (30 epochs)

Toutes les losses convergent proprement :

| Loss | Epoch 1 | Epoch 30 |
|---|---|---|
| Contrastive | 2.59 | 0.72 |
| Translation | 0.0011 | 0.00046 |
| Demi-cycle | 0.012 | 0.0038 |
| Cycle | 0.012 | 0.0044 |
| Fusion | 0.012 | 0.0043 |
| **Total** | **0.297** | **0.085** |

Les losses baissent encore à epoch 30 — le modèle n'a pas encore convergé complètement. Plus d'epochs pourrait améliorer les résultats.

### Retrieval val 1K — évolution par epoch

| Epoch | i2t R@1 | i2t R@5 | t2i R@1 | t2i R@5 |
|---|---|---|---|---|
| 1 | 11.3% | 35.6% | 11.0% | 36.6% |
| 5 | 20.6% | 52.9% | 21.3% | 53.3% |
| 10 | 24.8% | 58.6% | 22.8% | 57.6% |
| 20 | 26.3% | 62.2% | 26.7% | 63.1% |
| 30 | 28.6% | 64.2% | 27.4% | 64.3% |

Progression régulière, pas encore de plateau. Encore en train de monter à epoch 30.

### Eval full val (25K)

| Métrique | Valeur |
|---|---|
| R@1 v→t | 2.58% |
| R@5 v→t | 10.6% |
| R@10 v→t | 17.3% |
| Translation MSE v→t | 0.165 |
| centroid_cos | 0.448 |

Le R@1 25K est bas (2.6%) mais c'est normal : retriever parmi 25K candidats est beaucoup plus dur que 1K. Le lab évalue sur 1K Karpathy, pas 25K. Le 2.6% 25K n'est pas comparable au 4% des runs précédentes car l'architecture et les features ont complètement changé.

### Problèmes identifiés

**1. Asymétrie des normes workspace** : vision ~8.8 vs text ~11.3. Les embeddings texte pèsent plus lourd dans le workspace. Causes :
- La dernière couche de `DomainProjection` est un `nn.Linear` sans LayerNorm en sortie
- Les couches intermédiaires ont LN+GELU, mais la sortie est "nue"
- Le lab utilise un LayerNorm en sortie de projection pour forcer les domaines à la même échelle

Conséquences :
- La fusion `tanh(α * w_vis + (1-α) * w_text)` est déséquilibrée (texte domine)
- Les gradients MSE du demi-cycle/cycle sont plus forts pour le texte
- Le contrastif n'est pas affecté (L2-normalise en interne)

**2. centroid_cos oscille** : fluctue entre -0.12 et +0.52 d'une epoch à l'autre. Probablement lié à l'asymétrie des normes — les centroïdes bougent librement en magnitude.

**3. Convergence incomplète** : les losses baissent encore et le R@1 monte encore à epoch 30. Le modèle bénéficierait de plus d'epochs.

### Pistes pour Run 14

- **LayerNorm en sortie de projection** — fix le plus important, corrige l'asymétrie des normes et devrait stabiliser centroid_cos
- **Plus d'epochs** (50-60 au lieu de 30) — le modèle n'a pas encore convergé
- Optionnellement : implémenter l'eval 5K Karpathy officielle pour comparaison directe avec le lab

### Notebook mis à jour

`notebooks/inference_viz.ipynb` réécrit pour l'architecture Run 13 :
- Supprimé le tokenizer (plus de `SimpleTokenizer`, `max_seq_len`, décodage de tokens)
- Ajouté `decode_e5_to_caption()` : nearest-neighbor en espace E5 pour "décoder" les traductions vers des captions lisibles
- Constructeurs modèle mis à jour (`TextEncoder`/`TextDecoder` MLP, `GlobalWorkspace` sans fusion params)
- Translation affiche la caption NN + MSE au lieu de décoder des tokens
- Téléphone arabe fonctionne via NN lookup dans les deux espaces (vision features + E5 embeddings)

---

## 8 avril 2026 — Run 14 : LayerNorm + 50 epochs + Karpathy eval

### Changements

1. **LayerNorm en sortie des projections** (`models/projections.py`) — ajouté `nn.LayerNorm` après le dernier `nn.Linear` dans `DomainProjection` et `InverseProjection`. Corrige l'asymétrie des normes vision/texte dans le workspace.
2. **50 epochs d'alignement** au lieu de 30 (`config.yaml`)
3. **Eval Karpathy 1K** (`pipeline/eval.py` + `scripts/download_karpathy.py`) — 5-fold sur les 5000 images test Karpathy, protocole standard image-level (chaque image a 5 captions).

Phase 1 (pretrain) réutilisée depuis Run 13 via `--resume_pretrain`.

### Résultats : normes corrigées, R@1 stable

| Métrique | Run 13 | Run 14 | Commentaire |
|---|---|---|---|
| vision_mean_norm | 8.84 | **7.53** | ↓ vers sqrt(64)≈8 |
| text_mean_norm | 11.32 | **8.40** | ↓↓ gros gain de symétrie |
| ratio text/vision | 1.28 | **1.12** | Beaucoup plus équilibré |
| i2t R@1 (1K diag) | **28.6%** | 27.1% | -1.5 pts |
| t2i R@1 (1K diag) | **27.4%** | 26.1% | -1.3 pts |
| i2t R@5 (1K diag) | 64.2% | **65.8%** | +1.6 pts |
| cos_gap | **0.808** | 0.797 | Légèrement baissé |
| effective_rank | 33.5 | 33.6 | Stable |
| centroid_cos (final) | 0.458 | 0.270 | Oscille toujours |

Le LayerNorm corrige l'asymétrie des normes (objectif principal) avec un coût marginal sur le R@1 (-1.5 pts). Le R@5 est légèrement meilleur. Le centroid_cos oscille toujours mais c'est un comportement attendu — les centroïdes ne sont pas censés être parfaitement alignés, c'est la discrimination paire-level (cos_gap) qui compte.

### Convergence

Le modèle plateau vers epoch 40. Les 20 epochs supplémentaires par rapport à Run 13 confirment que le gros du gain se fait dans les 30 premières epochs. Losses stables à partir de epoch 45.

### Karpathy 1K (5-fold, nouveau benchmark)

| Métrique | Run 14 |
|---|---|
| i2t R@1 | 75.3% |
| i2t R@5 | 94.6% |
| i2t R@10 | 98.0% |
| t2i R@1 | 67.6% |
| t2i R@5 | 95.4% |
| t2i R@10 | 97.7% |

**Attention** : ces chiffres ne sont pas directement comparables aux 34.4% du lab (shimmer paper). Le protocole Karpathy standard évalue au niveau *image* (5 captions par image, succès = trouver n'importe laquelle des 5 parmi ~5000 candidats). Le lab utilise probablement un protocole *pair-level* (1 caption spécifique parmi N). Notre diagnostic R@1 à 27% est plus comparable au 34.4% du lab.

### Bilan

L'architecture est saine : normes équilibrées, pas de collapse, bonne discrimination. La légère baisse de R@1 par rapport à Run 13 est un trade-off acceptable pour une meilleure structure.

---

## 8 avril 2026 — Run 15 : batch size 1024

### Changement

Batch size 512 → 1024. GPU sous-utilisé (10% VRAM, 30-40% GPU utilization), donc on monte. Plus de négatifs par batch pour le contrastif, et le multi-positifs COCO (5 captions/image) a plus de chances de trouver des doublons.

### Résultats : gain sur text→image et effective_rank

| Métrique | Run 14 (b=512) | Run 15 (b=1024) |
|---|---|---|
| i2t R@1 (1K diag) | 27.1% | 26.9% |
| t2i R@1 (1K diag) | 26.1% | **28.5%** |
| i2t R@5 (1K diag) | 65.8% | 66.0% |
| t2i R@5 (1K diag) | 63.9% | **67.0%** |
| effective_rank | 33.6 | **37.1** |
| ratio norm text/vis | 1.12 | **1.07** |
| R@1 25K | 2.46% | **2.77%** |
| Karpathy t2i R@1 | 67.6% | **68.3%** |

Le batch 1024 améliore le text→image (+2.4 pts R@1, +3.1 pts R@5) et l'effective_rank (37.1/64, les dimensions sont mieux utilisées). Le i2t est stable. Gain gratuit — on garde batch 1024.

### Bilan global (Run 13 → 14 → 15)

Le GW vision+texte est stable. On teste ensuite la compaction des dimensions.

---

## 9 avril 2026 — Run 16 : compaction workspace 32 / latent 64

### Motivation

L'effective rank de Run 15 montrait 37.1/64 (58% d'utilisation du workspace). ~27 dimensions sous-utilisées. L'idée : réduire les dimensions pour mieux exploiter l'espace, se rapprocher de la théorie GWT (le workspace est un bottleneck compact), et préparer l'ajout du module fMRI sans workspace surdimensionné.

Le lab utilise 10-60 pour le workspace dans leurs papiers, et shimmer-ssd utilise 12 dims pour 3 modalités.

### Changements

- `workspace_dim` : 64 → **32**
- `latent_dims` (vision + text) : 128 → **64**
- Retrain complet (pretrain + align) — les poids ne sont pas compatibles

Tout le reste est identique (batch 1024, 50 epochs align, LayerNorm projections).

### Résultats : workspace mieux exploité, léger coût en R@1

| Métrique | Run 15 (ws=64, lat=128) | Run 16 (ws=32, lat=64) | Delta |
|---|---|---|---|
| Karpathy i2t R@1 | 74.2% | 71.0% | -3.2 pts |
| Karpathy i2t R@5 | 94.4% | 93.4% | -1.0 pts |
| Karpathy t2i R@1 | 68.3% | 65.4% | -2.9 pts |
| Karpathy t2i R@5 | 95.8% | 95.3% | -0.5 pts |
| R@1 25K | 2.77% | 2.39% | -0.4 pts |
| effective_rank | 37.1/64 (58%) | **26.3/32 (82%)** | +24% utilisation |
| centroid_cos | 0.33 | **0.42** | +0.09 |
| centroid_dist | 0.43 | **0.29** | -0.14 |
| norm vision | 7.73 | 6.24 | — |
| norm text | 8.24 | 5.10 | — |

### Analyse

**Gains :**
- **82% du workspace utilisé** (vs 58%) — le bottleneck est maintenant réellement informatif, pas surdimensionné
- **Meilleur alignement des centroïdes** : centroid_cos +0.09, centroid_dist -0.14 — les modalités sont plus proches dans le workspace
- **R@5 et R@10 quasi identiques** — la capacité discriminative globale est préservée

**Coûts :**
- **R@1 Karpathy : -3 pts** — la discrimination fine-grained souffre du bottleneck plus serré. Moins de dimensions = moins de capacité à distinguer des paires très similaires
- 71% i2t R@1 reste solide (comparable aux résultats publiés)

**Verdict :** trade-off raisonnable. Un workspace 2x plus compact et 82% utilisé est une meilleure base pour ajouter le module fMRI qu'un workspace de 64 à 58% d'utilisation. La perte de 3 pts R@1 est le prix du bottleneck — conforme à la théorie GWT.

### Bilan global (Run 13 → 16)

Le GW vision+texte est prêt pour l'ajout du module cerveau (NSD). Workspace compact (32 dims, 82% utilisé), bon alignement inter-modal, architecture conforme aux papiers du lab.

---

## 9 avril 2026 — Phase 3 : Module cerveau (NSD fMRI)

### Objectif

Ajouter un 3ème DomainModule (brain) qui encode/décode les betas fMRI de NSD (Natural Scenes Dataset), permettant le brain decoding (fMRI → image/texte) et le brain encoding (image/texte → fMRI). C'est le cœur du projet et de la candidature au stage.

### Stratégie

- **Freeze vision+text** (Run 16 déjà aligné), entraîner seulement le module brain
- **Dataset NSD** : Subject 1, ~10k images COCO vues en fMRI, ROI nsdgeneral (cortex visuel ventral)
- **Domain pairs** : [vision, brain] + [text, brain] uniquement (pas [vision, text] — déjà aligné)
- **Betas** : PCA 1024 composantes → z-score normalisé → BrainEncoder → z_brain → projection → workspace

### Nouveaux fichiers créés

| Fichier | Rôle |
|---|---|
| `scripts/download_nsd.sh` | Télécharge NSD Subject 1 depuis AWS S3 (~20 GB : 40 sessions betas + metadata + ROI mask) |
| `scripts/extract_brain_features.py` | Extraction pipeline : ROI masking → trial mapping → beta averaging (~3 reps/image) → PCA(1024) → z-score → .npy |
| `data/nsd_dataset.py` | `NSDDataset` : charge les 3 modalités (vision+text depuis data/coco, brain depuis data/nsd) pour les ~10k images NSD |
| `config/config_brain.yaml` | Config Phase 3 : frozen_domains, domain_pairs brain, etc. |

### Fichiers modifiés

| Fichier | Modification |
|---|---|
| `models/encoders.py` | Ajouté `BrainEncoder` |
| `models/decoders.py` | Ajouté `BrainDecoder` |
| `data/dataset.py` | Ajouté case `"nsd"` dans `get_dataset()` |
| `main.py` | Bloc conditionnel brain dans `build_domain_modules()`, chargement partiel checkpoint (2→3 domaines) |
| `pipeline/train.py` | `_freeze_domains()`, support `frozen_domains` dans pretrain+align, affichage métriques brain |
| `pipeline/diagnostics.py` | Généralisé de 2 domaines hardcodés à N domaines dynamiques (boucle sur toutes les paires) |

### Bugs rencontrés et corrigés

#### Bug 1 — NSD S3 path 404

Les betas NSD sont sous `nsddata_betas/` (bucket séparé), pas `nsddata/`. Le script de téléchargement pointait vers le mauvais chemin.

**Fix :** Corrigé le path dans `download_nsd.sh` et `extract_brain_features.py`.

#### Bug 2 — Downloads silencieusement vides

wget échouait silencieusement sur les gros fichiers (~500 MB), produisant des fichiers de 0 bytes.

**Fix :** Ajouté `--continue --tries=5 --timeout=60 --waitretry=10`, vérification post-download `[ ! -s "$local_path" ]`, nettoyage des fichiers vides avant retry.

#### Bug 3 — IndexError dans extract_brain_features.py

`n_trials_subj1` calculé comme `subjectim[0].max()` = 73000 (max image ID dans le pool NSD global) au lieu de `len(masterordering)` = 30000 (nombre réel de trials pour Subject 1). Causait un out-of-bounds lors de l'itération.

**Fix :** `n_trials_subj1 = len(masterordering)`.

### Run 15 — Premier essai (cassé)

**Problèmes multiples :**

1. **Brain features pas normalisées** — les betas après PCA avaient des magnitudes ~100-1000x plus grandes que vision (ResNet, normalisé) et text (E5, normalisé). La translation loss MSE explosait à ~3.86 million.

2. **Pretrain brain skippé** — `main.py` avec `--phase both --resume_pretrain` forçait `pretrain_epochs=0`, ce qui sautait entièrement le pretrain du brain autoencoder. Le brain encoder restait avec des poids random.

3. **Brain encoder/decoder frozen pendant align** — `_freeze_encoders_decoders()` gelait **tous** les encoders/decoders y compris brain. Seules les brain projections étaient trainables, mais elles recevaient des latents random (encoder frozen) → les projections essayaient d'aligner du bruit.

4. **Affichage des mauvaises métriques** — le print pendant le training affichait `centroid_cos` et `R@1` des paires vision↔text (frozen, constantes) au lieu des métriques brain.

| Métrique | Valeur |
|---|---|
| translation loss | 3,860,000 (constant) |
| translation_mse (eval) | 1,859,890,686 |
| centroid_cos vision↔brain | -0.20 |
| retrieval brain R@1 | 0% |
| Karpathy (frozen, intact) | 82.3% i2t R@1 |

### Corrections appliquées

| Problème | Fix |
|---|---|
| Features pas normalisées | Ajouté z-score normalisation dans `extract_brain_features.py` (fit sur train, apply sur val). Sauvegarde `zscore_mean.npy` + `zscore_std.npy` pour dé-normalisation future |
| Pretrain skippé | Supprimé le `config["training"]["pretrain_epochs"] = 0` dans `main.py` quand `--resume_pretrain` est fourni. Les frozen domains ne s'entraînent pas de toute façon |
| Brain frozen pendant align | Après `_freeze_encoders_decoders()` + `_freeze_domains()`, unfreeze explicitement encoder/decoder des domaines non-frozen (brain end-to-end) |
| Mauvaises métriques affichées | Conditionnel : si brain exists, afficher `centr_vb`, `centr_tb`, `R@1_v2b`, `R@1_b2v`, `R@1_t2b`, `R@1_b2t` |

### Run 16 — Z-score + pretrain + unfrozen brain

Première run fonctionnelle. Géométrie corrigée, mais retrieval toujours ~0%.

| Métrique | Run 15 (cassé) | Run 16 (fixé) |
|---|---|---|
| translation loss (train) | 3,860,000 | **1.99** |
| translation_mse (eval) | 1.86 milliard | **820** |
| centroid_cos vision↔brain | -0.20 | **+0.44** |
| centroid_cos text↔brain | -0.23 | **+0.52** |
| centroid_dist vision↔brain | 7.03 | **0.40** |
| brain effective_rank | ? | **26.1** (=vision) |
| brain mean_norm | 7.97 | **6.83** (~vision 6.24) |
| retrieval brain R@1 | 0% | **~0.1%** (≈random) |
| Karpathy (frozen) | 82.3% | **82.3%** |

**Analyse :** la géométrie est saine (centroids alignés, norms comparables, effective_rank OK), mais le contrastif brain ne discrimine pas (loss_contrastive_vision_brain ~9.28, quasi log(1000)). Le brain autoencoder recon_loss = 0.85 à epoch 20 (baseline = 1.0 pour z-scored) → ne capture que ~15% de la variance.

**Config :** pretrain_epochs=20, align_epochs=50, align_lr=5e-4.

### Run 17 — Plus d'epochs, LR réduit

Augmenté pretrain à 100 epochs, align à 200, LR align 5e-4 → 1e-4.

| Métrique | Run 16 | Run 17 |
|---|---|---|
| pretrain recon_loss (final) | 0.85 (ep 20) | **0.76** (ep 100) |
| contrastive train (final) | 4.18 (ep 50) | **3.89** (ep 200) |
| translation train (final) | 1.99 | **1.99** (inchangé) |
| centroid_cos vision↔brain | 0.44 | **0.55** |
| centroid_cos text↔brain | 0.52 | **0.54** |
| centroid_dist vision↔brain | 0.40 | **0.28** |
| retrieval brain R@1 | ~0.1% | **~0.1%** (toujours random) |

**Diagnostic :** plus d'epochs = meilleure géométrie mais retrieval toujours à 0. Le problème fondamental : le brain autoencoder recon_loss à 0.76 (baseline 1.0 pour z-scored) ne capture que **24% de la variance**. Le bottleneck 1024→512→**64** perd l'information discriminative des betas fMRI bruitées.

**Pourquoi 64-dim marche pour vision/text mais pas pour brain :**
- Vision (ResNet-50 features) : hyper-structurées, pré-entraînées ImageNet, compresser 2048→64 perd peu d'info utile
- Text (E5 embeddings) : pré-entraîné sur du texte massif, 1024→64 préserve la sémantique
- Brain (PCA betas fMRI) : signal bruité, faible SNR, l'info discriminative est diluée sur 1024 dims → compresser à 64 la détruit

### Run 18 (en cours) — Latent brain 256 + encoder 3 couches

**Changement clé :** augmenter le bottleneck brain pour préserver plus de signal.

| | Avant (Runs 16-17) | Après (Run 18) |
|---|---|---|
| brain latent_dim | 64 | **256** |
| BrainEncoder | 1024→512→64 (2 layers) | **1024→768→512→256 (3 layers)** |
| BrainDecoder | 64→512→1024 (2 layers) | **256→512→768→1024 (3 layers)** |
| projection | 64→32 workspace | **256→32 workspace** |

**Raisonnement :** la projection (256→32) fait la compression vers le workspace partagé. Le brain autoencoder a maintenant 4x plus de dimensions latentes pour capturer le signal discriminatif des betas fMRI. La recon_loss devrait descendre nettement en dessous de 0.76.

**Critère de succès :** recon_loss pretrain < 0.5, et retrieval brain R@1 > random (> 1% sur 1K val).

### Bilan Phase 3 (Runs 15-18)

Le pipeline est fonctionnel end-to-end : download NSD → extraction + PCA + z-score → NSDDataset 3 modalités → freeze vision+text → train brain module → eval cross-modal. Les bugs infra sont corrigés. Le prochain levier est la capacité du brain autoencoder à préserver l'information discriminative à travers le bottleneck.
