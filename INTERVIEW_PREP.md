# Interview Prep: Multi-Scale Graph Diffusion for Protein-Conditional Ligand Generation

**Scope:** everything you need to answer any question — technical, scientific, engineering, or behavioural — arising from this repository.
**Source of truth:** `Graph Based.ipynb` (44 cells), `README.md`, `requirements.txt`, committed outputs, `checkpoints/multiscale_diffusion_legacy/*.png`. Every number quoted below is taken from the notebook's own stored cell outputs, not from memory.

> **How to use this document**
> 1. Read §1–§3 until you can give the 30-second and 2-minute pitch without notes.
> 2. Read §12 (**Known defects**) twice. This is the highest-leverage section. An interviewer who reads the code will find these; your value is demonstrated by having found them first and knowing the fix.
> 3. Drill §13 (Q&A bank, 100+ questions) out loud.
> 4. The night before: §19 (cram sheet) only.
>
> **Framing rule for the whole interview:** present this as a *research prototype and systems-engineering exercise* that produced a working end-to-end pipeline and a clear-eyed list of what breaks — **not** as a validated drug-discovery method. The second framing is indefensible (no baselines, no docking, no held-out-protein split). The first is strong and honest, and it converts every bug in §12 into evidence of technical judgment.

---

## Table of contents

1. [Elevator pitches (30s / 2min / 5min)](#1-elevator-pitches)
2. [Hard fact sheet — numbers you must know cold](#2-hard-fact-sheet)
3. [Domain primer: the drug-discovery problem](#3-domain-primer)
4. [Data: source, schema, splits](#4-data)
5. [Stage 1 — Protein encoding with ESM-2](#5-stage-1--protein-encoding-with-esm-2)
6. [Stage 2 — Multi-scale ligand representation (Murcko / BRICS / physics)](#6-stage-2--multi-scale-ligand-representation)
7. [Stage 3 — Dataset assembly, PyG batching, collation](#7-stage-3--dataset-assembly-and-batching)
8. [Model architecture, component by component](#8-model-architecture)
9. [Diffusion mathematics from first principles](#9-diffusion-mathematics)
10. [Training engineering](#10-training-engineering)
11. [Results — and what the numbers actually mean](#11-results-and-what-they-mean)
12. [Known defects, ranked, with fixes](#12-known-defects-ranked)
13. [Technical Q&A bank (100+ questions)](#13-technical-qa-bank)
14. [Comparison to the literature](#14-comparison-to-the-literature)
15. [v2 roadmap: what I would do differently](#15-v2-roadmap)
16. [Behavioural interview: STAR stories](#16-behavioural-star-stories)
17. [Live-coding and whiteboard drills](#17-live-coding-and-whiteboard-drills)
18. [Questions to ask the interviewer](#18-questions-to-ask)
19. [One-page cram sheet](#19-one-page-cram-sheet)

---

## 1. Elevator pitches

### 30 seconds (use for "walk me through a project")

> "I built an end-to-end pipeline that generates small-molecule ligands conditioned on a target protein sequence. A frozen ESM-2 650M encoder turns the protein into per-residue embeddings; a learned attention encoder compresses that into conditioning vectors. The ligand side is decomposed into three scales — Bemis-Murcko scaffold, BRICS fragments, and the complete molecule — and I train a separate conditional graph-diffusion model at each scale, hierarchically, with a joint binding-affinity head. It runs end to end on 20k protein–ligand pairs from BindingDB-derived data, generates molecules, and renders them with drug-likeness properties. It is a prototype: the generation quality isn't validated against baselines, and I can give you a precise list of what I'd fix first."

The last sentence is deliberate. It pre-empts the "so does it work?" trap and signals calibration.

### 2 minutes (structure: problem → data → method → engineering → result → honest limits)

1. **Problem.** Finding a molecule that binds a given protein is a search over ~10⁶⁰ drug-like molecules. Virtual screening can only rank what's already in a library. Conditional generative models aim to *propose* novel candidates for a specific target, which is the de-novo design problem.
2. **Data.** `jglaser/binding_affinity` on Hugging Face (BindingDB/PDBbind-derived protein sequence + ligand SMILES + binding affinity as pKd-style `neg_log10_affinity_M`). I used 20,000 pairs: 11,795 unique proteins, 16,525 unique ligands, affinity 0.40–15.00 (mean 6.43), mean 33.6 heavy atoms per ligand, mean protein length 281 residues.
3. **Representation — the core idea.** Instead of generating atoms one at a time, I mirror how medicinal chemists think: a *scaffold* (the ring system that fixes the geometry), *fragments* (the substituents that tune potency and properties), and the *assembled molecule*. I extract these with RDKit — `MurckoScaffold` for level 1, `BRICS.BRICSDecompose` for level 2 (mean 5.5 fragments/molecule, 9,683 unique scaffolds), full graph for level 3 — and run a conditional denoising diffusion model at each level, where level 2 is conditioned on the pooled scaffold embedding and level 3 on scaffold + fragment embeddings.
4. **Conditioning.** ESM-2 `esm2_t33_650M_UR50D`, frozen, 1280-d per residue, with 1022-token chunking and overlap-averaging for long proteins. A 3-layer multi-head self-attention encoder with mean pooling produces three 128-d conditioning vectors, one per diffusion level, so each scale can attend to different protein information.
5. **Engineering.** The data pipeline is the real work: ESM inference checkpointed in 5,000-protein chunks; RDKit ligand processing across 10 worker processes with per-SMILES resumption so a crash costs minutes not days; a custom PyG collate that handles variable fragment counts and variable protein lengths; AMP, gradient accumulation (effective batch 16 from physical batch 4), `ReduceLROnPlateau`, full optimizer/scaler/scheduler checkpointing for resume.
6. **Result.** Trains stably to a hierarchical validation loss of 0.945 over 241 epochs; generates molecules for unseen proteins that RDKit can sanitize (e.g. 21 atoms, MW 274, QED 0.64, 3 rings) with an affinity prediction head attached.
7. **Honest limits.** Three things I'd fix before claiming anything scientific: a protein-embedding ID-collision bug that corrupts conditioning for the proteins beyond the first chunk; node features are never standardised, so the forward diffusion never reaches a Gaussian prior; and the generated topology actually comes from hand-written random graph templates, not from the model — the diffusion only refines continuous node features that are then discarded at molecule-reconstruction time. Fixing those three is my v2.

### 5 minutes

Use the 2-minute version, then add, in this order: the diffusion math (§9), the hierarchical curriculum and why it's a curriculum (§10), the exact evaluation gap and what metrics I'd add (§15), and one engineering war-story from §16.

---

## 2. Hard fact sheet

Memorise this table. Interviewers trust specificity.

### Data

| Fact | Value |
|---|---|
| Dataset | `jglaser/binding_affinity` (Hugging Face), `split="train[:200000]"`; **20,000 rows actually returned** |
| Saved as | `data/binding_data_200k.parquet` (name is aspirational; content is 20k) |
| Unique proteins | 11,795 |
| Unique ligands (processed) | 16,525 |
| Combined pairs after join | **19,989** |
| Dropped | 11 ligands >150 atoms; 85 proteins truncated at 1,000 residues; 0 "missing" |
| Affinity label | `neg_log10_affinity_M` (pKd/pKi-like), range **0.40 – 15.00**, mean **6.43** |
| Protein length | 24 – 1,000 residues (truncated), mean **280.66** |
| Molecular weight | 57.89 – 2,498.03, mean **483.83** |
| Heavy atoms | 4 – 177, mean **33.62**; bonds mean 35.64 |
| Fragments/molecule | mean **5.51** |
| Unique Murcko scaffolds | **9,683**; top-5: benzene `c1ccccc1` (1,068), an adenosine-like purine-THF `c1ncc2ncn([C@H]3CCCO3)c2n1` (430), tetrahydropyran `C1CCOCC1` (225), pyrrolidine `C1CCNC1` (172), a uridine-like `O=c1ccn([C@H]2CCCO2)c(=O)[nH]1` (116) |
| Split | 80 / 10 / 10 random over *pairs*, `random_state=42` → 15,975 train / 1,998 val / 1,997 test |
| **Actually used per epoch** | **500 train, 50 val** (`max_samples_per_epoch=500`), always the *first* 500 — see §12.6 |

### Model

| Hyperparameter | Value |
|---|---|
| `node_dim` | 8 continuous atom features |
| `atom_embed_dim` | 32 (`nn.Embedding(119, 32)` over atomic number) |
| `full_node_dim` | 8 + 32 = **40** (what is actually diffused) |
| `edge_dim` | 5 bond features |
| `protein_dim` | 1280 (ESM-2 650M hidden size) |
| `hidden_dim` | 256 |
| `cond_dim` | 128 |
| `time_emb_dim` | 64 (sinusoidal → 2-layer MLP) |
| `num_steps` T | 1000 |
| `beta_schedule` | **cosine** (linear also implemented) |
| `num_layers` | 3 conditional graph blocks per denoiser |
| GAT heads | 4 (`GATv2Conv`, `256/4 = 64` per head) |
| dropout | 0.1 |
| Denoiser input dims | L1: 40; L2: 40+128=168; L3: 40+128=168 |
| Denoiser output | 40 (predicts ε over the full node vector) |
| Affinity head | `Linear(384→256) → LayerNorm → ReLU → Dropout → Linear(256→1)`, input = concat(scaffold 128, fragment 128, molecule 128) |

### Training

| Setting | Value |
|---|---|
| Optimizer | AdamW, `lr=5e-5`, `weight_decay=1e-6` |
| Scheduler | `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)` |
| Batch size | 4 physical × 4 accumulation = **16 effective** |
| Mixed precision | `torch.amp.autocast('cuda')` + `GradScaler` |
| Epochs | 241 (resumed from checkpoint) |
| Curriculum | `hierarchical`: 10 epochs level1 → 10 level2 → 10 level3 → joint |
| Loss weights `mode='hierarchical'` | 0.2·L1 + 0.2·L2 + **0.4·L3** + 0.2·affinity |
| Loss weights `mode='all'` | 0.4·L1 + 0.3·L2 + 0.2·L3 + 0.1·affinity |
| Checkpointing | model + optimizer + scheduler + scaler + history, every epoch |

### Reported results (verbatim from cell outputs)

| Metric | Value | What it really means (§11) |
|---|---|---|
| Best val loss (hierarchical) | **0.945062** | weighted sum; ~1.0 is the trivial "predict zero noise" baseline |
| Final val loss (hierarchical) | 1.516138 | |
| "Test" loss (`mode='all'`) | 8.029586 | computed on the **val** loader, 50 samples, dominated by a broadcast affinity-loss bug |
| Test level-1 / 2 / 3 | 0.973925 / 0.952152 / 1.155999 | all ≈ 1.0 ⇒ ε-prediction barely beats trivial |
| Generated examples | 4 molecules, 12–27 atoms, MW 154–378, QED 0.43–0.64, predicted affinity **0.42–0.70** | predicted affinity far outside the training distribution (mean 6.43) ⇒ head is not usable |
| Sampling cost | ~7–10 s per 1000-step reverse chain, one chain per scaffold + one per fragment + one for the molecule (≈4 chains/molecule) | ~30–40 s per generated molecule on one GPU |

### Repo facts

- Two commits, both 2025-04-25, author `bsr11272`. Single notebook; no package, no tests, no CI.
- `.gitignore` excludes `*.pt`, so **no weights are in the repo** — results are not independently reproducible from this checkout.
- Environment: `torch==2.6.0`, `torch-geometric==2.6.1`, `transformers==4.51.2`, `fair-esm==2.0.0` (imported but ESM is actually loaded via `transformers.AutoModel`), `rdkit` (unpinned), `pytorch-lightning==2.5.1` (installed, never used).
- Outputs show paths like `/work/SensingandInnovationLab/boda.s/...` → run on a university HPC/shared GPU cluster, Python 3.12.

---

## 3. Domain primer

You will be asked "why does this matter?" and possibly tested on basic biochemistry. Know this much.

### The problem

- A **protein target** (e.g. a kinase implicated in cancer) has a **binding pocket**. A **ligand** (small organic molecule, typically 20–40 heavy atoms, MW < 500) binds in that pocket and modulates the protein's function. That is how most small-molecule drugs work.
- **Binding affinity** is the thermodynamic strength of that interaction. Reported as a dissociation constant K_d (or K_i for inhibition, IC₅₀ for half-maximal inhibition in a functional assay). Lower K_d = tighter binding.
- The dataset's label is `neg_log10_affinity_M` = −log₁₀(K in molar) = "p-value" form: **pK_d = 6 means K_d = 1 μM; pK_d = 9 means 1 nM**. Higher is better. Each unit is a 10× improvement in potency. A drug-like lead is usually pK_d ≥ 7–8.
  - Related: ΔG° = −RT·ln(1/K_d) ≈ −1.36 · pK_d kcal/mol at 298 K. So pK_d is linear in free energy — which is *why* it is the right target for a regression head, rather than raw K_d.
- **Chemical space** of drug-like molecules is estimated at 10⁶⁰ (often quoted figure for molecules under ~500 Da) — [Established Knowledge, order-of-magnitude]. Enumerable libraries (ZINC ~10⁹, Enamine REAL ~10¹⁰) are a vanishing fraction. Hence *generative* design.

### Where generative models fit

| Approach | What it does | Limit |
|---|---|---|
| High-throughput screening | Physically assay a library | Cost; limited to what you own |
| Virtual screening / docking | Score library molecules against a pocket structure | Only ranks existing molecules; scoring functions are noisy |
| QSAR / affinity prediction | Predict potency from structure | Doesn't propose anything new |
| **De-novo generative (this project)** | Propose new molecules conditioned on the target | Validity, synthesizability, and whether the conditioning is actually used are all open problems |

### Key chemistry vocabulary you must be able to define

- **SMILES** — a string encoding of a molecular graph (`c1ccccc1` = benzene; lowercase = aromatic). Canonicalisable; RDKit does so.
- **Bemis–Murcko scaffold** — the molecule reduced to its ring systems plus the linkers connecting them, side chains stripped. The "core". Acyclic molecules have an *empty* Murcko scaffold — this matters for this project (§12.9).
- **BRICS** — Breaking of Retrosynthetically Interesting Chemical Substructures. A rule set (16 bond-environment rules) that cuts a molecule at bonds a chemist could plausibly form/break, producing fragments with labelled attachment points (`[1*]`, `[16*]`, …). Chosen over naive bond-breaking because the fragments are *synthetically meaningful*.
- **QED** — Quantitative Estimate of Drug-likeness, 0–1, a weighted desirability function over 8 descriptors (MW, logP, TPSA, HBD, HBA, rotatable bonds, aromatic rings, structural alerts). ~0.6+ is drug-like.
- **Lipinski's Rule of Five** — oral bioavailability heuristic: MW ≤ 500, logP ≤ 5, HBD ≤ 5, HBA ≤ 10. The notebook reports "Lipinski: 4/4" meaning passes-out-of-four.
- **logP** — octanol/water partition coefficient (lipophilicity). **TPSA** — topological polar surface area (membrane permeability proxy, <140 Å² for oral drugs).
- **ETKDGv3** — RDKit's Experimental-Torsion Knowledge Distance Geometry conformer generator; produces a 3D conformer from the 2D graph. Then **MMFF94** or **UFF** force-field minimisation gives a strain energy.
- **Rotatable bonds / stereocenters / fraction Csp3** — flexibility, chirality, and 3D-character descriptors.

### Honest statement of scientific scope

This model is **2D-graph, sequence-conditioned**. It does **not** use the protein's 3D structure or pocket geometry, and it does not place the ligand in the pocket. That distinguishes it from structure-based generators (DiffSBDD, TargetDiff, Pocket2Mol). Sequence-conditioning is cheaper and applies to targets with no solved structure, but it is a much weaker conditioning signal — you should say this proactively, because it is the first question a computational chemist will ask.

---

## 4. Data

### Source and schema

```python
dataset = load_dataset("jglaser/binding_affinity", split="train[:200000]")   # returned 20,000 rows
df.to_parquet('data/binding_data_200k.parquet')
```

Columns used: `seq` (protein amino-acid sequence), `smiles` (ligand), `neg_log10_affinity_M` (label; fallback to `affinity`, then 0.0).

Note on provenance: `jglaser/binding_affinity` is a curated merge of public binding data (BindingDB/PDBbind lineage) pairing full protein sequences with ligand SMILES and a −log₁₀ affinity. [Inference from Context — I used it as given and did not re-derive its curation; if asked where the labels come from, say exactly that and that re-auditing provenance would be a v2 task.]

**Interview trap:** *"You named the file 200k but used 20k — why?"*
Answer: the slice requested 200k but the loader returned 20,000 (cell output: `Downloaded 20000 samples`); the filename was never corrected. I kept 20k because the ESM-2 and RDKit preprocessing cost dominated — ~12k ESM forward passes and ~16.5k conformer generations — and because I was bounded by shared-cluster GPU time. **Be straight about this: it's a naming bug plus a resource-driven scoping decision, not a claim.**

### Label distribution

Affinity 0.40–15.00 with mean 6.43. The range is suspiciously wide: pK of 15 implies K_d ≈ 10⁻¹⁵ M (femtomolar), which is at the edge of physical plausibility for a small molecule, and 0.40 implies K ≈ 0.4 M, which is no binding at all. **Flag this:** a v2 would clip to a defensible window (say 2–12), inspect the tails, and standardise the target before regression. Right now the head regresses an unnormalised, heavy-tailed label with MSE — a known recipe for a head that predicts near the mean.

### Splits — and the leakage problem

```python
train_data, test_data = train_test_split(self.data, test_size=0.1, random_state=42)
train_data, val_data  = train_test_split(train_data, test_size=0.1/0.9, random_state=42)
```

This is a **random split over protein–ligand pairs**. With 11,795 proteins in 19,989 pairs, the *same protein* appears in train and test, and often the same ligand. Consequences:

- Any affinity-prediction number is optimistically biased — the model can memorise protein/ligand identity.
- Generative "novelty" is unmeasured because the test proteins are not unseen targets.

**What to say:** "The correct splits are (a) protein-cluster split — cluster sequences at e.g. 30–40% identity with MMseqs2/CD-HIT and hold out whole clusters, and (b) scaffold split on the ligand side (Bemis-Murcko), to test generalisation to new chemotypes. I used a random pair split, which inflates apparent performance; that's a limitation I'd fix before reporting any metric."

---

## 5. Stage 1 — Protein encoding with ESM-2

### What the code does (`generate_protein_embeddings`, `process_in_chunks`)

1. Load `facebook/esm2_t33_650M_UR50D` via `transformers.AutoModel`, `.eval()`, on GPU. **Frozen** — no gradients, no fine-tuning.
2. For each unique sequence:
   - If `len(seq) ≤ 1020`: one forward pass, take `hidden_states[-1][0, 1:len+1, :]` → **(L, 1280)** per-residue embeddings, dropping `<cls>` and `<eos>`.
   - If longer: sliding window of 1020 residues with **128-residue overlap** (stride 892). Accumulate each window's embeddings into a full-length `(L, 1280)` array and divide by a per-position count → **overlap-averaged** embeddings. Windows <8 residues are skipped.
3. Save per 5,000-protein chunk as `.npz` with `allow_pickle` (a dict, because sequences are ragged), skipping chunks that already exist (resumability).
4. Build `protein_to_id_map: {sequence → id}` and a combined `protein_full_embeddings_all.npz`.

### Why these choices

- **Why ESM-2 at all?** A protein language model trained on UniRef with masked-language modelling learns representations that encode structure and function; per-residue embeddings are a standard drop-in protein featurisation that avoids needing a solved structure. [Established Knowledge]
- **Why 650M (`t33`) and not 150M or 3B/15B?** 650M is the usual quality/throughput compromise: 1280-d, 33 layers. The 3B+ models need far more memory for a one-off featurisation of 12k proteins on shared hardware. Honest answer: it was the largest I could run at batch-1 in available GPU time.
- **Why the last hidden layer?** Convention and simplicity. **Known better option:** for downstream tasks, intermediate layers are often more transferable than the last, and a learned layer-weighted sum (ELMo-style scalar mix) is a cheap improvement. I did not ablate this.
- **Why keep per-residue instead of mean-pooling to one vector?** Because a pocket is local — a single mean over 300 residues washes out the residues that matter. Keeping (L, 1280) leaves the door open for attention to select relevant positions. (In practice my encoder then mean-pools anyway — see §12.5. That is an inconsistency worth owning.)
- **Why chunking with overlap rather than truncation?** ESM-2's positional handling is bounded (1024 tokens incl. specials). Truncating a 2,000-residue protein throws away the C-terminal half, which may contain the pocket. Overlap-averaging gives every residue at least one full-context embedding and smooths the window seam. Alternatives: hierarchical pooling over windows, or a long-context PLM.

### The critical bug in this stage (own it before they find it)

`generate_protein_embeddings` assigns ids **local to its own call**:

```python
batch_ids = list(range(i, min(i+batch_size, len(sequences))))   # i is local to this chunk
```

so each 5,000-protein chunk emits ids `0…4999`. Then `process_in_chunks` merges chunks into one dict keyed by that id:

```python
for j, seq_id in enumerate(chunk_ids):
    all_embeddings[seq_id] = chunk_embeddings[seq_id]        # chunk 2 overwrites chunk 1
    protein_to_id_map[unique_proteins[i + j]] = seq_id        # protein 5000+j → id j
```

**Evidence it fired:** the combined file reports `Loaded full sequence embeddings for 5000 proteins` while the map reports `Loaded mapping for 11795 proteins`. Three chunks (5000+5000+1795) collapsed to 5,000 keys.

**Consequence:** every protein outside the final chunk is paired with *another protein's* embedding. Roughly 6,795 of 11,795 proteins (~58%) are mis-conditioned, silently — lengths and shapes all remain valid, so nothing crashes. The join step therefore reported "0 missing proteins" while being substantially wrong.

**Fix (one line):** make ids global — `batch_ids = [global_offset + k for k in ...]`, or drop ids entirely and key the embedding dict by a hash of the sequence:

```python
key = hashlib.sha1(sequence.encode()).hexdigest()
all_embeddings[key] = emb            # content-addressed, collision-free across chunks
protein_to_id_map[sequence] = key
```

**How to present it:** "The single most important thing I found when auditing this project was a silent ID-collision in the chunked embedding cache. It never raised an exception — shapes were all valid — and it corrupted the conditioning signal for most proteins. The lesson I took is that ID assignment in any sharded cache must be content-addressed or globally offset, and that a pipeline needs an assertion that the number of map entries equals the number of distinct embeddings. One `assert len(set(map.values())) == len(embeddings)` would have caught it on day one."

That answer is worth more than any result the project could have produced.

---

## 6. Stage 2 — Multi-scale ligand representation

Implemented in `process_smiles_batch_multiscale`, parallelised by `generate_multiscale_ligand_representations`.

### What is computed per ligand

| Block | Contents |
|---|---|
| `properties` (12) | MW, logP, HBD, HBA, rotatable bonds, TPSA, QED, heavy atoms, ring count, aromatic rings, fraction Csp3, stereocenters |
| `functional_groups` (15 SMARTS counts) | alcohol, aldehyde, ketone, carboxylic acid, ester, amide, amine, nitrile, nitro, sulfonic acid, sulfonamide, phosphate, halogen, aromatic, heterocycle |
| `physics_features` | ETKDGv3 embedding → MMFF94 optimisation (UFF fallback) → `total_energy`; plus ring-strain heuristic (3-ring = 5.0, 4-ring = 3.0, 5/6-ring = 0.1, else 1.0), ring count, ring sizes, and a `force_field_type` provenance tag |
| `multi_scale.level1_scaffold` | Murcko scaffold SMILES + graph |
| `multi_scale.level2_fragments` | list of BRICS fragment SMILES + graphs |
| `multi_scale.level3_complete` | full molecule SMILES + graph |

### Graph featurisation (`get_molecular_graph`)

Per atom: atomic number, formal charge, hybridization (int enum), aromatic flag, total Hs, degree, chiral tag (int), is-in-ring, ring sizes, total valence.
Per bond: endpoints, bond type (int enum), conjugated, aromatic, in-ring, stereo (int), ring sizes.

The dataset layer (§7) then keeps **exactly 8** continuous atom features — `[formal_charge, hybridization, aromatic, num_h, degree, chirality, is_in_ring, valence]` — and pulls `atomic_num` out into a separate `atom_types` long tensor for the learned embedding. 5 bond features: `[bond_type, is_conjugated, is_aromatic, is_in_ring, stereo]`.

**Why split atomic number out?** Atomic number is categorical and not ordinal — treating 6 (C), 7 (N), 8 (O), 16 (S) as a scalar implies C < N < O < S on a single axis, which is chemically meaningless. An `nn.Embedding(119, 32)` lets the model learn element similarity. **This is the single best design decision in the model** and you should lead with it when asked "what did you get right?"

**The symmetric criticism you must volunteer:** the same argument applies to `hybridization`, `chirality`, `bond_type` and `stereo`, which are *also* integer enums left as scalars. They should be one-hot or embedded too. (`bond_type` is the worst: RDKit's enum has SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=12 — so "aromatic" is numerically 12× a single bond. That is actively harmful as a continuous feature.)

### Engineering: why the pipeline looks the way it does

- **`multiprocessing.Pool` with `n_jobs=10`, `batch_size=5`.** RDKit conformer generation is CPU-bound, single-threaded, and releases nothing to share — processes, not threads (Python GIL). Small batches give frequent checkpoints.
- **Per-SMILES resumption.** `processed_smiles_map.pkl` maps each completed SMILES → its batch file. On restart, already-done SMILES are filtered out and new batches get fresh indices (`highest_batch_idx + 1`) so nothing is overwritten. This is the design I'd defend hardest: ETKDG + MMFF on 16.5k molecules on a shared cluster *will* be interrupted, and idempotent, resumable stages are the difference between a week and a day.
- **`gc.collect()` + `ctypes … malloc_trim(0)`.** RDKit allocations fragment the glibc heap; freed Python objects don't return RSS to the OS, and long-running workers get OOM-killed on a shared node. `malloc_trim` forces glibc to release free pages. Wrapped in `try/except` because it's Linux/glibc-only. Expect "why is there a libc call in your ML pipeline?" — this is the answer, and it's a good one.
- **`try/except` around everything, with defaults.** 16.5k molecules from a public dataset include valence-invalid SMILES, metals, and structures ETKDG can't embed. A crash in worker 7 must not kill the run. The cost is silent degradation — which is why `force_field_type` records *how* each entry was produced (`MMFF` / `UFF` / `skipped_problematic_atoms` / `embedding_failed` / `exception`). **Point at that field as the mitigation**: failures are recorded, not hidden.
- **Atoms >18 (excluding Br=35, I=53) skip force-field optimisation.** MMFF94 has no parameters for most metals; attempting it produces garbage energies or hangs.

### The honest punchline for this section

**All of the physics features, molecular properties, and functional-group counts are computed, stored, carried through the collate function… and never fed to the network.** The model consumes only `level{1,2,3}` graphs, `atom_types`, and the protein embedding. So the README's claim of *"Physics-informed features to enhance molecular validity"* is **unsupported by the code**. Say so first; then say what you'd do with them: concatenate the physics/property vector into the affinity head, or use them as auxiliary prediction targets (multi-task regularisation), or as a guidance signal at sampling time. Interviewers respect "the feature is computed but unwired, here's how I'd wire it" far more than a defensive rationalisation.

---

## 7. Stage 3 — Dataset assembly and batching

### `create_multiscale_combined_dataset`

Joins the three artefacts on (sequence, SMILES), truncating protein embeddings at `max_protein_length=1000` residues, and emits a flat list of dicts (one per pair) pickled to `processed_data/combined_multiscale_dataset.pkl`. Keeps counters for missing proteins / missing ligands / size-filtered / truncated — good practice, and the counters are what revealed the stats quoted in §2.

**Design critique to volunteer:** a single pickle holding 19,989 entries where each entry embeds a full `(L, 1280)` float array is enormous (order of 20k × 280 × 1280 × 4 B ≈ 29 GB if float32 — it fits only because of `float32` numpy + compression at the npz stage and because the pickle holds references to the loaded dict, but as written the pipeline is memory-hostile and each of the three `Dataset` instances re-loads the whole pickle). The right design: store ligand graphs and protein embeddings separately (e.g. LMDB / HDF5 / `webdataset` shards or memory-mapped `.npy`), index by key, and load lazily in `__getitem__`. Also: protein embeddings are duplicated across every pair sharing a protein. Deduplicating saves ~40% of the data volume.

### `MultiScaleProteinLigandDataset.__getitem__`

Returns a dict: `protein_embedding (L,1280)`, `affinity (1,)`, `level1_graph`, `level2_graphs` (list), `level3_graph`, `ligand_properties`, `physics_features`, `smiles`. Asserts every graph has exactly 8 node features (defensive invariant — good).

Empty graphs (an acyclic molecule's Murcko scaffold, or a failed fragment) become a **single zero node with a self-loop** `edge_index=[[0],[0]]`, `edge_attr=zeros(1,5)`. This keeps PyG batching happy. Downside: the model receives a meaningless "molecule" as conditioning and is trained to denoise it; see §12.9.

### `collate_multi_scale_data` — the hardest piece of engineering here

Three ragged dimensions have to be reconciled:

1. **Variable protein length** → right-pad with `F.pad` to the batch max, then `torch.stack` → `(B, L_max, 1280)`.
2. **Variable atom counts** → `Batch.from_data_list` (PyG's standard trick: concatenate node features along dim 0, offset `edge_index` by cumulative node counts, and record a `batch` vector mapping node → graph index; pooling is then a `scatter` by that vector). No padding, no masking, no wasted compute — this is why PyG's "one big disconnected graph" convention is the right representation.
3. **Variable fragment counts per molecule** → the awkward one. The code builds `max_fragments` batches, where batch *k* contains "fragment *k* of every molecule", padding molecules with fewer fragments using a dummy 1-node graph.

**Why that's a defensible-but-flawed choice, and what's better.** Defensible: it keeps every tensor rectangular and lets one denoiser process all of position-*k* in parallel. Flawed: fragment *k* of molecule A has nothing to do with fragment *k* of molecule B — the ordering comes from `BRICS.BRICSDecompose`'s set iteration, so the grouping is arbitrary; the number of forward passes is set by the *most* fragmented molecule in the batch (mean 5.5, max can be far higher ⇒ wasted compute); and the dummy graphs participate in the loss (§12.8). **Better:** flatten all fragments of all molecules into one `Batch` and carry a `mol_id` vector, so pooling to per-molecule fragment features is a single `scatter_mean` over `mol_id`. One forward pass, no dummies, no arbitrary pairing. That is the first refactor I'd make.

Also note the explicit `.clone()` / `.contiguous()` calls in `ensure_proper_tensor`: with `num_workers=2`, tensors are moved through shared memory, and PyG `Data` objects that share storage with a parent tensor can cause subtle aliasing and "received 0 items of ancestry" worker errors. Cloning buys safety at the cost of a copy — a deliberate trade.

---

## 8. Model architecture

```
protein (B, L, 1280)
   └─ ProteinEncoder: Linear→LayerNorm → 3× MultiheadAttention(8 heads, residual+LN) → mean-pool
         └─ three heads → level1_cond, level2_cond, level3_cond   (B, 128) each

ligand level 1 (Murcko scaffold graph)
   x (N1, 8) + Embedding(atom_types) (N1, 32)  →  x_full (N1, 40)
   forward diffusion at t1  →  x_t
   GraphDenoiser_L1(x_t, edge_index, edge_attr, t1, level1_cond, batch) → ε̂ (N1, 40)
   GraphEncoderBlock(scaffold) → scatter_mean → scaffold_features (B, 128)

ligand level 2 (BRICS fragment graphs, one Batch per fragment slot)
   x_full (N2, 40) → diffuse at t2 → concat scaffold_features[batch] → (N2, 168)
   GraphDenoiser_L2(..., level2_cond) → ε̂ (N2, 40)
   GraphEncoderBlock(fragments) → scatter_mean → mean over slots → fragment_features (B, 128)

ligand level 3 (complete molecule graph)
   x_full (N3, 40) → diffuse at t3 → concat (scaffold_features + fragment_features)[batch] → (N3, 168)
   GraphDenoiser_L3(..., level3_cond) → ε̂ (N3, 40)

affinity:  concat[scaffold (128), fragment (128), molecule_encoder→pool (128)] → MLP → ŷ (B, 1)

loss = w1·MSE(ε̂1, ε1) + w2·MSE(ε̂2, ε2) + w3·MSE(ε̂3, ε3) + wa·MSE(ŷ, y)
```

### `ProteinEncoder`

`Linear(1280→256)` → `LayerNorm` → 3 × [`nn.MultiheadAttention(256, 8 heads)` with pre-residual and post-`LayerNorm`] → mean pooling → three separate 2-layer MLP heads → three `(B, 128)` conditioning vectors.

- **Why three heads instead of one vector reused?** Each diffusion level needs different protein information: scaffold choice is driven by pocket shape/size, fragment decoration by specific interaction partners (H-bond donors/acceptors, hydrophobic patches), final refinement by fine geometry. Separate heads let the shared trunk specialise per scale with negligible parameter cost. It's also a cheap form of multi-task conditioning.
- **Why self-attention over the residue axis rather than an LSTM?** Parallelism, and because contacts in a protein are long-range in sequence space (residues far apart in sequence are adjacent in 3D). An LSTM branch exists in the code but is unimplemented (`Print("Define encoding method first")`) — i.e. it was scaffolding, never finished. Don't pretend otherwise.
- **`nn.MultiheadAttention` default is `batch_first=False`**, hence the `x.transpose(0,1)` into `(L, B, 256)` and back. Expect a question on this; it's a classic source of silent shape bugs.
- **Cost:** attention over L up to 1000 at 256-d, 3 layers — O(L²) = 10⁶ attention entries per sample per layer. With batch 4 this is why batch size had to be small.

### `GraphEncoderBlock`

`node_proj` and `edge_proj` MLPs (Linear→LayerNorm→ReLU→Dropout) → `GATv2Conv(hidden, hidden//heads, heads=4, edge_dim=hidden)` → residual + LN → second `GATv2Conv` → `output_proj` → residual + LN.

- **Why GATv2 rather than GCN / GIN / plain GAT?**
  - It consumes **edge features** (`edge_dim=`), which matters here: bond type and aromaticity are the chemistry.
  - Attention gives *anisotropic* aggregation — different neighbours weighted differently — which is right for molecules (a carbonyl O is not interchangeable with a methyl C).
  - **GATv2 over GAT specifically:** original GAT computes attention as `a^T[Wh_i ‖ Wh_j]` with the nonlinearity applied *after* the linear scoring, which makes the ranking of keys independent of the query — "static attention". GATv2 moves the nonlinearity inside (`a^T LeakyReLU(W[h_i ‖ h_j])`), giving *dynamic* attention, strictly more expressive at the same cost. [Established Knowledge — Brody, Alon & Yahav, "How Attentive are Graph Attention Networks?", ICLR 2022.]
  - Why not GIN: maximally expressive for graph isomorphism under the WL test, but sum-aggregation without edge-feature attention; weaker fit for conditional denoising where I want per-bond modulation.
- **`hidden_dim // heads` with `heads=4` and 256 hidden** → 64 per head, concatenated back to 256. Note this requires divisibility; `hidden_dim=250` would silently change output width and break the residual add. Worth flagging as a latent fragility.
- **Residual projections** (`residual1`, `residual2`) are learned rather than identity because the width changes between stages; `LayerNorm` after the add (post-norm) stabilises the 3-deep stack.

### `ConditionalGraphBlock`

Projects the `(B, cond_dim)` condition to `hidden_dim`, **broadcasts it to nodes by indexing with the `batch` vector** (`h_cond[batch]` → `(N, hidden)`), concatenates onto node features, and runs a `GraphEncoderBlock`.

- `h_cond[batch]` is the key idiom: it is how a per-graph vector reaches per-node computation in PyG's flat representation. Expect to be asked to write this line.
- **Critique:** concatenation-at-input is the weakest conditioning mechanism. Stronger, standard options: **FiLM** (`γ(c) ⊙ h + β(c)` per layer), **AdaLN / AdaLN-Zero** as in DiT, or **cross-attention** from nodes to the *per-residue* protein embeddings (which would actually use the sequence rather than a pooled mean, and would let the model attend to pocket residues). Also, since there is no classifier-free guidance, there is **no mechanism to trade sample fidelity against conditioning strength** — and no measurement of whether conditioning is used at all. The decisive experiment I did not run: **swap the protein embedding for a random protein and check whether the outputs change**. If they don't, conditioning is ignored. Volunteer this experiment; it shows you know what "conditional" has to mean.

### `GraphDiffusion`

Holds the noise schedule as registered buffers: `beta`, `alpha = 1−beta`, `alpha_bar = cumprod(alpha)`, `sqrt_alpha_bar`, `sqrt_one_minus_alpha_bar`. `forward_diffusion(x_0, t, batch)` draws ε ~ N(0, I), looks up the per-graph coefficients, **expands them to nodes via `[batch]`**, and returns `x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε` plus the ε (the training target).

- `register_buffer` rather than `nn.Parameter`: these are constants that must move with `.to(device)` and be saved in `state_dict` but never receive gradients.
- Per-graph timesteps expanded per node ⇒ all nodes of one molecule share t, which is correct: the molecule is the diffusion object, not the atom.

### `GraphDenoiser`

`SinusoidalPositionEmbeddings(64)` → MLP → concat with condition → `Linear(cond+time → cond)` → LayerNorm/ReLU/Dropout → 3 × `ConditionalGraphBlock` → output MLP to 40 dims (ε̂ over the full node vector).

- **Sinusoidal time embedding** (the transformer trick, exp-spaced frequencies, sin ‖ cos): gives the network a smooth, multi-scale encoding of t so that nearby timesteps have nearby embeddings and the same weights can serve all 1000 steps. The alternative — a learned `nn.Embedding(1000, d)` — works but generalises worse across t and is wasteful.
- **Time is injected once, fused with the condition, at the input.** Modern practice injects it at *every* block (AdaLN/FiLM per layer). Another concrete v2 item.

### `MultiScaleGraphDiffusion.forward` — the orchestration

- Samples `t_level1/2/3` independently per level. Defensible (each level is its own diffusion process) but note the alternative of a shared t with a coupled schedule.
- `mode` selects which levels are active: `'level1' | 'level2' | 'level3' | 'all' | 'hierarchical'`. In `'level2'`/`'level3'` mode it extracts the needed upstream features directly so a single level can be trained alone.
- Level-3 conditioning is `combined_cond = level1_features + level2_features` — **additive**, with the code comment conceding "may update in future with more complex feature aggregation". Additive fusion is not identifiable (you can't tell a strong scaffold signal from a strong fragment signal). Concatenation (and widen the denoiser), a gated sum, or cross-attention are all better. Flag it; the comment proves you knew.
- Loss weights differ by mode (see §2). Nothing principled chose them; they are hand-set. The principled options: uncertainty weighting (Kendall et al. learnable log-variance per task), GradNorm, or simply normalising each level's loss by its running mean. Also: because the weights differ between `'all'` and `'hierarchical'`, **the scalar losses from the two modes are not comparable** — which is exactly the mistake that produced the "test loss 8.03 vs val 0.945" confusion in §11.

### `sample()` — read this carefully, it's where the project's real weakness lives

1. Encode the protein → three conditioning vectors.
2. **Build an initial scaffold graph by hand-written stochastic rules**: pick a type from `{simple_ring 0.5, fused_rings 0.3, bridged_system 0.1, macrocycle 0.1}`; pick a size (5–25 atoms, with ring-size priors favouring 5 and 6); sample atom types from hard-coded element distributions (e.g. C 0.7 / N 0.15 / O 0.1 / S 0.05); then **construct the edge list procedurally** (a cycle for a ring; two cycles plus a fusion bond; a ring plus a 2-atom bridge; a macrocycle with every third bond double).
3. Run a 1000-step reverse chain over the *continuous node features only*, clamping to [−5, 5].
4. Repeat for 1–8 fragments per molecule, again with hand-built topologies (`chain`/`ring`/`branch`/`functional_group`), conditioned on the pooled scaffold features.
5. Assemble: copy scaffold nodes/edges, copy fragment nodes/edges, and join each fragment's **first atom** to a **uniformly random scaffold atom** with a single bond.
6. Run a third 1000-step chain on the assembled molecule.
7. `robust_convert_graph_to_mol` builds an RDKit molecule from `atom_types` and the edge list, **adding only single bonds**, with a conservative valence cap per element, and a three-level fallback ladder ending in methane.

**The conclusion you must state plainly:** the *topology and the element identities* of every generated molecule come from steps 2/4/5 — hand-written random graph templates and random element draws — not from the diffusion model. The diffusion model only refines the 40-d continuous node vectors, and the reconstruction step **discards those vectors entirely** (it reads `atom_types` and `edge_index`, never `x`). The atom-type embedding slice is extracted (`atom_embeddings = x_full[:, node_dim:]`) and then never used. Bond orders are forced to single. Therefore:

> **The generative model, as wired, does not determine the molecules it "generates."** The pipeline is a random-graph generator plus a learned feature denoiser whose output is dropped before the molecule is built.

This is uncomfortable but it is the truth of the code, and saying it is far stronger than being caught by it. It also sets up the best answer in the whole interview, §15's v2: the fix is to make the *discrete* objects the diffusion variables — node-type and edge-type categorical diffusion (D3PM-style absorbing/uniform transition matrices, or continuous diffusion on one-hot simplices with an argmax/Gumbel decode) over a dense `N×N` adjacency with a learned atom-count prior — at which point the model genuinely generates topology.

---

## 9. Diffusion mathematics

Be able to derive this on a whiteboard. This is the most likely deep-technical probe.

### Forward process

Fix a variance schedule β₁…β_T ∈ (0,1). Define

  q(x_t | x_{t−1}) = N(x_t; √(1−β_t) x_{t−1}, β_t I).

With α_t = 1 − β_t and ᾱ_t = ∏_{s≤t} α_s, the marginal has a closed form (this is the whole trick — you can jump to any t in one step):

  **q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1−ᾱ_t) I)**  ⟹  **x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε, ε ~ N(0, I).**

*Derivation sketch to recite:* compose two Gaussians; the means multiply (√α_t√α_{t−1}) and the variances add ((1−α_t) + α_t(1−α_{t−1}) = 1 − α_tα_{t−1}); induct.

As t → T, ᾱ_T → 0 and q(x_T | x_0) → N(0, I) — **the prior is independent of the data**, which is what makes sampling possible. (Keep this sentence: §12.2's bug is exactly a violation of it.)

### Reverse process and the true posterior

The reverse conditional q(x_{t−1} | x_t) is intractable, but **conditioned on x_0 it is Gaussian**:

  q(x_{t−1} | x_t, x_0) = N(x_{t−1}; μ̃_t(x_t, x_0), β̃_t I), where

  **μ̃_t = [√ᾱ_{t−1} β_t / (1−ᾱ_t)] · x_0 + [√α_t (1−ᾱ_{t−1}) / (1−ᾱ_t)] · x_t**  
  **β̃_t = β_t (1−ᾱ_{t−1}) / (1−ᾱ_t)**

Memorise which coefficient multiplies which term: **the β_t-containing coefficient multiplies x₀; the α_t-containing coefficient multiplies x_t.** (Sanity check: at t=1, ᾱ₀=1 so the second coefficient vanishes and μ̃ = x₀ — the posterior collapses onto the data. If you ever write it with the terms swapped, this check fails.)

### Training objective

The variational bound decomposes into per-step KL terms; with the ε-parameterisation x̂₀ = (x_t − √(1−ᾱ_t) ε̂)/√ᾱ_t, each KL reduces to a weighted ‖ε − ε̂‖². Ho et al. (DDPM) drop the weights, giving the simple loss this project uses:

  **L_simple = E_{x₀, ε, t} ‖ ε − ε_θ(x_t, t, c) ‖²**  — i.e. `F.mse_loss(noise_pred, noise)`.

Why ε-prediction rather than x₀-prediction or v-prediction: ε has unit scale at every t, so the loss is well-conditioned across timesteps; x₀-prediction is better at high noise, v-prediction (`v = √ᾱ ε − √(1−ᾱ) x₀`) interpolates and is preferred for very noisy/short schedules. Know all three names.

### Sampling (ancestral / DDPM)

  x_{t−1} = μ̃_t(x_t, x̂₀(x_t, ε̂)) + σ_t z, z ~ N(0, I), σ_t² = β̃_t (or β_t; both are used in practice).

### Schedules

- **Linear** (as implemented): β from 1e-4 to 2e-2.
- **Cosine** (as used, from Nichol & Dhariwal "Improved DDPM"): ᾱ_t = cos²((t/T + s)/(1+s) · π/2) with s = 0.008, normalised by ᾱ₀, then β_t = 1 − ᾱ_t/ᾱ_{t−1}, clipped to [1e-4, 0.9999]. The code implements exactly this.
  **Why cosine:** linear schedules destroy information too early — ᾱ_t collapses in the first third of the chain, so most of the 1000 steps are spent on near-pure noise and contribute nothing. Cosine keeps ᾱ_t high longer and decays smoothly, which empirically improves sample quality and especially helps at low resolution / small objects. Molecules are "small objects" (tens of nodes), so this is the right default.
- The clip at 0.9999 prevents α_t = 0 (which would make √ᾱ zero and the x̂₀ reconstruction divide by zero).

### Faster sampling (be ready for "1000 steps is slow")

- **DDIM**: deterministic (σ=0) non-Markovian reverse process, lets you take a strided subsequence of timesteps — 20–100 steps for similar quality. Essentially free to implement on top of a trained ε-model. **This is the first thing I'd add**, since the current cost is ~4 × 1000 network evaluations per molecule (§2 shows 7–10 s per chain).
- Also: DPM-Solver / higher-order ODE solvers; progressive distillation / consistency models; or simply training with fewer steps and a cosine schedule.

### Discrete / graph diffusion (the question that separates candidates)

Molecules are **discrete**: atom types are categorical, bonds are categorical, and the graph is permutation-invariant. Gaussian diffusion in ℝ^d on integer-coded features is a mismatch. Know the alternatives:

- **D3PM** (Austin et al.) — diffusion with categorical transition matrices Q_t (uniform, absorbing/`[MASK]`, or nearest-neighbour), loss on the categorical posterior.
- **DiGress** (Vignac et al.) — discrete diffusion over node *and* edge categories with a graph transformer, plus auxiliary structural features (cycle counts, spectral features) to compensate for the limited expressivity of message passing. The reference point for 2D molecular graph generation.
- **EDM / E(3)-equivariant diffusion** (Hoogeboom et al.) — joint continuous diffusion over 3D coordinates with an EGNN, categorical for types; the standard for 3D conformer generation.
- **MolDiff** — explicitly addresses the atom–bond inconsistency problem that arises when types and bonds are diffused independently.
- **Simplex / Dirichlet diffusion** — continuous diffusion constrained to the probability simplex, then argmax; a middle road if you want to keep a Gaussian formulation.

Then say: "My implementation is continuous Gaussian diffusion over a mixed integer/binary feature vector with a fixed, externally supplied topology. That is the central methodological compromise, and D3PM/DiGress-style discrete diffusion over node *and* edge categories is what I'd replace it with."

### Permutation equivariance

A molecular graph has no canonical node order. Message-passing GNNs are permutation-*equivariant* by construction (node outputs permute with the input) and `scatter_mean` pooling is permutation-*invariant*, so the architecture respects this — good. But the *sampling* code is not: it generates templates with a fixed index convention (node 0 is a ring start, fragment node 0 is the attachment atom), which quietly imposes an ordering the training distribution never had.

---

## 10. Training engineering

### Hierarchical curriculum

```python
stages = [('level1', 10), ('level2', 10), ('level3', 10)]   # then joint 'hierarchical'
```

Train the scaffold denoiser alone, then the fragment denoiser (conditioned on scaffold features), then the molecule denoiser, then everything jointly.

**Why:** levels 2 and 3 are conditioned on *learned* features of level 1. If everything trains at once from random init, level 2 learns to denoise using garbage conditioning, and the gradient signal through `scaffold_encoder` is noise-on-noise. Staging gives downstream levels a stable input distribution — the same logic as curriculum learning, progressive growing of GANs, or layer-wise pretraining.

**Weaknesses to volunteer:** (a) no layer freezing — the "level-1 stage" still updates all parameters that receive gradients, so earlier stages are not protected; (b) the stage bookkeeping is inconsistent (`stage_start = min(start_epoch // 2, len(stages))` and `start_epoch = 6` assume **2** epochs per stage while the table says **10**, and the comments say 2) — a latent resume bug; (c) `eval(f'train_l{stage_idx+1}')` is used for logging, which is both fragile and unnecessary (`[train_l1, train_l2, train_l3][stage_idx]`); (d) catastrophic forgetting of level-1 during later stages is unmeasured.

### Mixed precision + gradient accumulation

```python
with torch.amp.autocast(device_type='cuda'):
    results = self.model(batch, mode=mode)
    loss = results['loss'] / self.accumulation_steps
self.scaler.scale(loss).backward()
if (i+1) % self.accumulation_steps == 0 or (i+1) == len(self.train_loader):
    self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad()
```

Be ready to explain each piece:
- **autocast** runs matmuls/convs in fp16/bf16, keeps reductions and norms in fp32 → ~2× throughput and ~half the activation memory on tensor-core GPUs.
- **GradScaler** multiplies the loss by a large factor before backward so small fp16 gradients don't underflow to zero, then unscales before the step; it detects inf/NaN and skips those steps, adapting the scale. Without it, fp16 training silently stops learning.
- **Why divide the loss by `accumulation_steps`?** Gradients accumulate by summation; dividing makes the accumulated gradient the *mean* over the effective batch, so the gradient magnitude (and hence the appropriate LR) is independent of the accumulation factor.
- **Why accumulate at all?** Physical batch 4 was forced by memory — O(L²) protein attention with L up to 1000, plus three graph denoisers. Accumulation recovers the statistical benefit of batch 16 (less gradient noise, stable BatchNorm-free training) without the memory.
- **The `or (i+1) == len(loader)` tail condition** flushes a partial accumulation group at epoch end — otherwise the last few batches' gradients leak into the next epoch.
- **Missing:** `torch.nn.utils.clip_grad_norm_` after `scaler.unscale_(optimizer)`. With diffusion losses and unnormalised features, gradient spikes are likely; no clipping is a real gap.

### Optimizer, schedule, checkpointing

- **AdamW** with `weight_decay=1e-6`: decoupled weight decay (decay applied to the weights, not folded into the adaptive moment estimates as in L2-with-Adam). 1e-6 is very small — effectively no regularisation; dropout 0.1 is doing that work.
- **`lr=5e-5`** is low for a from-scratch model (3e-4 is the usual diffusion default, and the code's own default is 3e-4). With only 500 unique samples per epoch, a low LR was probably masking instability; worth saying "I'd run an LR sweep with a warmup + cosine decay instead of plateau-based reduction."
- **`ReduceLROnPlateau(factor=0.5, patience=5)`** — halve on 5 epochs of no val improvement. Defensible for an unknown-landscape prototype; less good than warmup + cosine for diffusion, where long stable training at a known LR is standard.
- **Checkpointing saves model + optimizer + scheduler + scaler + history.** This is correct and often done wrong — resuming with a fresh optimizer throws away Adam's moments and causes a visible loss spike; resuming without the scaler state resets loss scaling. Call this out as something you got right.
- `verbose=True` on the scheduler triggers a deprecation warning in torch 2.6 (visible in output) — cosmetic.

### Reproducibility

`torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`, `ETKDGv3.randomSeed = 42`. Missing for full determinism: `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic = True`, `DataLoader(worker_init_fn=...)` and a `generator=`, and `torch.use_deterministic_algorithms(True)` (note `scatter` ops are nondeterministic on CUDA by default). Also: the notebook format itself is hostile to reproducibility — out-of-order execution, and the model/encoder classes are redefined across cells.

---

## 11. Results and what they mean

### What the notebook reports

```
Best validation loss (hierarchical): 0.945062
Final validation loss:                1.516138
Test Loss (mode='all'):               8.029586
  Level 1: 0.973925   Level 2: 0.952152   Level 3: 1.155999
Generated: 4 molecules; predicted affinity 0.42 / 0.70 / 0.56 / 0.59
```

### The three things you must say about these numbers *before* being asked

**1. An ε-prediction MSE of ~1.0 is the trivial baseline.** The target is ε ~ N(0, I), so E‖ε‖²/d = 1. A model that outputs **zero** scores 1.0. Levels at 0.95–1.16 therefore mean the denoisers are, on average over t, barely distinguishable from — and at level 3 *worse than* — predicting no noise at all. (Worse-than-1.0 is possible because the loss averages over timesteps and features with wildly different scales.) **Any claim that "the model learned to denoise" is unsupported by this metric.** The right diagnostic is loss *stratified by timestep*: a working ε-model is near 0 at large t (where x_t ≈ ε and the task is nearly trivial) and above 0 at small t. A flat ~1.0 across t means it isn't learning the signal. Report that stratified curve, not a scalar.

**2. The "test loss" of 8.03 is not a test loss.** `Trainer.validate()` iterates `self.val_loader` — there is no test loader wired in — so the number is computed on the **validation** split, and on only the **50** samples the loader was capped to. Further, 8.03 (`mode='all'`) and 0.945 (`mode='hierarchical'`) use different loss weights and so are not comparable, and 8.03 is dominated by the broken affinity term (§12.3). Correct statement: *"I don't have a test metric. The pipeline reports a validation loss computed on 50 samples, with a mode mismatch and a broadcasting bug in the affinity term."*

**3. The affinity predictions are implausible on their face.** Training affinities are 0.40–15.00 with mean 6.43; the generated molecules get 0.42–0.70. A head trained with the §12.3 broadcast bug receives a gradient that pushes every prediction toward a batch-wise average of all targets, and is then evaluated on out-of-distribution graphs (random topologies, all-single bonds). Both effects point the same way. There is **no reported R², RMSE, Pearson or Spearman** for the affinity head on held-out pairs — and that is the single cheapest missing experiment in the whole project (one evaluation loop, no retraining).

### What *is* genuinely demonstrated

Be precise about the real claims, because they are defensible:

1. **A working end-to-end pipeline**: raw HF dataset → 12k ESM-2 embeddings with long-sequence chunking → 16.5k multi-scale RDKit decompositions across 10 processes with resumption → joined tensor dataset → PyG batching of three ragged structures → AMP + accumulation training with full-state resume → sampling → RDKit reconstruction → rendered property cards. Every stage runs; the artefacts exist in the repo.
2. **Stable optimisation** of a 4-term multi-task objective across a 3-stage curriculum for 241 epochs without divergence.
3. **Chemically sanitizable output**: the 4 committed molecules pass RDKit sanitisation with reasonable gross properties (MW 154–378, QED 0.43–0.64, 2–3 rings, 0 stereocenters) — though §8 explains how much of that is attributable to the templates and the single-bond reconstruction.
4. **A dataset-level EDA** that surfaced real structure: 9,683 unique scaffolds with a benzene-dominated head, mean 5.5 BRICS fragments, protein-embedding PCA, atom-count vs affinity.

### Metrics that are missing (know the standard list cold)

- **Generative:** validity (RDKit-sanitizable %), uniqueness (@1k/@10k), novelty (not in training set), **Fréchet ChemNet Distance (FCD)**, KL divergence of descriptor distributions, internal diversity (1 − mean pairwise Tanimoto on Morgan fingerprints), scaffold diversity, **synthetic accessibility (Ertl SA score — note the notebook's `SA_Score` is a hand-rolled heuristic, not Ertl's)**, PoseBusters-style validity checks.
- **Conditioning:** does swapping the protein change the output? Affinity of generated molecules under an *independent* oracle (docking: AutoDock Vina/smina/QuickVina; or a separately trained affinity predictor), enrichment vs. a protein-agnostic generator, novelty relative to the target's known actives.
- **Affinity head:** RMSE / MAE / Pearson r / Spearman ρ on held-out pairs, stratified by protein-identity bucket; comparison against the two trivial baselines — predict the global mean, and predict the per-protein mean.
- **Benchmarks a reviewer will name:** GuacaMol, MOSES (distribution learning + goal-directed), CrossDocked2020 / PDBbind for structure-based conditional generation.

Being able to recite this list, and to say which ones you'd run first and why (validity/uniqueness/novelty + the random-protein swap, because they're cheap and they test the central claim), is worth more than any of the numbers above.

---

## 12. Known defects, ranked

Order is by scientific impact. For each: what, evidence, consequence, fix. **Rehearse the top five until you can state them in two sentences each.**

### 12.1 Protein-embedding ID collision (silent data corruption) — CRITICAL
- **What:** chunk-local sequence ids collide across chunks; the merged embedding dict is keyed by those ids.
- **Evidence:** `Loaded full sequence embeddings for 5000 proteins` vs `Loaded mapping for 11795 proteins`.
- **Consequence:** ~58% of proteins conditioned on another protein's embedding. No exception, no shape error.
- **Fix:** content-address the cache (`sha1(sequence)`) or offset ids globally; assert `len(set(map.values())) == len(embeddings)`.

### 12.2 Node features are never standardised, so the forward process never reaches N(0, I) — CRITICAL
- **What:** the diffused vector is `[formal_charge, hybridization(0–6), aromatic, num_h(0–4), degree(1–4), chirality, is_in_ring, valence(1–6)] ‖ atom_embedding(32)`. Magnitudes of 4–6 sit next to unit-variance noise, and the 32 embedding dims have a learned, drifting scale.
- **Consequence:** at t = T, x_T = √ᾱ_T x₀ + √(1−ᾱ_T) ε still carries data signal, so the training marginal ≠ the sampling prior N(0, I). This is a **train/sample distribution mismatch at the very first step of generation** and it alone can explain the ~1.0 losses. It also means the per-dimension SNR differs by an order of magnitude, so one loss weight cannot be right for all dims.
- **Fix:** standardise each continuous feature (dataset-level μ/σ) or scale to [−1, 1] as image diffusion does; better, one-hot the categoricals and use discrete diffusion. For the atom embedding, either freeze its scale (LayerNorm, or `nn.Embedding` with normalised init) or — correctly — don't diffuse a learned embedding at all; diffuse the one-hot type.

### 12.3 Affinity loss is computed with broadcasting — CRITICAL and visible in the logs
- **What:** `F.mse_loss(predicted_affinity, actual_affinity)` with shapes `(B, 1)` and `(B,)`.
- **Evidence:** the notebook's own warning: *"Using a target size (torch.Size([4])) that is different to the input size (torch.Size([4, 1]))… likely lead to incorrect results due to broadcasting."*
- **Consequence:** broadcast to `(B, B)` — every prediction is penalised against **every** target in the batch. The minimiser is "predict the batch mean", which is exactly the collapsed 0.4–0.7 predictions observed. It also inflates the `mode='all'` loss (the 8.03).
- **Fix:** `F.mse_loss(predicted_affinity.squeeze(-1), actual_affinity)` — and add a shape assertion. Generally: never rely on broadcasting in a loss; assert equality.

### 12.4 Generated topology and elements come from hand-written templates, and the denoised features are discarded — CRITICAL (methodological)
- **What:** see §8 `sample()`. `robust_convert_graph_to_mol` reads only `atom_types` and `edge_index`; `x` (the model's output) is unused; bond orders are forced to SINGLE; `atom_embeddings` extracted during sampling are never decoded back to types.
- **Consequence:** the model does not generate molecules. Any apparent quality is attributable to the ring/chain templates, the element priors, and the conservative single-bond reconstruction.
- **Fix:** make node types and edge types the diffusion variables (D3PM/DiGress-style), diffuse a dense adjacency with an atom-count prior, and decode by argmax; at minimum, decode atom types by nearest-neighbour in the embedding table and map the denoised bond-type channel back to RDKit bond orders.

### 12.5 Padding is not masked in the protein encoder — HIGH
- **What:** `collate` right-pads protein embeddings with zeros; `forward` calls `self.protein_encoder(..., padding_mask=None)`. The attention layers attend over pad positions and the mean pooling averages them in.
- **Consequence:** a short protein batched with a long one gets a representation diluted toward zero — so **the conditioning vector depends on the other samples in the batch**. Non-deterministic conditioning; worse at inference where `num_samples` repeats are the batch.
- **Fix:** build a boolean length mask in `collate`, pass it as `key_padding_mask`, and use the masked-mean branch that is *already written* in `ProteinEncoder` but never reached. (Note also the dead `attn_mask = padding_mask.repeat(8,1,1)` line — leftover, unused.)

### 12.6 Only the first 500 training samples are ever seen — HIGH
- **What:** `self.data = self.data[:max_samples]` with `max_samples_per_epoch=500`, applied once at construction, not resampled per epoch.
- **Consequence:** 500 of 15,975 training pairs (3%) and 50 of 1,998 val pairs. 241 epochs over the *same* 500 samples. Reported losses describe fitting a tiny fixed subset; generalisation is unmeasured. Related: `max_samples_per_epoch=None` crashes `create_data_loaders` (`None * 0.1`).
- **Fix:** sample a fresh random subset each epoch (`torch.utils.data.RandomSampler(num_samples=…)` or `SubsetRandomSampler` re-drawn per epoch), and guard the `None` path.

### 12.7 The reverse-diffusion update is mathematically wrong — HIGH
- **What (code):**
  ```python
  posterior_mean = ( torch.sqrt(alpha_bar_next) * beta / (1 - alpha_bar) * x_full
                   + torch.sqrt(alpha) * (1 - alpha_bar_next) / (1 - alpha_bar)
                     * (x_full - noise_pred * torch.sqrt(1 - alpha_bar) / torch.sqrt(alpha_bar)) )
  ```
  Compare with §9: the β_t coefficient must multiply **x̂₀**, and the α_t coefficient must multiply **x_t**. Here they are **swapped** — the x̂₀ coefficient is applied to x_t and vice-versa.
- **Also:** the injected noise is `torch.randn_like(...) * 0.5` ("reduced noise") instead of √β̃_t z, so the stochasticity doesn't match the posterior variance either; and x̂₀ is recomputed inline rather than clipped to a valid data range (standard practice is to clamp x̂₀, not x_t).
- **Consequence:** the sampler does not implement the reverse process it was trained for. Combined with 12.2 and 12.4, this is why sampling "works" only in the sense of not crashing.
- **Fix:** implement the posterior exactly, with the t=1 sanity check from §9; add a unit test that running the sampler on a model whose ε̂ ≡ true ε recovers x₀ from x_T.

### 12.8 Dummy padding fragments contribute to the loss — MEDIUM
- **What:** the level-2 guard is `if graph.x.size(0) > 1`, but `graph` is a *Batch* of fragment-slot-k across the whole mini-batch, so it almost always has >1 node — the guard never excludes the 1-node dummy graphs inserted by the collate. They are diffused, denoised, and included in both the MSE and `extract_fragment_features`' mean.
- **Consequence:** the model is trained to denoise meaningless all-zero graphs, and fragment conditioning is diluted by them; the dilution depends on the most-fragmented molecule in the batch.
- **Fix:** flatten fragments into one Batch with a `mol_id` vector (§7), or carry a `is_real` node mask and use it in both the loss and the pooling.

### 12.9 Empty Murcko scaffolds become a zero node with a self-loop — MEDIUM
- **What:** acyclic molecules have no Murcko scaffold; `_process_graph_data` returns `x=zeros(1,8)`, `edge_index=[[0],[0]]`.
- **Consequence:** level-1 "scaffold" is a meaningless single atom for an unknown fraction of the data (never measured — *measuring it is a 3-line pandas job and I should have*), and `scatter_mean` over it yields near-constant conditioning.
- **Fix:** count them; either exclude acyclic ligands from the scaffold level, or fall back to the largest BRICS fragment / a generic framework, and add an `has_scaffold` flag to the conditioning.

### 12.10 Edge features at sampling time are out of distribution — MEDIUM
- **What:** training `edge_attr = [bond_type_enum, is_conjugated, is_aromatic, is_in_ring, stereo]` where `bond_type_enum ∈ {1, 2, 3, 12}`. Sampling constructs `[1,0,0,0,0]` for single (coincidentally consistent), but `[0,1,0,0,0]` for double — which the trained model reads as *"bond_type = 0, is_conjugated = 1"* — and `[0,0,0,1,0]` for aromatic, read as *"bond_type = 0, in_ring = 1"*.
- **Consequence:** the denoiser receives edge features it never saw during training; conditioning on bond chemistry is effectively scrambled at sampling time.
- **Fix:** one encoding, defined once, shared by both paths (ideally one-hot bond type + binary flags), with a round-trip unit test.

### 12.11 Random pair split ⇒ protein and ligand leakage — MEDIUM (invalidates generalisation claims)
- Fix: protein-cluster split (MMseqs2/CD-HIT at 30–40% identity) and scaffold split; report both.

### 12.12 `ProteinEncoder.forward` as committed cannot run — HIGH (reproducibility)
- **What:** the attention branch computes `pooled_output` and then `return level1_cond, level2_cond, level3_cond` — **those three names are never assigned**; the `level{1,2,3}_output` heads are built in `__init__` but never called. The else-branch calls `Print(...)` (capital P, NameError).
- **Consequence:** running the committed notebook top-to-bottom raises `NameError` in the first forward pass. The stored outputs prove a *different, working* version existed in the live kernel and was overwritten before commit. **The committed notebook is not reproducible.**
- **Fix:** restore the three head calls (`levelN_cond = self.levelN_output(pooled_output)`), and more importantly move the model out of the notebook into a package with an import-time smoke test. This defect is also the reason to answer "can you share the code?" with "yes, with the caveat that the committed cell for the protein encoder is a truncated edit — here's the corrected version."

### 12.13 Smaller items (good to have on hand)
- `requirements.txt` pins `fair-esm` and `pytorch-lightning` that are never used (ESM is loaded via `transformers`); `rdkit` is unpinned. Dead dependencies are a supply-chain and reproducibility smell.
- No grad clipping (§10). No warmup. No EMA of weights — standard and cheap in diffusion, usually worth a visible quality jump.
- `plot_learning_curves` guards on `hasattr(self.scheduler, 'param_groups')`, which is False for `ReduceLROnPlateau` (it holds `.optimizer`), so the LR panel never renders. Harmless, but it means LR was never actually visualised.
- `torch.tensor(0.0, requires_grad=True)` placeholders to keep `.backward()` safe when a level is inactive — works, but a cleaner pattern is to accumulate a list of active losses and sum.
- `max_atoms` filter is applied before the split, so split proportions are over the filtered set (fine, but worth knowing: 19,970 of 19,989 survive).
- Notebook-only delivery: no tests, no CI, no module structure, no config file (hyperparameters are literals in cells), 480 KB of JSON with embedded output images in git.
- The "Synthesizability" and "Stability" scores in the rendered cards are **hand-invented heuristics**, not literature scores. The images say "Synthesizability: 3.2/10" with no validation behind it. If asked, say plainly: a placeholder, to be replaced by Ertl's SA score (`sascorer.py`) and a real reactive-group alert set.
- `'neg_log10_affinity_M' in row` works on a pandas Series (checks the index) — correct, but subtle; worth knowing in case they probe it.

---

## 13. Technical Q&A bank

Answers are compressed; expand with §§5–12 as needed.

### A. Project framing (10)

1. **"Explain your project in one sentence."** A hierarchical, protein-sequence-conditioned graph diffusion model that generates candidate ligands at three structural scales — scaffold, fragments, full molecule — with a joint affinity-prediction head.
2. **"Why multi-scale? Why not just generate the molecule?"** Three reasons: it mirrors medicinal-chemistry workflow (scaffold hopping vs. R-group optimisation are different decisions); it factorises a hard joint distribution into conditionals, which is the same argument as coarse-to-fine generation in images; and it gives *controllability* — you can fix a scaffold and resample substituents, which is exactly what a chemist wants. The cost is error propagation and three times the machinery.
3. **"Did it work?"** It runs end to end and optimises stably, but I can't claim generative quality: the losses sit at the trivial ε-baseline of 1.0, there's no docking or baseline comparison, and I found a sampling-path defect where the topology comes from templates rather than the model. I'd call it a validated *pipeline* and an unvalidated *method*.
4. **"What are you most proud of?"** The data pipeline's fault tolerance — per-SMILES resumption, chunked ESM caching, failure provenance tags — and the decision to embed atomic number rather than treat it as a scalar. And, honestly, the post-hoc audit in §12: I'd rather be the person who found the ID collision than the person who shipped the plot.
5. **"What would you do differently?"** §15, in priority order: fix the ID collision, standardise features, fix the affinity loss shape, then move to discrete diffusion over node *and* edge types so the model actually generates topology.
6. **"How long did it take / how did you scope it?"** Scoped to fit shared-cluster GPU time: 20k pairs rather than 200k, batch 4 with accumulation, 500 samples/epoch. Those were resource decisions; the 500-sample cap was meant as a smoke-test and should have been removed before the 241-epoch run — that's on me.
7. **"Was this solo?"** Single-author repo (two commits, one notebook). Say so plainly, and describe the review you *didn't* get as the reason several defects survived — which is an argument for the code review / CI you'd want in a team.
8. **"Why a notebook?"** Fast iteration on a research question with heavy interactive EDA. In hindsight it cost me reproducibility (§12.12): cells overwrote class definitions, and the committed state doesn't run. I'd keep a notebook for EDA only and move models/data/training into a package with a config system and a smoke test.
9. **"Who is the user?"** Nobody yet. If it worked, a computational chemist generating target-focused virtual libraries for downstream docking — which means the real deliverable is *ranked, synthesizable candidates with uncertainty*, not raw SMILES.
10. **"What's the business/scientific case?"** Lead discovery cost and cycle time. But be careful and honest: the credible near-term contribution of a model like this is *library enrichment* (propose 10⁴ target-biased candidates to be filtered by docking and chemist review), not "designing a drug."

### B. Data and preprocessing (15)

11. **"Why 20k pairs?"** The loader returned 20k for my slice; the ESM + RDKit preprocessing cost (12k transformer passes, 16.5k conformer generations) and shared-GPU time bounded scaling. The filename says 200k — a naming bug I should have fixed.
12. **"Would more data help?"** Almost certainly, but it's not the binding constraint: I was training on 500 samples (§12.6). Fix that first, then scale. Scaling 10× also changes the storage design (§7) — the single-pickle approach won't survive it.
13. **"How did you handle proteins longer than ESM's context?"** 1020-residue windows, 128-residue overlap, average the overlaps per position. Alternatives: truncate (loses pocket), hierarchical window pooling, or a long-context PLM.
14. **"Why overlap 128?"** Enough context for the ~dozens-of-residues scale over which local structure is determined, cheap at 892 stride (≈13% overhead). Not tuned — I'd ablate it.
15. **"Is averaging overlapping embeddings principled?"** It's a pragmatic smoothing. Each position's two embeddings come from different contexts, so the average is not any single model's representation. A cleaner option: take each position's embedding from the window where it is most central.
16. **"Why not fine-tune ESM?"** Memory and time (650M params, no gradient budget on a shared node), and overfitting risk with 11.8k proteins. The standard middle ground is LoRA/adapters on the top blocks, or at minimum a learned scalar mix over layers.
17. **"Why keep per-residue embeddings if you mean-pool anyway?"** Caught — that's an inconsistency (§12.5). Keeping them is the right *storage* decision because it preserves the option of cross-attention; mean-pooling throws that option away. Cross-attention from ligand nodes to residues is the v2.
18. **"How do you know the embeddings are correct?"** I checked shapes, length distributions, and a PCA of mean-pooled embeddings. That was insufficient: the ID collision (§12.1) passes all three checks. The test that would have caught it is an identity round-trip — pick 100 random sequences, recompute the embedding, assert it matches the cached one.
19. **"Why Murcko for level 1?"** It's the standard, canonical definition of a molecular core, it's deterministic, and it's what "scaffold hopping" and scaffold splits are defined on.
20. **"Why BRICS and not RECAP or simple bond cuts?"** BRICS cuts at bonds that correspond to plausible synthetic transformations and labels the attachment-point chemistry, so fragments are buildable units rather than arbitrary subgraphs. RECAP is the older, simpler rule set; naive cuts give chemically meaningless pieces.
21. **"Mean 5.5 fragments per molecule — what does the distribution look like?"** I have the mean and a plotted histogram (`fragment_counts.png`); the tail matters for the collate design (§7), because the per-slot batching cost is set by the max.
22. **"Why 150 atoms as the cutoff?"** Memory and the long tail (max 177 in the data; only 11 molecules exceeded 150). Drug-like ligands are typically 20–50 heavy atoms, so 150 is already generous.
23. **"How do you handle invalid SMILES / RDKit failures?"** Skip with a log, and record provenance in `force_field_type`. The risk is silent attrition, which is why the join step counts missing/filtered/truncated.
24. **"You compute MMFF energies — what for?"** They were intended as physics-informed features. They are never fed to the model (§6). Candid answer, plus the three ways I'd wire them in.
25. **"Your affinity label ranges 0.4 to 15. Is that physical?"** pK 15 is femtomolar, at the edge of plausibility; 0.4 is non-binding. I'd clip to ~2–12, inspect the tails for assay-type mixing (K_d vs K_i vs IC₅₀ are not interchangeable), and standardise the target before regression.

### C. Diffusion (20)

26. **"Derive the closed-form forward marginal."** §9.
27. **"Write the posterior mean."** §9 — and state the t=1 sanity check. (If asked to check the code, this is §12.7.)
28. **"Why predict ε rather than x₀?"** Unit-scale target at all t ⇒ well-conditioned loss. Mention v-prediction.
29. **"Why cosine over linear?"** Linear destroys signal too early; most of the chain is wasted on pure noise. Cosine keeps ᾱ high longer — matters more for small objects.
30. **"What does ᾱ_T → 0 buy you?"** The sampling prior becomes data-independent N(0, I). **This is what my unnormalised features break** (§12.2).
31. **"Your loss is ~0.95. Good?"** No — 1.0 is the predict-zero baseline. Report the loss stratified by t.
32. **"How would you speed up 1000-step sampling?"** DDIM strided sampling first (free, 10–50× fewer steps), then DPM-Solver, then distillation/consistency models. Also: batch the fragment chains instead of looping them.
33. **"Is Gaussian diffusion appropriate for molecular graphs?"** No, and this is the core methodological compromise. Discrete diffusion (D3PM, DiGress) for types and bonds; EDM/EGNN if you go 3D. §9.
34. **"How do you diffuse edges?"** I don't — topology is fixed/supplied, which is defect §12.4. The right approach diffuses a dense N×N edge-type tensor jointly with node types.
35. **"How is the number of atoms decided?"** Hand-coded priors at sampling time. Correct approaches: learn p(N) from data and sample N first (DiGress/EDM do this), or use a fixed-size canvas with a "no-atom" absorbing type.
36. **"What's classifier-free guidance and do you use it?"** Train with the condition dropped a fraction of the time, then sample with ε̂ = ε̂_uncond + w(ε̂_cond − ε̂_uncond) to trade diversity for conditioning strength. I don't use it — which also means I have no knob for "how strongly should this molecule respect the target?" Adding it is cheap and would also give me the unconditional model needed for the random-protein control experiment.
37. **"Could you do affinity-guided sampling?"** Yes — that's the interesting extension: use the differentiable affinity head as a classifier guidance term, ∇_x log p(y|x), added to the score at each step (or target-affinity conditioning à la "generate me a pK_d 9 ligand"). The `sample()` signature already takes `target_affinity`; it is **accepted and never used** — another honest gap, and a clean next step.
38. **"Why separate diffusion processes per level rather than one?"** Different object sizes and statistics, and it enables stage-wise training and scaffold-constrained resampling. Cost: three sets of parameters and no shared noise schedule coupling.
39. **"Independent timesteps per level — why?"** Each level is its own process. A coupled schedule (coarse levels denoise first) would be more principled and is what hierarchical/cascaded image diffusion does.
40. **"What's the prior at sampling for level 2 and 3?"** `randn` features on a template topology, with scaffold/fragment conditioning concatenated — not a clean prior, see §12.2/12.4.
41. **"Why clamp to [−5, 5]?"** Empirical stability hack to stop divergence in the chain. Principled version: clamp the *predicted x̂₀* to the valid data range (standard "dynamic/static thresholding"), not the running x_t.
42. **"Why `noise * 0.5`?"** An undocumented variance reduction. It is not the posterior σ_t, so the sampler is off-distribution; I'd replace it with √β̃_t (§12.7).
43. **"What is `register_buffer` for?"** Non-trainable tensors that must follow `.to(device)` and appear in `state_dict`.
44. **"How would you validate your sampler independent of model quality?"** Unit test with an oracle denoiser: if ε̂ returns the true ε used in the forward pass, the reverse chain must reconstruct x₀ to numerical tolerance. That single test catches §12.7.
45. **"Can diffusion be conditioned on a continuous property?"** Yes — concatenate/FiLM the property embedding (as with t), optionally with CFG; or guide with a differentiable predictor at sampling time.

### D. GNNs and architecture (20)

46. **"How does message passing work?"** For each edge, compute a message from (source, target, edge) features; aggregate per node with a permutation-invariant operator (sum/mean/max/attention); update the node state. Stack L layers ⇒ each node sees its L-hop neighbourhood.
47. **"GATv2 vs GAT?"** Static vs dynamic attention — §8.
48. **"Why not GCN?"** Isotropic (degree-normalised mean), no edge features, no learned neighbour weighting.
49. **"Why not GIN?"** Maximally expressive under 1-WL for isomorphism, but sum aggregation without edge-feature attention; worse fit for bond-chemistry-conditioned denoising. Good baseline to ablate against.
50. **"3 layers — why? What's the receptive field?"** 3-hop. Molecules are small (mean 34 atoms, diameter ~10–15 bonds), so 3 layers see a local chemical environment but **not** the whole molecule — which is a real limitation for ring/aromaticity reasoning. Deeper GNNs over-smooth; the standard fixes are virtual nodes, graph transformers with global attention, or explicit structural features (cycle counts, Laplacian eigenvectors) — the last is exactly what DiGress adds.
51. **"What's over-smoothing?"** Repeated neighbourhood averaging drives node representations toward a common value, so depth stops helping. Mitigations: residuals, normalisation, jumping knowledge, PairNorm.
52. **"How does PyG batch graphs of different sizes?"** Block-diagonal concatenation: node features stacked along dim 0, `edge_index` offset by cumulative node counts, a `batch` vector for node→graph. Pooling = scatter by `batch`. No padding.
53. **"What does `scatter_mean` do here?"** Per-graph mean pooling over nodes → one `cond_dim` vector per molecule; permutation-invariant.
54. **"Why mean and not sum/max/attention pooling?"** Mean is size-invariant (sum correlates with atom count, which for affinity is partly real signal and partly spurious). Set2Set or attention pooling would be strictly more expressive; worth an ablation.
55. **"How does the per-graph condition reach the nodes?"** `h_cond[batch]` — the gather idiom. Be able to write it.
56. **"Why is concatenation a weak conditioning mechanism, and what's better?"** It only shifts the input; FiLM/AdaLN modulate every layer multiplicatively; cross-attention lets each atom query the protein residues. §8.
57. **"Why an atom-type embedding instead of a feature?"** Atomic number is categorical, not ordinal. §6. Then volunteer that hybridization/bond_type should get the same treatment.
58. **"What's wrong with `bond_type` as a scalar?"** RDKit's AROMATIC enum is 12, so aromatic reads as 12× single. §6/§12.10.
59. **"Is your model permutation-equivariant?"** The GNN is; `scatter_mean` is invariant. The *sampler* isn't (template index conventions). §9.
60. **"Where does the protein's 3D structure enter?"** It doesn't — sequence-only. That's the honest, major scope limit vs. structure-based methods.
61. **"How would you add pocket structure?"** Pocket residue graph or point cloud → E(3)-equivariant encoder (EGNN) → cross-attention to ligand nodes; or go fully 3D with joint coordinate diffusion (EDM/TargetDiff style), which then requires equivariance throughout.
62. **"Parameter count?"** Not recorded in the notebook — don't guess a number. Say: "dominated by three denoisers with 3 GATv2 blocks at hidden 256, plus a 3-layer 256-d attention encoder; I'd report `sum(p.numel() for p in model.parameters())` — it's the one line I should have logged." Frozen ESM-2 650M is not part of the trainable count.
63. **"Why `hidden_dim // heads`?"** So concatenated multi-head output returns to `hidden_dim` and the residual add stays shape-valid. Requires divisibility — a latent fragility.
64. **"What does LayerNorm do here and why not BatchNorm?"** Normalises per node over features — independent of batch composition and of graph size, so it works with batch 4 and variable node counts. BatchNorm's batch statistics are unreliable at batch 4 and leak across graphs.
65. **"Where's the dropout and does it make sense in a denoiser?"** 0.1 in every projection and in GATv2 attention. Mild regularisation; note that heavy dropout in a diffusion denoiser can hurt, since the task is already noise-robust — EMA + weight decay is the more standard combination.

### E. Training, systems, MLOps (15)

66. **"Explain AMP and GradScaler."** §10.
67. **"Explain gradient accumulation and the loss division."** §10.
68. **"Why batch size 4?"** O(L²) protein attention at L up to 1000, three graph denoisers, and a shared GPU. Accumulation recovered effective batch 16.
69. **"What's in your checkpoint and why does it matter?"** model + optimizer + scheduler + scaler + history. Resuming without optimizer moments or scaler state causes a loss spike / silently disabled loss scaling.
70. **"How does resume work and is it correct?"** It restores all state, but the hierarchical stage bookkeeping assumes 2 epochs/stage while the table says 10 (§12.13/§10) — a resume bug I'd fix with an explicit `stage_idx` in the checkpoint rather than arithmetic on epoch counts.
71. **"How would you scale to multi-GPU?"** DDP with `DistributedSampler`; for graph batches, PyG's `DataLoader` plus careful seeding of the collate; gradient accumulation interacts with `no_sync()` for all but the last micro-step. Watch out: my per-slot fragment batching makes the number of forward passes data-dependent, which causes DDP desync — another reason to flatten fragments (§7).
72. **"Where's the bottleneck?"** For preprocessing: CPU-bound RDKit (hence 10 processes). For training: protein self-attention and three sequential level passes. For sampling: 4 × 1000 sequential network calls per molecule. I'd profile with `torch.profiler` before optimising — which I did not do, and should have.
73. **"How do you prevent OOM?"** Batch 4, accumulation, AMP, 1000-residue truncation, 150-atom filter, chunked preprocessing, `malloc_trim`, `gc.collect`, `torch.cuda.empty_cache()` between chunks.
74. **"Why `multiprocessing` and not threads?"** GIL; RDKit work is CPU-bound in Python-wrapped C++.
75. **"How do you make a long preprocessing job restartable?"** Idempotent stages keyed by content (skip existing chunk files; SMILES→batch map; fresh batch indices on resume). This is the part of the project I'd defend in any systems interview.
76. **"What's missing from your training loop?"** Grad clipping, LR warmup, EMA weights, early stopping, a proper experiment tracker (W&B/MLflow — I logged to matplotlib PNGs), and a config system instead of literals in cells.
77. **"How would you productionise this?"** Package the code; config via Hydra/YAML; DVC or an object store for data artefacts with content hashes; unit tests for the sampler identity, the collate shapes, and the posterior formula; CI running a 1-step training smoke test; model registry; batch inference as a job, not a notebook.
78. **"How would you test an ML pipeline like this?"** Shape/invariant assertions (already partly present); a collate test with ragged inputs including the empty-graph case; the oracle-denoiser sampler test (§C44); an overfit-one-batch test (loss → ~0) — which, had I run it, would have exposed §12.2 and §12.3 immediately; and a data-integrity test (embedding round-trip) that would have caught §12.1.
79. **"What does 'overfit a single batch' tell you?"** That the optimisation path, loss, and shapes are wired correctly. If a model can't drive a single batch's loss to near zero, no amount of data will help. It is the cheapest high-value test in deep learning and it is the test I most regret not running.
80. **"What monitoring would you add?"** Loss by timestep bucket and by level; gradient norms per module; the affinity head's R²/Spearman on a fixed validation slice; per-epoch validity/uniqueness/novelty of 100 generated molecules; and the random-protein control as a tracked metric.

### F. Chemistry and evaluation (12)

81. **"What is pK_d and why regress it?"** §3 — it's linear in binding free energy.
82. **"What's QED / Lipinski / TPSA / logP?"** §3.
83. **"Is your generated-molecule property profile reasonable?"** MW 154–378, 2–3 rings, QED 0.43–0.64, LogP −1.0 to 0.1, 0 stereocenters. Gross properties are drug-like-ish, but LogP near or below 0 across all four and zero stereocenters everywhere are tells of the single-bond, template-driven reconstruction, not of learned chemistry.
84. **"How do you measure generative quality?"** §11's list: validity/uniqueness/novelty, FCD, internal diversity, SA score, then target-specific docking.
85. **"How would you validate that conditioning works?"** (a) Swap in a random protein and measure the change in output distribution; (b) compare docking scores against the intended target vs. decoy targets; (c) compare to an unconditional generator; (d) check whether generated molecules resemble the target's known actives more than chance (Tanimoto to known ligands).
86. **"What's Tanimoto similarity?"** |A∩B|/|A∪B| over fingerprint bits (e.g. Morgan/ECFP4) — the standard chemical similarity measure; 1 − mean pairwise Tanimoto is the usual internal-diversity metric.
87. **"What's an SA score?"** Ertl & Schuffenhauer's synthetic accessibility: fragment-contribution score (1 easy – 10 hard) from a frequency analysis of PubChem fragments plus complexity penalties. **My `SA_Score` is not this** — it's a hand-rolled heuristic (§12.13).
88. **"Could these molecules be made?"** Unknown, and I shouldn't imply otherwise: no retrosynthesis check (AiZynthFinder/ASKCOS), and a template-assembled graph with all-single bonds is likely to contain strained or implausible motifs.
89. **"What's docking and why is it the right external oracle?"** Physics/empirical scoring of a ligand pose in a pocket (AutoDock Vina, smina, Glide). It's independent of my model, so it can't be gamed by my own learned head — though docking scores correlate only modestly with measured affinity, so it's an enrichment signal, not truth.
90. **"What are PAINS / structural alerts?"** Pan-assay interference compounds — substructures that produce false positives in assays. A standard filter (RDKit FilterCatalog) that any generative output should pass before anyone looks at it. Not implemented here.
91. **"Why does scaffold diversity matter?"** A generator that emits 10,000 decorations of one benzene core has no practical value. The data's own scaffold head is benzene-dominated (1,068/9,683), so the model has every incentive to collapse there — making scaffold-level diversity metrics mandatory.
92. **"If the affinity head worked, what could you do with it?"** Rank candidates, guide sampling (classifier guidance), condition on a target affinity, and run active learning: generate → dock/assay a few → retrain. That loop is the actually valuable product.

### G. Behavioural-technical hybrids (8)

93. **"Tell me about the hardest bug."** The ESM ID collision (§12.1) — hardest because nothing failed. See the §16 STAR version.
94. **"Tell me about a time you were wrong."** The README's "physics-informed features" claim, which the code doesn't support (§6). I wrote the claim from intent, not from the implementation. I now hold the rule: documentation claims must be traceable to a line of code or they get moved to a "planned" section.
95. **"How do you know when to stop engineering and start measuring?"** I got this wrong here: I built three diffusion levels before establishing a single-level baseline or an overfit-one-batch test. The lesson is a working, measured v0 (one level, validity + uniqueness, a trivial baseline to beat) before adding hierarchy.
96. **"How do you handle a result you don't believe?"** The affinity predictions of 0.4–0.7 against a training mean of 6.4 were the signal. The right move — which I took only during the audit, not during the run — is to treat distributional implausibility as a bug report and trace it, which led straight to the broadcast MSE.
97. **"What would you tell a junior engineer starting this project?"** Write the data-integrity assertion before the model. Overfit one batch before training for 241 epochs. And pick your evaluation metric before your architecture, or you'll build something you can't grade.
98. **"How do you prioritise a list of 13 defects?"** By whether they invalidate a claim: data corruption and train/sample mismatch first (they make every number meaningless), then loss correctness, then the methodological gap, then efficiency and hygiene. §12 is ordered exactly that way.
99. **"If I gave you one week on this, what would you do?"** Day 1: fix §12.1/12.3/12.5 and add the overfit-one-batch + sampler-identity tests. Day 2: standardise features, re-run level 1 only, and look at loss-by-timestep. Days 3–4: replace the template sampler with a learned atom-count prior plus discrete node/edge diffusion for level 1. Day 5: validity/uniqueness/novelty + random-protein control. Then decide whether the hierarchy earns its complexity.
100. **"Convince me this project was worth doing."** It taught me the whole stack — PLM featurisation, chemical decomposition, graph batching, diffusion math, HPC-scale fault tolerance — and it produced a defect list that I can generalise: content-address your caches, assert shapes in losses, normalise your diffusion inputs, and never let a sampler diverge from the process you trained. That's transferable; a lucky metric wouldn't have been.

---

## 14. Comparison to the literature

Know the landscape and where this sits. **[Established Knowledge]** for what each method does; I quote **no** benchmark numbers, because I haven't verified them — say "I'd need to verify the exact figures" if pressed.

| Family | Representative work | What it does | Relation to this project |
|---|---|---|---|
| 2D graph autoregressive | GraphAF, GCPN, MolecularRNN | Build the graph atom/bond by atom/bond | Alternative to diffusion; easy valency enforcement during construction |
| Fragment/junction-tree VAE | JT-VAE, HierVAE | Generate over a tree of chemical substructures, then assemble | **Closest in spirit to my multi-scale idea** — and the honest comparison: JT-VAE guarantees validity by construction because it only assembles valid substructures; my assembly step does not |
| SMILES language models | ChemBERTa-style, REINVENT | Token-level generation, RL for goal-directed design | Strong, simple baselines I should have compared against |
| Discrete graph diffusion | **D3PM, DiGress** | Categorical diffusion over node *and* edge types, graph transformer, auxiliary structural features | **The method my model should have been** (§12.4) |
| 3D molecule diffusion | **EDM**, GeoDiff, MolDiff | E(3)-equivariant diffusion over coordinates + types | Needs equivariance; MolDiff specifically targets atom–bond inconsistency |
| Structure-based (pocket-conditioned) | **TargetDiff, DiffSBDD, Pocket2Mol, LiGAN** | Generate ligands inside a given 3D pocket | The real competition for "protein-conditional generation"; they use structure, I use sequence only |
| Docking / pose prediction | DiffDock | Diffusion over ligand pose, not identity | Complementary: a generator's output needs a docking oracle |
| Protein language models | **ESM-2**, ProtTrans, ProtBert | Sequence representations | My conditioning backbone |
| Co-folding / complex structure | AlphaFold-Multimer, AF3-class models, RoseTTAFold-AA | Predict complex structure incl. ligands | Changes the field's premise: if you can co-fold, you can condition on predicted structure instead of sequence — the strongest argument for a v2 that drops sequence-only conditioning |
| Affinity prediction | DeepDTA, GraphDTA, and PLM-based regressors | Predict affinity from sequence + SMILES | My affinity head is a weak instance of this; these are the baselines it must beat |

**How to use this table in the room:** when asked "how does your work compare to X," never claim superiority. Say: "Same problem family, different conditioning and different noise space. DiGress diffuses the discrete graph, which is the correct formulation and which I'd adopt; TargetDiff conditions on the 3D pocket, which is a strictly richer signal than my sequence embedding. What's distinctive in mine is the explicit three-scale factorisation with per-scale conditioning vectors — but I haven't demonstrated that the factorisation pays, and that's the experiment that would make it publishable: an ablation of hierarchical vs. single-level under identical budgets."

---

## 15. v2 roadmap

Priority-ordered. Each item says *why it's first* — that reasoning is what's being assessed.

**Tier 0 — correctness (nothing can be claimed until these land)**
1. Content-address the protein-embedding cache; add `assert len(set(map.values())) == len(embeddings)` and a 100-sequence re-embedding round-trip test. *(§12.1 — corrupts the conditioning signal.)*
2. `F.mse_loss(pred.squeeze(-1), target)` plus a shape assertion in every loss. *(§12.3 — the affinity head is currently unlearnable.)*
3. Standardise all continuous node/edge features (dataset μ/σ, persisted with the checkpoint); verify empirically that q(x_T|x₀) ≈ N(0, I) by measuring the mean/variance of x_T. *(§12.2 — the prior mismatch invalidates sampling.)*
4. Rewrite the reverse posterior exactly; add the oracle-denoiser round-trip test. *(§12.7.)*
5. Mask protein padding (`key_padding_mask` + masked mean). *(§12.5 — conditioning currently depends on batch composition.)*
6. Remove the 500-sample cap; resample per epoch. *(§12.6.)*
7. Add the two tests that would have caught most of the above: overfit-one-batch, and a collate test with ragged/empty inputs.

**Tier 1 — make it a real generative model**
8. Replace Gaussian-on-integers with **discrete diffusion over node types and edge types** (D3PM absorbing or uniform kernels), over a dense N×N edge tensor, with a learned p(N) atom-count prior. Decode by argmax; enforce valency as a post-hoc projection or via a masked transition matrix. *(§12.4 — this is what makes the model, rather than the templates, generate the molecule.)*
9. Delete the hand-written scaffold/fragment templates and the random attachment logic.
10. One shared edge/node encoding used by both training and sampling, with a round-trip test. *(§12.10.)*
11. DDIM sampling (strided, 50 steps) and batch the fragment chains. *(~4000 → ~50 network calls per molecule.)*

**Tier 2 — make the conditioning real and measurable**
12. Cross-attention from ligand nodes to **per-residue** protein embeddings (replacing mean pooling), optionally restricted to predicted pocket residues.
13. Classifier-free guidance (condition dropout during training, guidance weight at sampling) — which also yields the unconditional model needed for controls.
14. Affinity-guided sampling using the (now-correct) affinity head, and target-affinity conditioning — wiring up the `target_affinity` argument that `sample()` already accepts and ignores.
15. FiLM/AdaLN conditioning per block, and time injection at every block rather than once at the input.

**Tier 3 — evaluation, which should honestly have been Tier 0**
16. Protein-cluster split (MMseqs2 at 30% identity) + ligand scaffold split; re-report everything.
17. Generative metrics: validity, uniqueness@1k, novelty, FCD, internal diversity, scaffold diversity, Ertl SA, PAINS/alert filters.
18. The random-protein swap control, and docking (smina/Vina) of generated molecules against intended vs. decoy targets.
19. Affinity head: RMSE/Pearson/Spearman vs. three baselines — global mean, per-protein mean, and a simple ECFP+ESM-mean-pool gradient-boosted regressor. If it can't beat the per-protein mean, the head isn't learning chemistry.
20. **The ablation that justifies the paper's premise:** hierarchical (3-level) vs. single-level diffusion at equal parameter and compute budget, measured on the Tier-3 metrics. If the hierarchy doesn't win, the honest conclusion is to publish that.

**Tier 4 — engineering**
21. Notebook → package (`data/`, `models/`, `train/`, `eval/`), Hydra config, CLI entry points, smoke-test CI.
22. Artefact storage: dedupe protein embeddings, memory-map or shard (HDF5/LMDB/webdataset), content-hash the artefacts.
23. W&B/MLflow tracking; EMA weights; grad clipping; warmup + cosine LR; parameter-count and FLOPs logging.
24. Drop unused dependencies (`fair-esm`, `pytorch-lightning`); pin `rdkit`.

---

## 16. Behavioural STAR stories

Nine stories, all grounded in this project so you never have to invent. Keep each to 90 seconds; lead with the result sentence if time is short.

### 1. "Hardest bug" / "attention to detail" — the silent ID collision
- **S:** After training, I audited the pipeline end to end rather than just reading the loss curve.
- **T:** Verify that the conditioning signal was actually what I thought it was.
- **A:** I cross-read two log lines that nobody looks at: the combined embedding file reported 5,000 proteins while the sequence→id map reported 11,795. That gap shouldn't exist. I traced it to ids being assigned *locally inside each 5,000-protein chunk*, so merging chunks into one id-keyed dict silently overwrote entries and remapped most proteins onto someone else's embedding.
- **R:** ~58% of samples had been trained with the wrong protein. The fix is one line — content-address the cache by a hash of the sequence — plus an assertion that the number of distinct map values equals the number of cached embeddings. The durable lesson: **a bug that doesn't raise is more dangerous than one that does, and the only defence is an invariant you write down before you need it.** I now add data-integrity assertions as part of writing a cache, not as a response to a failure.

### 2. "Time you failed" / "learned a hard lesson" — building before measuring
- **S:** I designed a three-level hierarchical diffusion model with a joint affinity head.
- **T:** Show that hierarchy helps conditional molecule generation.
- **A:** I built all three levels, the curriculum, and the sampler before establishing a baseline or a single quality metric. Then the numbers told me nothing: the ε-losses sat at ~0.95–1.16, and the trivial "predict zero" baseline is 1.0 — so I couldn't distinguish learning from not learning. I also never ran an overfit-one-batch test, which would have exposed the shape bug in the affinity loss on the first day.
- **R:** No defensible claim about the hierarchy. I reordered how I work: define the metric and the trivial baseline first, overfit one batch second, build the ablation harness third, add architecture last. That ordering is now non-negotiable for me, and I'd apply it on day one of any model work on your team.

### 3. "Dealing with ambiguity" / "making a judgment call" — three-scale representation
- **S:** Generating a 35-atom molecule as one joint object is a hard, high-dimensional problem with no obvious factorisation.
- **T:** Choose a representation.
- **A:** I borrowed the structure from the domain rather than from ML: chemists reason about a scaffold (the core) and substituents (the decoration). I mapped that to RDKit's Bemis–Murcko scaffold and BRICS fragmentation — BRICS specifically because it cuts at synthetically meaningful bonds — and verified the decomposition was non-degenerate on real data: 9,683 unique scaffolds and 5.5 fragments per molecule across 20k pairs.
- **R:** A factorisation that is interpretable and gives a controllability knob chemists actually want (fix the scaffold, resample substituents). I also know its unproven status: the ablation that would justify it is the first experiment in my v2.

### 4. "Engineering under constraints" / "resilience" — the resumable preprocessing pipeline
- **S:** Preprocessing meant ~12k ESM-2 forward passes and ~16.5k 3D conformer generations with force-field minimisation, on a shared university cluster with job time limits and OOM-killed workers.
- **T:** Get through it without losing a day's work to every interruption.
- **A:** Three things. (1) Chunked the ESM stage into 5,000-protein `.npz` files and skipped any chunk already on disk. (2) Built per-SMILES resumption for the RDKit stage: a persistent SMILES→batch map, with new batch indices allocated above the existing maximum so a resume can never overwrite completed work. (3) Diagnosed RSS growth in long-lived workers as glibc heap fragmentation from RDKit allocations and added an explicit `malloc_trim(0)` via `ctypes` after each batch group, wrapped in a try/except for portability.
- **R:** Interruptions cost minutes instead of a full re-run, and workers stopped being killed. This is the part of the project I'd reuse unchanged — idempotent, content-keyed stages are the cheapest reliability win in any data pipeline.

### 5. "Debugging a performance problem" — memory, not speed
- **S:** The model wouldn't train: protein self-attention over up to 1,000 residues at 256-d, three graph denoisers, one shared GPU.
- **T:** Fit the model and still get a usable gradient signal.
- **A:** Layered the standard toolkit deliberately rather than randomly: truncate proteins at 1,000 residues and filter ligands at 150 atoms (data-level, bounded by distribution tails — only 11 molecules and 85 proteins affected); AMP autocast + GradScaler (activation memory and throughput); physical batch 4 with 4-step gradient accumulation and loss division so the effective batch is 16 and the LR stays meaningful; `empty_cache` between preprocessing chunks.
- **R:** Stable training for 241 epochs. What I'd do differently: profile first. I chose those knobs from experience, not from `torch.profiler` output, and I can't tell you which one mattered most — that's a measurement I skipped.

### 6. "Receiving/giving criticism" / "intellectual honesty" — the README claim
- **S:** My README advertises "physics-informed features to enhance molecular validity."
- **T:** Verify my own documentation while preparing to present the work.
- **A:** I traced the feature path and found that the MMFF energies, ring-strain terms, descriptor set, and functional-group counts are computed, stored, and carried through the collate function — and never passed to the network. The claim described my intent, not my code.
- **R:** I corrected the claim and wrote down the three concrete ways to wire those features in (affinity-head input, auxiliary multi-task targets, sampling-time guidance). My rule now: **every claim in a README must point at a line of code, or it moves to a "planned" section.** I'd rather be the person who catches that in my own work than have a reviewer catch it.

### 7. "Communicating technical work to a non-expert"
- **S:** Explaining a conditional graph diffusion model to people outside ML.
- **T:** Convey the idea without jargon.
- **A:** I use two analogies. Diffusion: "take a photo, add static until it's noise, then train a network to remove one step of static at a time; run it from pure static and you get a new photo." Conditioning and hierarchy: "the protein is the lock and we're generating keys — first the blank's shape, then the teeth, then the fine filing." Then I show the generated-molecule property cards, because a picture of a molecule with MW and QED on it lands where a loss curve doesn't.
- **R:** People ask the right follow-up — "how do you know it's a good key?" — which is exactly the evaluation question I want to be asked, and it lets me talk about docking as an independent oracle.

### 8. "Working with limited resources / scoping" — and a mistake inside it
- **S:** Shared-cluster GPU time and a preprocessing cost that scaled with unique proteins and ligands.
- **T:** Get an end-to-end result within budget.
- **A:** I scoped deliberately: 20k pairs instead of 200k, batch 4 + accumulation, and a `max_samples_per_epoch=500` cap to smoke-test the loop quickly.
- **R:** The pipeline ran end to end — but I left the 500-sample cap in place for the full 241-epoch run, so the model only ever saw 3% of the training set, and the same 3% every epoch. That's the kind of mistake that comes from a debugging flag outliving its purpose. My fix for the future is mechanical, not willpower: debug flags get logged loudly in the run header and asserted off in any run longer than N epochs.

### 9. "Taking initiative / ownership" — the self-audit
- **S:** The project "worked": it trained, it generated molecules, there were pictures.
- **T:** Decide whether to present it or interrogate it.
- **A:** I did a line-by-line audit against the math and the logs, and produced a ranked defect list of 13 items with fixes — including the three that invalidate the results (ID collision, unnormalised diffusion inputs, broadcast affinity loss) and one methodological finding I didn't want to be true: that the generated topology comes from hand-written templates and the model's denoised output is discarded at reconstruction time.
- **R:** I can now state exactly what the project does and does not demonstrate, and I have a prioritised plan where every item is justified by a specific failure. **I'd rather bring you a correct account of a flawed prototype than a flattering account of one.**

### Mapping: question → story

| Question | Use story |
|---|---|
| Hardest/most interesting bug | 1 |
| Biggest failure / what you'd do differently | 2, 8 |
| Ambiguous problem, no clear path | 3 |
| Systems/infra or reliability work | 4 |
| Performance/memory debugging | 5 |
| Intellectual honesty / disagreement with yourself | 6, 9 |
| Explaining to non-experts / stakeholders | 7 |
| Prioritising under constraints | 8, 9 |
| Taking ownership beyond the ask | 9 |
| Teamwork/conflict | **Weak spot — this was solo work.** Answer from coursework or another setting, and use 9 to show you self-review in lieu of peer review. Do not invent a teammate. |

### "Tell me about yourself" — a 60-second frame

> "I work at the intersection of ML systems and the life sciences. My most substantial project is a protein-conditional molecule generator: ESM-2 embeddings for the target, a hierarchical graph diffusion model for the ligand, built end to end from a raw Hugging Face dataset through a fault-tolerant HPC preprocessing pipeline to a trained model and rendered generated molecules. What I took from it is as much about rigour as about architecture — I audited my own pipeline afterwards and found a silent data-corruption bug and a train/sample distribution mismatch, which reshaped how I sequence work: metric and trivial baseline first, overfit one batch second, architecture last. I'm looking for a team where that kind of rigour is the norm and where I'll have reviewers who catch what I miss."

### Red-flag questions and how to handle them

- **"So your project doesn't work?"** — "The pipeline works and is reproducible-by-design; the *method* is unvalidated, and I can tell you precisely why in three sentences." Then §12.1/12.2/12.4. Never get defensive; the content is the defence.
- **"Why should I hire someone whose project had this many bugs?"** — "Every real system has them; the difference is whether they're found by the author or the reviewer. I found these in my own code, ranked them by whether they invalidate a claim, and wrote the fix for each. That's the behaviour you want on a code review."
- **"Did you use AI to write this?"** — Answer honestly about whatever is true for you. The defensible position is tool use plus full comprehension: you can derive the diffusion posterior, explain every design choice, and you found the defects. Demonstrate that rather than asserting it.
- **"Is this novel?"** — "The three-scale factorisation with per-scale conditioning is the distinctive piece, but I haven't run the ablation that would show it pays, so I wouldn't call it a contribution yet. The honest framing is a strong engineering exercise and a clear research plan."

---

## 17. Live-coding and whiteboard drills

Practise writing these cold. Each has appeared in interviews for this kind of role.

1. **Forward diffusion in one line.**
   ```python
   def q_sample(x0, t, sqrt_ab, sqrt_1mab, batch):
       eps = torch.randn_like(x0)
       a, b = sqrt_ab[t][batch].unsqueeze(-1), sqrt_1mab[t][batch].unsqueeze(-1)
       return a * x0 + b * eps, eps
   ```
2. **Cosine schedule.** From §9; then plot ᾱ_t for linear vs cosine and explain the difference.
3. **DDPM posterior sampling step**, correctly (the §12.7 fix), including the t=1 check.
4. **Sinusoidal timestep embedding** from scratch (exp-spaced frequencies, sin ‖ cos, odd-dim pad).
5. **Broadcast per-graph vector to nodes:** `cond[batch]`. Then: implement `scatter_mean` using `index_add_` and a count tensor.
6. **PyG-style manual batching:** given a list of `(x, edge_index)`, produce concatenated `x`, offset `edge_index`, and the `batch` vector.
7. **A single message-passing layer** by hand with `index_select` + `scatter_add` (no PyG): messages from `edge_index[0]`, aggregate at `edge_index[1]`.
8. **Collate with two ragged axes:** pad sequences to batch max and return a boolean mask; then use the mask for a masked mean.
9. **SMILES → graph tensors** with RDKit: atoms → features, bonds → both directions of `edge_index`.
10. **Valency-safe graph → RDKit molecule** (`RWMol`, per-element max valence, `UpdatePropertyCache(strict=False)`, `SanitizeMol`) — essentially §8 step 7, which you should be able to justify line by line.
11. **Compute validity / uniqueness / novelty** for a list of generated SMILES against a training set (canonicalise first — this is the classic trap).
12. **Gradient accumulation loop** with AMP, including the tail flush and the loss division.
13. **Overfit-one-batch harness:** loop a single batch until loss < ε, with an assertion and a step cap.
14. **Oracle-denoiser sampler test:** substitute `eps_hat = true_eps` and assert the reverse chain reconstructs x₀.
15. **Tanimoto similarity** from two RDKit Morgan fingerprints, and 1 − mean pairwise Tanimoto for diversity.

---

## 18. Questions to ask

Pick three or four; they signal what you'd be like to work with.

**On the work**
- Where does this team sit on the research↔production axis, and how does that change the bar for evidence before something ships?
- What's the evaluation stack — is there an in-house oracle (docking, assay data, a held-out benchmark), and who owns it?
- For generative chemistry: how do you handle the synthesizability gap between a proposed molecule and something a chemist will actually make?
- How are splits defined here — protein identity clustering, temporal holdout, scaffold split? (This is the question a serious practitioner asks, and §4 is why.)

**On engineering**
- What does the path from notebook to production look like? Is there a shared training framework, config system, and experiment tracker?
- How is data versioned, and how do you catch silent data corruption? (Then offer §12.1 as the reason you ask — it's a great two-way moment.)
- What's the code-review norm for research code?

**On the team**
- What's the ratio of model work to data/infra work in practice for someone in this role?
- How do you decide to kill a research direction, and who makes that call?
- What would you want me to have shipped or learned by the end of the first 90 days?

---

## 19. One-page cram sheet

**Pitch:** ESM-2 650M (frozen) → attention encoder → 3× conditioning vectors; ligand decomposed into Murcko scaffold / BRICS fragments / full graph; three conditional GATv2 denoisers trained as a curriculum; joint affinity head.

**Numbers:** 20,000 pairs → 19,989 joined · 11,795 proteins · 16,525 ligands · affinity 0.40–15.00 (μ 6.43) · protein μ 281 residues (cap 1000) · ligand μ 33.6 atoms (cap 150) · μ 5.5 BRICS fragments · 9,683 scaffolds (benzene 1,068) · split 80/10/10 (15,975/1,998/1,997) **but only 500/50 used**.

**Dims:** node 8 (+32 atom embedding = **40** diffused) · edge 5 · protein 1280 · hidden 256 · cond 128 · time 64 · T=1000 cosine · 3 layers · 4 heads · dropout 0.1 · affinity head 384→256→1.

**Training:** AdamW 5e-5, wd 1e-6 · batch 4 × accum 4 = 16 · AMP + GradScaler · ReduceLROnPlateau(0.5, 5) · 241 epochs · curriculum L1→L2→L3→joint · weights hier 0.2/0.2/0.4/0.2, all 0.4/0.3/0.2/0.1 · full-state checkpoints.

**Results:** best val 0.945 · final val 1.516 · "test" 8.03 (**on the val loader, 50 samples, different loss weights**) · levels 0.97/0.95/1.16 — **≈1.0 is the predict-zero baseline** · generated MW 154–378, QED 0.43–0.64 · predicted affinity 0.42–0.70 vs training mean 6.43 · ~30–40 s/molecule (4 × 1000 steps).

**Math:** x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε · μ̃_t = [√ᾱ_{t−1}β_t/(1−ᾱ_t)]·**x₀** + [√α_t(1−ᾱ_{t−1})/(1−ᾱ_t)]·**x_t** · β̃_t = β_t(1−ᾱ_{t−1})/(1−ᾱ_t) · L = ‖ε − ε_θ(x_t,t,c)‖² · check: at t=1, μ̃ = x₀.

**The five defects to lead with:**
1. ESM chunk id collision → ~58% of proteins conditioned on the wrong embedding (5,000 embeddings vs 11,795 map entries). Fix: content-address the cache + assert.
2. Node features unnormalised → q(x_T|x₀) ≠ N(0, I) → train/sample prior mismatch. Fix: standardise (or go discrete).
3. `mse_loss((B,1), (B,))` broadcasts to (B,B) → affinity head regresses the batch mean (warning is in the logs). Fix: squeeze + assert.
4. Generated topology/elements come from hand-written templates; the denoised features are discarded at reconstruction. Fix: discrete diffusion over node *and* edge types + learned p(N).
5. Reverse-posterior coefficients swapped and noise scaled by 0.5 → the sampler isn't the process that was trained. Fix: exact posterior + oracle round-trip test.

**Also know:** protein padding unmasked (conditioning depends on batch composition) · only the first 500 train samples, never resampled · dummy fragment graphs in the loss · empty Murcko scaffolds → zero node + self-loop · edge encoding differs between train and sample · random pair split ⇒ protein/ligand leakage · committed `ProteinEncoder.forward` doesn't run (`level*_cond` unassigned) · physics features computed but never used.

**Three sentences to close any technical round:** "The pipeline is real and fault-tolerant; the method is unvalidated and I know exactly why. The three fixes that matter are a content-addressed embedding cache, standardised diffusion inputs, and discrete diffusion over node and edge types so the model generates the topology rather than a template. If I had one week, I'd spend it on correctness and metrics before touching the architecture."
