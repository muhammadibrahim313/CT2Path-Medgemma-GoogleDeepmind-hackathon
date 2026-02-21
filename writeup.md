# CT2Path: Few-Shot Rare Cancer Diagnosis via Cross-Domain and Modality Transfer

**Accurate rare type cancer diagnosis from just 5 labeled samples**

---

### Project Name

**CT2Path** — A fully novel framework that transfers knowledge across both domain boundaries (CT imaging to pathology) and modality boundaries (3D volumetric to 2D planar) for few-shot rare cancer classification, powered by three Google HAI-DEF models working in unified orchestration.

### Your Team

**Zulqarnain Ali** — ML Pipeline Design and Production. Designed and built the end-to-end cross-domain, cross-modality training pipeline, including the attention bridge architecture, adversarial domain alignment, prototypical network integration, and HAI-DEF model orchestration. Led all architecture decisions, model training, and experimental evaluation.

**Muhammad Ibrahim Qasmi** — Technical Infrastructure and Communication. Managed the technical setup, environment configuration, and reproducibility across Kaggle notebooks. Authored the competition writeup, produced the video demonstration, and handled project documentation.

**GitHub:** [github.com/zulqarnainalipk/Cross-Dimensional-Learning-for-Data-Efficient-Medical-Imaging](https://github.com/zulqarnainalipk/Cross-Dimensional-Learning-for-Data-Efficient-Medical-Imaging) | **Video Demo:** [Link]

---

### Problem Statement

Consider a pathologist working at a district hospital in a low-resource setting. A tissue biopsy arrives from a patient suspected of having head and neck cancer. The pathologist needs to determine whether the tissue represents tumor stroma or invasion front, a distinction that directly influences surgical planning and prognosis. In well-funded academic centers, AI tools trained on thousands of annotated slides can assist with this classification. But here, there are perhaps five or ten annotated examples available for this specific task. No existing AI system can learn reliably from so few samples.

This scenario plays out daily across hospitals worldwide. Annotating pathology images requires board-certified specialists whose time is scarce and expensive, often requiring 30 minutes to several hours per whole-slide image. For rare cancer subtypes and institutions in developing regions, the labeled datasets needed to train conventional deep learning models simply do not exist. Head and Neck Squamous Cell Carcinoma alone accounts for over 890,000 new diagnoses annually, and the classification of tissue into tumor stroma versus invasion front directly informs surgical resection planning and prognosis. Yet annotated datasets for this task remain extremely rare.

CT2Path was built to solve this problem through an approach that is, to our knowledge, **fully novel**. No prior work has attempted to transfer learned knowledge simultaneously across both domain and modality boundaries in this manner. The key insight is that a CT scan and a pathology slide of the same disease describe the same underlying biology from fundamentally different perspectives. CT captures anatomy and tissue density in 3D volumetric space (one **modality**); pathology captures cellular morphology and molecular markers in 2D planar space (a different **modality**). CT imaging and pathology slides represent entirely different **domains** of medical data. They speak different languages across different domains and modalities, but they tell the same story. CT2Path builds the bridge that lets one inform the other.

HNSCC classification is just the first demonstration. CT2Path is a general-purpose cross-domain, cross-modality transfer framework by design. The same architecture can extend to other rare cancer types, other imaging modalities, and other organs entirely. Anywhere one imaging modality has abundant data and another has scarce labels, CT2Path can provide the transfer pathway.

**In short, the problem CT2Path solves:**

**1.** **Rare cancer classification is bottlenecked by labeled data** — expert annotation is slow, expensive, and unavailable in most hospitals worldwide.

**2.** **Traditional AI needs 1000+ labels** to reach acceptable accuracy, making it useless for rare subtypes and low-resource settings.

**3.** **CT scans and pathology slides show the same disease** from different domains (CT vs histology) and different modalities (3D vs 2D), but no existing system bridges both gaps simultaneously.

**4.** **CT2Path is the first framework** to transfer knowledge across both domain and modality boundaries at once, achieving 85.45% accuracy from just 5 labeled samples.

**5.** **This is not a single-use tool** — the framework can expand to other rare cancers, other organs, and other imaging modality pairs (MRI-to-pathology, PET-to-pathology), opening a new family of clinical AI applications.

---

### Overall Solution

CT2Path integrates three Google HAI-DEF models within a unified architecture, each addressing a specific capability gap that no single model could fill alone.

**MedGemma 1.5 4B** serves as a semantic enrichment layer for the 3D CT pathway. The framework samples three representative slices from each CT volume and feeds them through MedGemma with clinically crafted prompts asking for tissue characterization. The resulting text embeddings are projected to 256 dimensions and concatenated with convolutional encoder outputs, combining pixel-level features with language-level semantic understanding. MedGemma runs in 4-bit quantized mode with all features pre-extracted and cached, avoiding the cost of running a 4B model inside the training loop.

**Path Foundation** operates as a parallel feature extractor for the 2D pathology pathway. Its ViT-S backbone produces 384-dimensional patch embeddings capturing tissue architecture and cellular morphology that a general-purpose ImageNet encoder would miss. These embeddings are mean-pooled, projected to 512 dimensions, and fed into the attention bridge alongside the CNN encoder's task-specific features.

**MedSigLIP** provides cross-modal semantic alignment. For each pathology image, it computes alignment scores against carefully crafted clinical descriptions of both target classes (tumor stroma and invasion front). These scores serve as an additional supervision signal and inject clinical knowledge directly into the embedding space, grounding representations in medically meaningful concepts.

![](https://raw.githubusercontent.com/muhammadibrahim313/CT2Path-Medgemma-GoogleDeepmind-hackathon/refs/heads/main/digarms/d5.png)

**Figure 1: HAI-DEF Model Integration.** MedGemma enriches the 3D pathway with semantic understanding, Path Foundation provides pathology-specific embeddings, and MedSigLIP aligns visual and textual representations across modalities.

**How the three HAI-DEF models work together in CT2Path:**

**1.** **MedGemma 1.5** reads CT scans and provides **language-grounded anatomical understanding** that convolutional features alone cannot capture — it knows what it is looking at.

**2.** **Path Foundation** provides **broad pathology knowledge from large-scale pretraining** that our limited training data cannot — it understands tissue architecture at a level our small dataset never could teach.

**3.** **MedSigLIP** provides **explicit alignment between visual patterns and clinical meaning** across modalities — it grounds everything in medically meaningful concepts, not just statistical patterns.

**4.** The **cross-modality attention bridge** fuses all five feature streams (3D CNN + MedGemma + 2D CNN + Path Foundation + MedSigLIP) and dynamically identifies which CT knowledge matters for each pathology query.

**5.** The result: **85.45% accuracy, 94.45% AUC-ROC from just 5 labeled samples** — matching methods that need 200x more data, running on a single Kaggle T4 GPU.

---

### Technical Details

## Framework Architecture

CT2Path consists of five integrated components that together enable cross-domain, cross-modality knowledge transfer.

![](https://raw.githubusercontent.com/muhammadibrahim313/CT2Path-Medgemma-GoogleDeepmind-hackathon/refs/heads/main/digarms/d1.png)

**Figure 2: CT2Path Overall Framework Architecture.** Five feature streams from two input modalities are fused through the cross-modality attention bridge and classified via prototypical networks.

A 3D convolutional encoder (ResNet3D-18, ~33M parameters) processes CT volumes, extracting hierarchical volumetric features from 16-slice subvolumes at 128x128 resolution with Hounsfield unit normalization. In parallel, a 2D convolutional encoder (ResNet2D-50, ~25M parameters) processes pathology tiles, handling multi-channel immunofluorescence input (DAPI, PanCK, CD3, CD8, FOXP3, PDL1) through attention-weighted channel aggregation. Both encoders feed into the cross-modality attention bridge alongside the three HAI-DEF model outputs. Downstream, a domain discriminator enforces modality-invariant representations, and a prototypical network enables few-shot classification.

## The Cross-Modality Attention Bridge

The attention bridge is the core architectural innovation. 3D volumetric features and 2D planar features exist in entirely different representational spaces across different imaging modalities. Simple concatenation cannot capture the complex, content-dependent relationships between them. What is needed is a mechanism that dynamically identifies which aspects of the 3D knowledge are relevant for each specific 2D query.

The 2D pathology features serve as queries, and the 3D CT features serve as keys and values. For every pathology image, the attention mechanism computes compatibility scores between the pathology representation and each CT-derived feature, producing a weighted combination that captures exactly the volumetric context most relevant to that tissue sample.

![](https://raw.githubusercontent.com/muhammadibrahim313/CT2Path-Medgemma-GoogleDeepmind-hackathon/refs/heads/main/digarms/d2.png)

**Figure 3: Cross-Modality Attention Bridge.** Pathology features query the CT feature space through scaled dot-product attention across 8 parallel heads, producing 256-dim fused features.

Given query features **q** from the 2D encoder and context features {**c**_1, ..., **c**_m} from the 3D encoder, attention scores are computed as scaled dot products, passed through softmax to produce weights summing to one. The fused representation is the weighted sum of context features. With 8 parallel heads, the bridge simultaneously attends to different types of cross-modality correspondences. The design is content-dependent (different inputs activate different CT knowledge), interpretable (attention weights reveal which features contributed), and efficient (standard matrix operations on GPU).

## Adversarial Domain Alignment

The feature distributions from 3D and 2D encoders differ substantially due to the inherent domain and modality gap. CT2Path employs adversarial alignment through a gradient reversal layer. A domain discriminator classifies whether features originated from the 3D or 2D pathway. The gradient reversal layer negates gradients during backpropagation, pushing encoders to produce modality-indistinguishable features. Reversal strength is annealed from strong (aggressive alignment early) to weak (preserving discriminative features later).

![](https://raw.githubusercontent.com/muhammadibrahim313/CT2Path-Medgemma-GoogleDeepmind-hackathon/refs/heads/main/digarms/d3.png)

**Figure 4: Adversarial Domain Alignment with Gradient Reversal.** The domain discriminator and gradient reversal layer force encoders to learn modality-invariant representations.

## Few-Shot Learning with Prototypical Networks

Instead of learning a fixed classifier requiring large labeled datasets, prototypical networks learn an embedding space where classification reduces to nearest-neighbor computation. During episodic training, each episode samples a support set (K examples per class) and a query set. Class prototypes are computed as mean embeddings, and queries are classified by Euclidean distance to each prototype with softmax over negative distances.

![](https://raw.githubusercontent.com/muhammadibrahim313/CT2Path-Medgemma-GoogleDeepmind-hackathon/refs/heads/main/digarms/d4.png)

**Figure 5: Prototypical Network Few-Shot Classification.** Class prototypes from the support set classify query samples by proximity in the learned embedding space.

## Datasets

The **3D source domain** uses the LungCT-Diagnosis dataset (TCIA): 61 CT volumes preprocessed into 16-slice subvolumes at 128x128 with HU normalization. The **2D target domain** uses the HNSCC-mIF-mIHC dataset (TCIA): 268 multiplex immunofluorescence samples across two classes at 224x224 resolution.

## Training Protocol

Phase 1: Self-supervised contrastive pretraining on CT volumes, 20 epochs. Phase 2: Episodic meta-learning with the full framework, 36 epochs per K-shot setting using 3-fold stratified cross-validation (~151 train, 27 val, 89 test per fold). AdamW optimizer, cosine annealing LR (1e-4 for Phase 1, 1e-3 for Phase 2), batch size 4, early stopping with patience 10. All experiments on a single Kaggle T4 GPU.

## Results

**Aggregate Performance (3-fold cross-validation):**

| Metric | Mean | Std Dev | 95% CI |
|--------|------|---------|--------|
| **Accuracy** | **85.45%** | 2.40% | [78.15%, 92.76%] |
| **AUC-ROC** | **94.45%** | 1.51% | [89.86%, 99.05%] |
| **Balanced Accuracy** | 85.74% | 2.49% | [78.17%, 93.31%] |
| **F1 Score (Macro)** | 85.41% | 2.42% | [78.06%, 92.76%] |
| **Cohen's Kappa** | 71.01% | 4.81% | [56.38%, 85.63%] |

**Cross-Validation Consistency:**

| Fold | Accuracy | F1 Score | AUC-ROC | Cohen's Kappa |
|------|----------|----------|---------|---------------|
| Fold 1 | 84.44% | 84.38% | 92.33% | 68.77% |
| Fold 2 | 83.15% | 83.11% | 95.34% | 66.57% |
| Fold 3 | 88.76% | 88.75% | 95.69% | 77.68% |

**K-Shot Scaling:**

| K-Shot | Best Accuracy | Best AUC-ROC | Range |
|--------|---------------|--------------|-------|
| K = 1 | 74.07% | 0.79 | 62.96% - 74.07% |
| K = 3 | 85.19% | 0.95 | 74.07% - 85.19% |
| K = 5 | 88.89% | 0.97 | 74.07% - 88.89% |

**Comparison with Baseline Approaches:**

| Approach | Labels Required | Typical Accuracy |
|----------|----------------|------------------|
| Train from scratch | 1000+ | ~75-80% |
| Transfer learning (ImageNet) | 100+ | ~78-82% |
| Self-supervised pretraining | 50+ | ~80-85% |
| **CT2Path (ours)** | **5-50** | **85.45%** |

## Model Architecture Summary

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| 3D Encoder | ResNet3D-18 | ~33M |
| 2D Encoder | ResNet2D-50 | ~25M |
| Attention Bridge | Multi-Head (8 heads) | ~2M |
| Domain Discriminator | 3-layer MLP | ~0.5M |
| Prototypical Network | Distance-based | ~0.1M |
| **Total (trainable)** | | **~61M** |

| HAI-DEF Model | Parameters | Quantization | Output Dim |
|---------------|------------|--------------|------------|
| MedGemma 1.5 4B-it | 4B | 4-bit | 256 |
| Path Foundation v1.0 | ~100M | FP16 | 512 |
| MedSigLIP 448 | ~800M | FP16 | 512 |

## Future Directions

CT2Path demonstrates cross-domain and cross-modality transfer for one cancer type, but the framework can naturally expand along several axes:

**Same organ, same tissue level:** Lung CT scans can transfer knowledge to lung pathology slides, keeping both domain and organ aligned while still bridging the 3D-to-2D modality gap.

**Different organ, different tissue level:** Lung CT knowledge can potentially transfer to head-and-neck pathology (as demonstrated here), or to other organs entirely, because the underlying tissue characterization patterns share structural similarities across anatomical regions.

**New modality pairs:** The architecture can accommodate MRI-to-pathology, PET-to-pathology, or ultrasound-to-pathology transfer by substituting the appropriate source domain encoder and data.

**Other rare cancers:** Any rare cancer type where pathology labels are scarce but related volumetric imaging is available becomes a candidate for CT2Path's transfer approach.

Additional future work includes unpaired cross-modality transfer via cycle-consistency constraints (removing the need for same-cohort training data), full-volume 3D processing through attention pooling, and more advanced meta-learning classification heads such as MAML or relation networks.

---

### Resources

GitHub: [github.com/zulqarnainalipk/Cross-Dimensional-Learning-for-Data-Efficient-Medical-Imaging](https://github.com/zulqarnainalipk/Cross-Dimensional-Learning-for-Data-Efficient-Medical-Imaging)

Datasets: [HNSCC-mIF-mIHC](https://www.cancerimagingarchive.net/collection/hnscc-mif-mihc-comparison/) | [LungCT-Diagnosis](https://www.cancerimagingarchive.net/collection/lungct-diagnosis/) (TCIA)

Models: MedGemma 1.5 4B | Path Foundation v1.0 | MedSigLIP 448s
