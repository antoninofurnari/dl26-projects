# Course Instructions and Projects

## 📚 Initial Instructions for Students

This repository is the official starting point for all course projects. 

### 📏 Note on Project Sizes & Group Dynamics
The "Suggested Size" (Small, Medium, Large) is a rough estimate of the workload, mapping approximately to group sizes of 1, 2, and 3 members. However, these are just recommendations! 
- A 3-person group selecting a "Small" project will be expected to complete all **Extra Objectives** and demonstrate a higher level of polish and depth. 
- Conversely, a solo student who ambitiously selects a "Large" project will be evaluated with adapted expectations regarding the breadth of completed tasks.

Here are the steps to get started:
1. **Choose a project**: Consult the [project list below](#project-list) to read the available tracks and check which ones are free or already assigned. Then communicate your chosen project to the professor via email.
2. **Fork**: Create a **fork** of this repository in your personal GitHub account. While it is preferable to use the Fork button in the top right (to keep the history visible for evaluation), you can also create a standalone repository and keep it private if you prefer.
3. **Clone**: Clone your fork locally.
4. **Work in this root**: Consult [CONTRIBUTING.md](CONTRIBUTING.md) for the required conventions on how to structure folders (`src/`, `data/`, `notebooks/`), how to write clean code, and how to use Git professionally in a team. Replace the placeholders in the `README.md` file with the technical information about the repository and `docs/REPORT.md` with the description of the project and the work done.
5. **AI Usage Policy**: The use of generative AI tools (ChatGPT, GitHub Copilot, Claude, etc.) is **permitted, but regulated**. The use of these tools is encouraged to speed up boilerplate code writing, for debugging, or as documentation support. However, **never delegate strategic thinking and architectural choices to AI**. Elaborate your strategy, write or generate the code, and take full responsibility for every line. The use of such tools must be explicitly declared in the final report.
6. **License**: It is good practice to release your work open source. You will find a `LICENSE` file (pre-set to MIT license). Open the file, replace `[Year]` and `[Name and Surname]` with the current year and the members of your team. Remember to choose a different one if you do not want to freely share your code.
7. **Submission**: Your GitHub fork is the **final deliverable** of the project. Ensure the code is reproducible following the instructions below and that the slides for the exam presentation are placed inside the `docs/` folder. If you opted for a private repository, evaluation can take place by making the repo visible to the professor (handle `antoninofurnari`) or by sending the repository source code via email.

---

This file contains the list of available projects, complete details for each project, and the formed groups.

## Project List

| ID | Title | Reference Module | Suggested Size | Dataset | Assigned |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | [Metric Learning for Face Recognition](#project-1) | Metric Learning | Small | CASIA-WebFace | G25 |
| 2 | [Few-shot Learning for Gesture Recognition](#project-2) | Metric Learning | Medium | HAGRID | G32 |
| 3 | [Graph-based Metric Learning for Scene Understanding](#project-3) | Metric Learning | Large | GQA | G18 |
| 4 | [Feature-based Knowledge Distillation](#project-4) | Knowledge Distillation | Small | CIFAR-100 | G26 |
| 5 | [Cross-Modal Knowledge Distillation (Audio to Vision)](#project-5) | Knowledge Distillation | Large | EPIC-Kitchens | Overfittony |
| 6 | [Knowledge Distillation for Mobile Action Recognition](#project-6) | Knowledge Distillation | Small | HMDB-51 | G24 |
| 7 | [Domain Adaptation for Action Recognition – Exocentric → Egocentric](#project-7) | Domain Adaptation | Medium | Assembly101 | CassiaBranca |
| 8 | [Domain Adaptation with Image-to-Image Translation (CycleGAN)](#project-8) | Domain Adaptation | Medium | Office-31 | Free |
| 9 | [Multi-source Domain Adaptation for Action Recognition](#project-9) | Domain Adaptation | Large | HMDB-51/UCF-101/Kinetics | DataLost |
| 10 | [Contrastive Learning for Video Representation (SimCLR)](#project-10) | Self-Supervised Learning | Small | UCF-101 | G16 |
| 11 | [Masked Autoencoders for Image Representation Learning](#project-11) | Self-Supervised Learning | Small | ImageNet 1K | G21 |
| 12 | [Clustering-based SSL for Action Discovery](#project-12) | Self-Supervised Learning | Medium | Kinetics-400 | DataMinds |
| 13 | [Temporal Action Segmentation from Video](#project-13) | Video Understanding | Small | EGTEA Gaze+ | G26 |
| 14 | [Action Anticipation from Video](#project-14) | Video Understanding | Medium | EPIC-KITCHENS | Le larunghie |
| 15 | [Vision-Language Alignment with CLIP for Video](#project-15) | Vision & Language | Medium | EPIC-KITCHENS | Justgood AI |
| 16 | [Multimodal Action Recognition – Video + Audio](#project-16) | Video Understanding | Large | EPIC-KITCHENS | DeepTeam |
| 17 | [Egocentric Video + Gaze for Action Recognition](#project-17) | Video Understanding | Large | EGTEA Gaze+ | The Outliers 2.0 |
| 18 | [State-Space Models (Mamba) for Mistake Detection](#project-18) | Advanced Sequential Modeling | Large | Assembly101 | LeMeCla |
| 19 | [Transformer vs RNN for Procedural Video Understanding](#project-19) | Advanced Sequential Modeling | Medium | EGO4D Goal-Step | Free |
| 20 | [Image & Language Representation Learning](#project-20) | Vision & Language | Medium | MS-COCO | G37 |
| 21 | [Deep Reinforcement Learning for Frame Selection in Video](#project-21) | Reinforcement Learning | Large | UCF101 | Free |
| 22 | [Learn to Play Super Mario Bros with Deep Reinforcement Learning](#project-22) | Reinforcement Learning | Medium | Super Mario Bros Environment | G20 |
| 23 | [Align a Small LLM with GRPO for Strict Code or JSON Generation](#project-23) | Reinforcement Learning | Medium | Synthetic Logic Array | G23 |
| 24 | [Cross-Modal Knowledge Distillation (Audio to Vision)](#project-24) | Knowledge Distillation | Large | VGGSound | Zero e Uno |

## Detailed Project Descriptions


<a id='project-1'></a>
### Track 1: Metric Learning for Face Recognition
**Suggested Size**: Small  
**Reference Module**: Metric Learning  

#### Problem Description
Face recognition systems are ubiquitous, but identifying individuals accurately across varying lighting, poses, and expressions is challenging. This project involves learning robust, identity-preserving embeddings by training a model on a large face dataset and testing its generalization on unseen individuals. Split the dataset into a training set and a test set, where the test set contains only individuals that are not present in the training set.

#### Dataset
- **CASIA-WebFace** (https://www.kaggle.com/datasets/debarghamitraroy/casia-webface) or a subset.
- ~500,000 images representing ~10,000 subjects.

#### Minimum Objectives
1. **Baseline**: Train a standard image classification model using a fine-tuned ResNet-18 backbone, relying on the classifier's features and a K-Nearest Neighbors (KNN) search to generalize to new faces.
2. **Metric Learning**: Implement Triplet Loss with hard negative mining. Given an anchor face, dynamically mine challenging positives (same person) and negatives (different people).
3. **Retrieval Evaluation**: Compute mAP @1, 5, 10 to evaluate if the model successfully retrieves the correct identity within its top rank predictions.
4. **Cluster Analysis**: Perform an analysis of the latent space (e.g., using t-SNE or PCA) to verify that faces of identical subjects cluster closely together.

#### Extra Objectives
- Implement and compare with advanced margin-based loss functions (e.g., ArcFace, CosFace, Siamese networks).
- Perform ablations on hyper-parameters like mining strategies (offline vs. online) and batch size.
- Create a small demo that processes and recognizes localized facial images collected directly by your team.

---

<a id='project-2'></a>
### Track 2: Few-shot Learning for Gesture Recognition
**Suggested Size**: Medium  
**Reference Module**: Metric Learning  

#### Problem Description
Recognizing human hand gestures is vital for sign language translation and human-computer interaction. However, relying on massive labeled datasets for every potential gesture is impractical. This project tackles the few-shot learning paradigm: a system that learns to recognize new, unseen hand gestures by being provided with only 1 to 10 labeled examples during inference. You will train the backbone on a large corpus and then generalize to the unseen set with linear probing on few labeled data.


#### Dataset
- **HAGRID** (https://github.com/hukenovs/hagrid) (or a subset)
- ~1M RGB frames covering 33 classes of gestures.

#### Minimum Objectives
1. **Baseline**: Use a 2D/3D CNN or Vision Transformer on gesture images treated as a standard classification problem. Extract the pre-softmax embeddings and use KNN to classify novel gesture classes.
2. **Metric Learning**: Train using Triplet Loss with hard negative mining to enforce clustering of identical gestures. Use a linear probe or nearest-centroid tracking for generalization to new class sets.
3. **Few-Shot Evaluation**: Measure classification performance systematically using Accuracy on 1/5/10-shot.
4. **Data Analysis Report**: Document how the predictive performance scales as the number of provided shots ("examples") increases from 1 to 5 to 10.

#### Extra Objectives
- Utilize 1D CNNs, RNNs, or Transformers analyzing temporal skeleton coordinates extracted via MediaPipe instead of raw RGB frames.
- Provide a rigorous failure case analysis detailing why the model confuses specific gestures.
- Create a demo demonstrating the model generalizing instantly to a few newly recorded gestures.

---

<a id='project-3'></a>
### Track 3: Graph-based Metric Learning for Scene Understanding
**Suggested Size**: Large  
**Reference Module**: Metric Learning  

#### Problem Description
Images exist as a rich tapestry of interrelated objects, yet standard CNN architectures often reduce this to a single flat vector. This project aims to represent complex scenes (like kitchens or offices) structurally as graphs, where nodes are objects and edges describe their spatial or semantic relationships. By learning dynamic embeddings of these graphs, you will create a system capable of robust scene-to-scene retrieval based on compositional similarity rather than just pixel-level textures.

#### Dataset
- **GQA** (https://cs.stanford.edu/people/dorarad/gqa/download.html) or a subset.
- 100K images, utilizing provided scene graphs and localized scene labels.

#### Minimum Objectives
1. **Baseline**: A standard CNN processing the raw image to predict the scene label/embedding, ignoring the explicit graph structure.
2. **Graph Encoder**: Implement a Graph Convolutional Network (GCN) that ingests ground truth scene graphs (nodes and edges) resulting in a unified scene embedding.
3. **Contrastive Training**: Train the embeddings using contrastive (or triplet) loss so that paired graphs from similar scenarios (e.g., identical activity and place) cluster tightly.
4. **Retrieval Evaluation**: Given a query scene graph, perform retrieval on the dataset to fetch structurally similar graphs, evaluating via standard classification metrics (Accuracy, Precision, Recall).

#### Extra Objectives
- Test robustness to structural perturbations (e.g., artificially drop nodes/edges from test graphs to see if retrieval degrades).
- Build a dynamic graph extractor: utilize Vision-Language Models (VLMs) or Object Trackers to automatically extract scene graphs from raw videos instead of relying on ground truth.
- Visualize and interpret which edges/relations act as critical focal points for the model's similarity metric.

---

<a id='project-4'></a>
### Track 4: Feature-based Knowledge Distillation
**Suggested Size**: Small  
**Reference Module**: Knowledge Distillation  

#### Problem Description
Deep neural networks achieve staggering accuracies but often require massive computational power, making them unsuited for edge devices. Knowledge Distillation (KD) provides a bridge by forcing a small, efficient "student" network to mimic a heavy "teacher" network. In this project, you will explore advanced KD strategies. By moving beyond simple logit-matching, you will extract and distill intermediate convolutional features from the teacher to force the student to learn identically rich hierarchical representations.

#### Dataset
- **CIFAR-100** or a subset of **ImageNet**.

#### Minimum Objectives
1. **Teacher and Student**: Define a large teacher CNN (e.g., pretrained ResNet-50) and initialize a small student CNN (e.g., ResNet-18 or MobileNet).
2. **Baseline**: Train the small student network purely from scratch using cross-entropy, without any distillation framework.
3. **Distillation Algorithms**: Implement standard Logit-based KD alongside feature distillation methods (e.g., FitNets).
4. **Evaluation**: Compare the test accuracies of the Teacher, the standard Student (Baseline), and the KD-trained Student. Report metrics on model size (MB) and raw inference latency (ms).

#### Extra Objectives
- Experiment with blending Feature-based distillation with classical Logit-based distillation via weighted loss terms.
- Implement advanced distillation mechanisms such as Attention Transfer or Relational Knowledge Distillation.

---

<a id='project-5'></a>
### Track 5: Cross-Modal Knowledge Distillation (Audio to Vision)
**Suggested Size**: Large  
**Reference Module**: Knowledge Distillation  

#### Problem Description
In many real-world edge settings, deploying multi-modal sensors is too expensive or computationally heavy. This project addresses "modality hallucination" by distilling knowledge from a robust Video-based teacher into an Audio-only student model. This enables the resulting lightweight audio model to internally leverage visual context during inference—even when dealing exclusively with sound clips in production.

#### Dataset
- **EPIC-Kitchens** (https://epic-kitchens.github.io/) for visual frames and **EPIC-Sounds** (https://epic-kitchens.github.io/epic-sounds/) for aligned auditory tracks.

#### Minimum Objectives
1. **Baseline**: Train a standard audio encoder (e.g., Audio Spectrogram Transformer, AST) using purely audio spectrograms and cross-entropy loss.
2. **Teacher Model**: Fine-tune an image encoder (e.g., ResNet-50) on EPIC-Kitchens visual frames to act as the robust expert.
3. **Student Model**: Train the AST audio encoder using distillation loss mapped from the visual embeddings of the teacher, bridging the sensory gap.
4. **Evaluation**: Compare classification accuracy of the vision teacher, the audio baseline, and the distilled audio student. Produce a size (MB) and inference time (ms) comparison.

#### Extra Objectives
- Enforce strict dimensional alignment using Contrastive cross-modal distillation between audio and video embeddings.
- Implement a temporal 3D CNN or a Video Transformer as the visual teacher to account for longitudinal temporal dynamics.
- Explore and benchmark lightweight alternatives to AST.

---

<a id='project-6'></a>
### Track 6: Knowledge Distillation for Mobile Action Recognition
**Suggested Size**: Small  
**Reference Module**: Knowledge Distillation  

#### Problem Description
Action recognition in videos is inherently expensive due to the added temporal dimension, making deployment to mobile applications highly restrictive. This project focuses on aggressive model compression, translating the robust, high-dimensional knowledge learned by a heavy 3D ResNet-50 into an ultra-lightweight MobileNet 3D. The resulting model must maintain respectable temporal reasoning while reducing parameter count by 5–10x.

#### Dataset
- **HMDB-51** or **UCF-101** (focusing on sports/daily actions).
- Download and utilize pre-extracted video features if computational restraint is an issue.

#### Minimum Objectives
1. **Teacher Model**: Evaluate and solidify a pre-trained 3D ResNet-50 to establish the upper bound baseline accuracy.
2. **Baseline**: Train a 3D MobileNet from scratch using purely hard labels, defining the lower bound capability of the architecture.
3. **Knowledge Transfer**: Re-train the 3D MobileNet student allowing it to learn from the soft probability distributions (logits) of the teacher.
4. **Evaluation**: Graph the test accuracies representing Teacher vs. Baseline Student vs. Distilled Student. Report deployment metrics: model size (MB) and sequential inference tracking (ms).

#### Extra Objectives
- Perform temperature scaling ablations (T = 1, 5, 10, 20) to analyze how smoothing the teacher's logits impacts knowledge transfer.
- Expand KD to include Attention Transfer across intermediate temporal activation mappings.
- Provide t-SNE visualizations of the latent space to illustrate the structural differences in how the teacher and student map actions.

---

<a id='project-7'></a>
### Track 7: Domain Adaptation for Action Recognition – Exocentric → Egocentric
**Suggested Size**: Medium  
**Reference Module**: Domain Adaptation  

#### Problem Description
Models trained on vast repositories of security camera or YouTube footage (exocentric views) catastrophically fail when deployed on smart glasses (egocentric views) due to massive shifts in perspective, occlusion, and background statistics. This project requires you to implement Domain Adaptation (DA) techniques that can effectively transfer semantic understanding from an exocentric labeled source to an egocentric unlabeled target, mitigating the domain gap.

#### Dataset
- **Assembly101** (https://assembly101.github.io/).
- Exploit multi-view recordings: assign one external view as the exocentric Source, and the head-mounted view as the egocentric Target.

#### Minimum Objectives
1. **Baseline**: Train a classification model strictly on the labeled target data to document standard performance bounds (or train on source and directly zero-shot test on the target to demonstrate domain shift collapse).
2. **Adversarial DA**: Implement a Gradient Reversal Layer (GRL) module consisting of a shared feature encoder, action classification head, and a domain discriminator (predicting source vs. target).
3. **Evaluation**: Optimize using adversarial backpropagation and plot target validation accuracy against standard classification matrices.
4. **Analysis**: Report whether the shared representations successfully confused the domain discriminator while improving the classification accuracy relative to the baseline.

#### Extra Objectives
- Experiment with the Maximum Mean Discrepancy (MMD) distribution alignment statistic as an alternative to GRL.
- Extract t-SNE mapping of features pre- and post-DA to visualize domain alignment.
- Conduct a per-class discrepancy analysis to identify which specific physical actions resist viewpoint adaptation the most.

---

<a id='project-8'></a>
### Track 8: Domain Adaptation with Image-to-Image Translation (CycleGAN)
**Suggested Size**: Medium  
**Reference Module**: Domain Adaptation  

#### Problem Description
When confronted with massive domain shifts (e.g., from generated synthetic simulations to real-world photographs), aligning embedding spaces may not be enough. This project takes a highly visual approach: using a CycleGAN to actively synthetically re-render images from Domain A to look like they were photographed in Domain B. This visual translation acts as a powerful pre-processing augmentation, forcing the downstream classifier to operate on familiar looking features.

#### Dataset
- **Office-31** (Source: Amazon, Target: DSLR) (https://www.kaggle.com/datasets/xixuhu/office31)
- Alternative: **VisDA** (synthetic → real) (https://ai.bu.edu/visda-2019/)

#### Minimum Objectives
1. **Baseline**: Train an image classifier exclusively on the Source dataset and test it on the Target dataset to establish baseline domain collapse.
2. **CycleGAN Training**: Implement two adversarial generators (A→B, B→A) and two discriminators relying on cycle-consistency losses without any aligned image pairs.
3. **Translation Execution**: Actively force the trained CycleGAN to translate all Source visuals to visually emulate Target aesthetics.
4. **Evaluation**: Utilize the translated frames (and originals) to train a classifier. Measure the resulting downstream classifier accuracy on the Target validation split, supplemented by qualitative human analysis of translation quality.

#### Extra Objectives
- Establish a simultaneous Domain Adversarial Neural Network (DANN) that dynamically translates features while training.
- Reverse the perspective: train the final classifier purely on the Target domain, but test it on translated Source images. Does the pipeline work better backward?
- Compare Image-to-Image translations against basic feature-level domain adaptation methods.

---

<a id='project-9'></a>
### Track 9: Multi-source Domain Adaptation for Action Recognition
**Suggested Size**: Large  
**Reference Module**: Domain Adaptation  

#### Problem Description
Relying on a single source dataset for domain pre-training frequently biases a model, restricting its capability to generalize to an unlabeled target. Multi-source Domain Adaptation embraces multiple distinct labeled datasets (Source 1, Source 2) and actively combines their varied knowledge bases to inform the predictions of a highly distinct unlabeled target dataset. This project challenges you to develop a framework capable of dynamically weighting multiple distinct knowledge domains simultaneously.

#### Dataset
- **Source 1**: HMDB-51
- **Source 2**: UCF-101
- **Target**: Kinetics (subset of classes)

#### Minimum Objectives
1. **Baseline**: Train an encoder combining all data from both sources natively without any domain-adaptation formatting, testing zero-shot on the target dataset.
2. **Multi-Source Architecture**: Build an architecture containing a shared CNN feature encoder, two independent source-specific classifiers, and one combined target classifier.
3. **Weighted Ensemble**: Formulate a mechanism that structurally weights the reliance on each Source based on dynamic embedding similarity to the current Target batch.
4. **Evaluation**: Optimize the entire network simultaneously utilizing domain discriminators, detailing target accuracy evolution and the numerical influence ratio of Source 1 vs Source 2.

#### Extra Objectives
- Perform an incomplete batch simulation: mathematically observe test stability when Source data is abruptly dropped or unbalanced during training epochs.
- Conduct an analogy study defining exactly how global performance curves behave as the dataset count increases or dataset overlaps vary.

---

<a id='project-10'></a>
### Track 10: Contrastive Learning for Video Representation (SimCLR)
**Suggested Size**: Small  
**Reference Module**: Self-Supervised Learning  

#### Problem Description
Annotating massive video datasets with localized, specific action tags is phenomenally expensive. Self-supervised contrastive learning (SimCLR) seeks to bypass human labeling by pre-training a video encoder to simply understand raw visual patterns. The model is forced to recognize that two heavily augmented crops of the *same* video clip represent the identical semantic concept, whilst strongly differing from other background videos. This creates highly powerful initial weights for downstream tasks.

#### Dataset
- **UCF-101** (https://www.crcv.ucf.edu/data/UCF101.php)
- 1k videos, 101 action categories. Split into entirely unlabelled structural logic sets and a sparse labeled probe set.

#### Minimum Objectives
1. **Baseline**: Train a standard 3D ResNet-18 purely from scratch exclusively utilizing the small labeled dataset, noting the massive overfitting potential.
2. **Self-Supervised Pre-Training**: Apply SimCLR algorithms on the unlabeled partition. Generate spatio-temporal video augmentations (crops, color jitters, temporal flips) and enforce embedding similarities.
3. **Linear Probing**: Freeze the SimCLR pre-trained encoder completely and train only a terminal fully-connected layer on the limited labeled dataset to classify specific actions.
4. **Evaluation**: Output standard top-1/top-5 accuracy metrics, charting the robust success of unsupervised linear probes versus the fragile from-scratch baseline.

#### Extra Objectives
- Modify the temperature hyper-parameters (e.g., T=0.1, 0.5, 1.0) inside the contrastive InfoNCE loss, charting divergence convergence rates.
- Visualize representation logic via t-SNE mapping; similar physical actions should visually cluster without being taught labels.
- Transition from SimCLR to Momentum Contrast (MoCo) to drastically artificially expand batch sizing for improved contrastive mapping.
- Compare classical linear probing results directly alongside full model un-frozen fine-tuning.

---

<a id='project-11'></a>
### Track 11: Masked Autoencoders for Image Representation Learning
**Suggested Size**: Small  
**Reference Module**: Self-Supervised Learning  

#### Problem Description
The Masked Autoencoder (MAE) revolutionized self-supervised learning by proving models can learn phenomenal relational context simply by playing a jigsaw puzzle. By aggressively masking out over 50-75% of an image patch grid, the vision transformer is brutally forced to predict and reconstruct the missing visual data from surrounding semantic context. This task focuses entirely on building and pre-training these powerful contextual generators natively prior to probing them.

#### Dataset
- **ImageNet 1K** or a structurally balanced subset (https://www.image-net.org/download.php).
- 1K classes, roughly 1M images.

#### Minimum Objectives
1. **Baseline**: Train a supervised visual transformer model natively from scratch solely utilizing the small supervised label set.
2. **MAE Pre-Training**: Construct an asymmetric Masked Autoencoder (heavy encoding, lightweight decoding) utilizing raw ImageNet frames lacking their associated labels.
3. **Linear Probing**: Destroy the decoder, freeze the heavy contextual encoder, and attach a linear layer to map to classes using traditional supervised learning.
4. **Evaluation**: Generate the classification accuracy metric, rigorously comparing the MAE structural pre-training to the randomized baseline initialization.

#### Extra Objectives
- Implement decoders capable of exporting visual reconstructions, mapping the predicted masked patches to human-readable matrices.
- Systematically ablate patch masking percentages (e.g., 25%, 50%, 75%, 90%) tracking reconstruction loss against ultimate probe accuracy.
- Contextualize results by comparing MAE performance against a pure pixel-level classical Autoencoder and a contrastive learning method.

---

<a id='project-12'></a>
### Track 12: Clustering-based SSL for Action Discovery
**Suggested Size**: Medium  
**Reference Module**: Self-Supervised Learning  

#### Problem Description
Imagine observing hours of YouTube DIY tutorials without speaking the language. You naturally start clustering repetitive conceptual movements—this is an "unscrewing" motion, this is "hammering." This project mirrors that human deduction protocol using unsupervised iterative clustering pseudo-labels. Instead of teaching a network classes directly, you identify recurring actions in a continuous vacuum, establishing latent clusters corresponding to true semantic tasks without any supervised intervention.

#### Dataset
- **Kinetics-400** (https://github.com/cvdfoundation/kinetics-dataset).
- Narrow down to a subset of ~10-20 highly localized action classes or completely unlabeled instructional footage.

#### Minimum Objectives
1. **Baseline**: Establish a uniform pseudo-labeling schema or generic k-means pass acting on raw untuned ResNet embeddings.
2. **Feature Extraction**: Train or utilize an advanced foundational self-supervised mechanism (VideoMAE or SimCLR) to process raw video arrays into dense embeddings.
3. **Iterative K-Means**: Run progressive cluster optimization loops mapping similar temporal features into pseudo-label groupings.
4. **Evaluation**: Correlate mathematically the overlap of unsupervised cluster divisions (purity scores) to true grounded procedural steps (using underlying human annotations merely as an evaluation key).

#### Extra Objectives
- Compare density-based techniques (DBSCAN) against geometric methods (K-Means).
- Incorporate distinct SSL architectures as the feature foundation to evaluate differing embedding spacing.
- Provide a qualitative interpretability analysis mapping the abstract pseudo-clusters backwards to decipher meaningful human-readable labels regarding the distinct procedural movements identified by the network.

---

<a id='project-13'></a>
### Track 13: Temporal Action Segmentation from Video
**Suggested Size**: Small  
**Reference Module**: Video Understanding  

#### Problem Description
While action *recognition* identifies *what* happens in a unified video clip, temporal *segmentation* acts like a meticulous editor identifying exactly *when* each individual action begins and ends across a long, erratic continuous stream. In this task, you map sequential long-form video vectors to densely predict specific activity states for every individual frame segment, establishing accurate bounds for procedural human interactions.

#### Dataset
- **EGTEA Gaze+** (https://cbs.ic.gatech.edu/fpv/).
- Utilize pre-extracted dense features (e.g., derived from RULSTM protocols) complete with annotated start and end bound timestamps.

#### Minimum Objectives
1. **CNN Temporal Baseline**: Utilize a standard 1D Convolutional Neural Network scaling horizontally across the temporal frame distributions predicting the local status.
2. **RNN Module**: Engineer an LSTM to sequentially process the temporal features accounting for rolling local contexts and historical inertia.
3. **xLSTM Architecture**: Update the historical methodology by implementing the recent extended LSTM (xLSTM) framework targeting exponential scaling sequences.
4. **Evaluation**: Compile comparative matrices demonstrating overall accuracy metrics while qualitatively logging systematic boundary leakage (e.g., predicting an action is concluding five frames late uniformly across all models).

#### Extra Objectives
- Establish a Soft-NMS (Non-Maximum Suppression) post-processing subroutine mathematically merging overlapping temporal detections.
- Deploy Mamba-based modules evaluating alternative state-space calculations over recursive attention boundaries to mitigate long-string decay.

---

<a id='project-14'></a>
### Track 14: Action Anticipation from Video
**Suggested Size**: Medium  
**Reference Module**: Video Understanding  

#### Problem Description
The highest level of automated video intelligence is predictive. Instead of merely logging that a chef is slicing a carrot, a robust architecture will actively anticipate that the next immediate action requires picking up a bowl. Transitioning from recognition to anticipation flips the paradigm entirely forward, requiring complex memory states modeling historical context to probabilistically chart impending procedural sequences utilizing the proven RULSTM challenge paradigm.

#### Dataset
- **EPIC-KITCHENS** (https://epic-kitchens.github.io).
- Exploit pre-extracted foundational temporal features. Analyze mechanics through the CodaBench anticipation challenge specifications (https://www.codabench.org/competitions/14471/).

#### Minimum Objectives
1. **Baseline**: Deploy the foundational RULSTM model mathematically establishing standard horizon-focused predictive accuracy markers.
2. **xLSTM Integration**: Implement an extended Long Short-Term Memory logic core addressing temporal decay drop-offs and complex procedural dependencies.
3. **SSM Deployment**: Run a state-space sequential model (Mamba/Hippo variant) analyzing temporal contextual propagation.
4. **Evaluation**: Benchmark all three foundational concepts against the primary top-5 horizon accuracy parameters charting degradation relative to distance of the impending action.

#### Extra Objectives
- Re-architect the core RULSTM pipeline swapping native memory modules entirely for the xLSTM protocol charting specific memory drop-out recoveries.
- Expand parameterization to include modern sparse Attention networks (Transformers) measuring efficiency and anticipation scaling limits.

---

<a id='project-15'></a>
### Track 15: Vision-Language Alignment with CLIP for Video
**Suggested Size**: Medium  
**Reference Module**: Vision & Language  

#### Problem Description
Searching for videos traditionally relies on manually curated metadata rather than visual content. This project explores zero-shot cross-modal retrieval by aligning video features with natural language text using a contrastive loss model reminiscent of CLIP. You will generate an embedding space where a raw video clip correctly aligns with its descriptive textual sentence allowing for robust, arbitrary text queries like "person chopping vegetables".

#### Dataset
- **EPIC-KITCHENS** (https://epic-kitchens.github.io).
- Utilize the multi-instance retrieval challenge metadata (https://www.codabench.org/competitions/12008/).

#### Minimum Objectives
1. **Baseline**: Use a pre-trained frozen video encoder and a frozen text encoder, mapping both directly through an untrained linear layer to attempt retrieval.
2. **Video & Text Encoders**: Utilize robust architectural foundations (e.g., TimeSformer/SlowFast for video, BERT/DistilBERT for text) to extract respective features.
3. **Contrastive Loss Alignment**: Train a dedicated alignment module (typically multi-layer perceptrons) acting on top of the encoders enforcing identical embedding mapping for positive visual/text pairs.
4. **Evaluation**: Benchmark text-to-video capabilities evaluating R@1, R@5, R@10 (top-K recall values). Explicitly define and test zero-shot capabilities acting on entirely unseen verbs/nouns.

#### Extra Objectives
- Transition from frozen encoders to full model fine-tuning noting improvements versus massive computational cost increases.
- Provide qualitative failure analysis demonstrating which verb/noun pairings consistently confuse the modality mapping algorithm.

---

<a id='project-16'></a>
### Track 16: Multimodal Action Recognition – Video + Audio
**Suggested Size**: Large  
**Reference Module**: Video Understanding  

#### Problem Description
In the physical world, sound is an incredible precursor and confirmation of action. Evaluating video alone discards critical discriminative markers—such as the sizzling of a pan heavily implying "frying." This project integrates parallel modalities to maximize recognition robustness, requiring you to engineer a dual-stream architecture analyzing both visual frame sequences and raw auditory wave outputs simultaneously.

#### Dataset
- **EPIC-KITCHENS** (https://epic-kitchens.github.io).
- Assess structures via the action recognition challenge parameters (https://www.codabench.org/competitions/13636/).

#### Minimum Objectives
1. **Baseline**: A foundational uni-modal Vision isolated 3D CNN (and alternatively, an isolated Audio 1D CNN baseline), benchmarking the independent limits of each sensor.
2. **Multimodal Encoders**: Deploy dual feature extraction: a 3D CNN iterating on visual frames and a 1D CNN processing extracted Librosa Mel-spectrograms.
3. **Fusion Architecture**: Implement a late-fusion embedding concatenation strategy bound by a centralized fully-connected classification head.
4. **Evaluation**: Compare fused multimodal classification accuracy against the isolated uni-modal baselines. Extract quantitative data mapping audio vs. video contribution metrics highlighting which modality leads which class.

#### Extra Objectives
- Execute a missing-modality ablation: artificially silence the audio or blackout the video mid-inference charting architectural robustness.
- Discard simple concatenation for an intermediate Cross-modal Attention mechanism (e.g., a Perceiver-like bottleneck module).

---

<a id='project-17'></a>
### Track 17: Egocentric Video + Gaze for Action Recognition
**Suggested Size**: Large  
**Reference Module**: Video Understanding  

#### Problem Description
Human attention is highly localized during procedural interaction. Where an actor chooses to physically look is frequently an early temporal indicator of their immediate subsequent action. By injecting physical biometric gaze tracking into standard egocentric video processing, you will force the network to prioritize human-attention clusters, drastically reducing background noise interference in hyper-complex environments.

#### Dataset
- **EGTEA Gaze+** (https://cbs.ic.gatech.edu/fpv/).
- Integrates video frames perfectly synchronized with 2D visual gaze tracking heatmaps.

#### Minimum Objectives
1. **Baseline**: Train a standalone 2D CNN (ResNet-18) natively on the video arrays entirely ignoring the supplied biometric gaze datasets.
2. **Biometric Encoders**: Establish dual networks: a 2D CNN mapping visual structures and a localized 2D CNN mapping Gaussian focal blobs originating from the gaze coordinates.
3. **Fusion Mechanism**: Stitch the distinct latent spatial embeddings utilizing concatenation mapping directly to the distinct action classifier.
4. **Evaluation**: Benchmark action accuracy systematically detailing if and where gaze data functionally improves physical prediction over the uni-modal baseline logic.

#### Extra Objectives
- Construct saliency map visualizations answering: does the mathematical visual-attention of the isolated baseline CNN inherently overlap with biological human gaze?
- Identify and document semantic bottlenecks, highlighting actions where human gaze provides zero additional contextual advantage.

---

<a id='project-18'></a>
### Track 18: State-Space Models (Mamba) for Mistake Detection
**Suggested Size**: Large  
**Reference Module**: Advanced Sequential Modeling  

#### Problem Description
As procedural timelines expand towards an hour of video, predicting temporal boundaries deteriorates rapidly due to memory constraints within Transformers and memory decay within LSTMs. This project pivots to the mathematics of continuous State-Space Models, relying on recent architectures like Mamba to traverse extraordinarily long rolling sequences memory-efficiently to locate highly anomalous temporal interactions—specifically, mistakes.

#### Dataset
- **Assembly101** (pre-extracted architectural features).
- Utilize a singular view mapping explicit per-frame annotations targeting distinct error bounds.

#### Minimum Objectives
1. **Baseline**: Implement and successfully replicate the foundational C2F error-detection baseline established in the original data paper.
2. **Mamba Protocol**: Utilize native SSM libraries (`mamba-ssm`) to replace temporal block processing with continuous state-space matrices.
3. **xLSTM Variant**: Construct the procedural logic utilizing the distinct xLSTM framework for multi-metric comparison.
4. **Verification**: Calculate explicit benchmark matrices demonstrating true long-sequence retention mapping Mamba vs xLSTM vs Baseline frameworks.

#### Extra Objectives
- Establish an asynchronous testing variant utilizing online temporal Transformers (e.g., TeSTra) mapping relative architectural advantages.
- Conclude how artificially compressing or expanding the input sequence horizons drastically manipulate performance dependencies between LSTMs and SSMs.

---

<a id='project-19'></a>
### Track 19: Transformer vs RNN for Procedural Video Understanding
**Suggested Size**: Medium  
**Reference Module**: Advanced Sequential Modeling  

#### Problem Description
Procedural sequential understanding relies on interpreting physical momentum. But does historical inertia genuinely require recurrent looping architectures, or does widespread parallel attention trump sequential logic? By putting highly configured LSTMs directly against heavily parallelized Transformers, this track explicitly maps their performance bounds charting frame-level classification across strict procedural interaction data.

#### Dataset
- **EGO4D Goal-Step** (https://github.com/facebookresearch/ego4d-goalstep).
- Apply evaluation via pre-extracted embeddings targeting the explicit online step-detection task sequence.

#### Minimum Objectives
1. **Baseline (RNN)**: Maintain a standard, optimized LSTM array navigating the temporal features iteratively.
2. **Attention Architecture (Transformer)**: Strip recursive blocks, utilizing absolute Positional Encoding chained with Multi-Head Self-Attention layers exclusively evaluating past episodic histories.
3. **Classification Mapping**: Both formats must predict current, distinct actions driven exclusively by past temporal momentum limits.
4. **Evaluation**: Correlate frame-level accuracy alongside strict latency parameters mapping parallel-inference speed vs sequential processing locks.

#### Extra Objectives
- Architect a hybridized variant attaching a native recursive layer mathematically preceding the temporal Transformer modules mapping performance benefits.
- Decipher the Transformer mapping documenting which distinct self-attention heads functionally act generically versus temporally specific.

---

<a id='project-20'></a>
### Track 20: Image & Language Representation Learning
**Suggested Size**: Medium  
**Reference Module**: Vision & Language  

#### Problem Description
Before expanding to video frameworks, establishing concrete grounding between static spatial imagery and complex syntactic language formatting remains paramount. Creating native zero-shot multi-modal capability forces neural networks to structurally associate words directly with pixel geometry patterns utilizing bidirectional contrastive parameters mirroring modern CLIP deployment methodologies.

#### Dataset
- **MS-COCO** (https://cocodataset.org/).
- Over 1.5M native image-text string correlations (utilizing a workable subset).

#### Minimum Objectives
1. **Baseline**: Use foundational unaligned Image (ViT) and Text (BERT) embeddings chained via a singular linear projection layer.
2. **Encoders**: Initialize pre-trained CNNs/ViTs paired perfectly with dedicated Transformer NLP logic.
3. **Contrastive Loss Alignment**: Generate the alignment sub-module trained explicitly across dense batches to minimize intra-modal Euclidean distance mapping.
4. **Evaluation**: Chart retrieval parameters indexing Image-to-Text capability yielding R@1, R@5, R@10 markers documenting pure zero-shot capability against external inputs.

#### Extra Objectives
- Expand parameters systematically testing frozen architectural encoders versus aggressively un-frozen end-to-end multi-modal fine-tuning.
- Investigate which semantic image/text pairings most frequently force catastrophic alignment failure.

---

<a id='project-21'></a>
### Track 21: Deep Reinforcement Learning for Frame Selection in Video
**Suggested Size**: Large  
**Reference Module**: Reinforcement Learning  

#### Problem Description
Analyzing raw 60 FPS video creates debilitating computational bloat, much of it completely redundant static background motion. To maximize temporal accuracy whilst simultaneously minimizing arithmetic execution logic, this project utilizes Reinforcement Learning to physically train an agent model. This agent actively curates and discards useless frames, acting dynamically inside the video array to retain only universally informative frames required for accurate downstream classification.

#### Dataset
- **UCF101** (https://www.crcv.ucf.edu/data/UCF101.php).

#### Minimum Objectives
1. **Baseline**: A standard ResNet-18 array ingesting universally defined/random uniform frame samples without dynamic selectivity logic.
2. **Agent Module**: A rapid EfficientNet-B0 format analyzing physical frames emitting binary predictions signaling immediate retention or discard protocols.
3. **Downstream Execution**: Apply the heavy baseline ResNet-18 exclusively targeting the fractional retained batch frame lists.
4. **Evaluation & Reward**: Deploy Reinforcement protocols (e.g., REINFORCE, DQN) manipulating the native Reward structure penalizing heavy frame counts while hyper-rewarding explicit downstream action accuracy.
5. **Metric Reporting**: Graph functional action accuracy explicitly mapped directly against physical frame retention limits charting agent effectiveness versus manual random baseline filtering.

#### Extra Objectives
- Switch foundational learning protocols alternating between DQN and PPO noting variance in training stabilization.
- Manipulate internal reward math experimenting heavily with linear vs exponential frame retention penalties.

---

<a id='project-22'></a>
### Track 22: Learn to Play Super Mario Bros with Deep Reinforcement Learning
**Suggested Size**: Medium  
**Reference Module**: Reinforcement Learning  

#### Problem Description
Reinforcement learning shifts neural prediction to dynamic interaction format where an agent observes an active state, selects an explicit directional action, and receives immediate deterministic environmental feedback. In this classic simulation logic project, you will build and iterate a Q-Learning agent required to observe raw moving pixel states and autonomously decode the mechanics necessary to successfully navigate Super Mario bounds.

#### Dataset
- **Super Mario Bros Environment** (Gym Implementation: https://github.com/yfeng997/MadMario).

#### Minimum Objectives
1. **Baseline**: Instantiate a purely generic exploration agent emitting fully randomized movements mapping rudimentary survival times establishing absolute minimum benchmarks.
2. **Perception Module**: Encode spatial simulation data via rapid 2D CNN layers distilling pixel values into numerical environmental embeddings.
3. **DQN Protocol**: Architect the generalized Deep Q-Network memory formatting predicting explicit scalar rewards mapped to specific environmental button commands.
4. **Evaluation**: Log explicitly charting massive cumulative training rewards over sequential iterations establishing maximum functional level/distance penetration limits.

#### Extra Objectives
- Modify the raw reward function actively hyper-rewarding lateral movement momentum over sheer point-score accumulation.
- Inject sequential historical data utilizing temporal RNN matrices mapped directly alongside spatial observations mitigating positional occlusion mechanics.

---

<a id='project-23'></a>
### Track 23: Align a Small LLM with GRPO for Strict Code or JSON Generation
**Suggested Size**: Medium  
**Reference Module**: Reinforcement Learning

#### Problem Description
While Large Language Models generate human-like text admirably, enforcing strict programmatic formatting limitations continuously causes catastrophic parser failures. Rather than relying on fuzzy "Helpfulness" reward models, this system applies Group Relative Policy Optimization (GRPO) to fine-tune generative outputs via a strict, programmatic feedback loop evaluating explicitly if an emitted code or JSON sequence flawlessly compiles and parses.

#### Dataset
- **Synthetic Logic Array**: Generatively create precise instructional requests mandating extreme syntactical accuracy (e.g., Python algorithms / rigid JSON objects).
- **Rule-Based Reward**: Replace Neural Rewards mathematically assigning scalar values entirely reliant on programmatic validation tools (e.g., `json.loads` / `ast.parse` returning absolute True/False markers).

#### Minimum Objectives
1. **Baseline**: Evaluate the base programmatic output capabilities of an off-the-shelf lightweight 2B parameter open LLM (yielding naturally poor syntactical adherence).
2. **Agent Selection**: Integrate the target open-weight model parameterizing memory limits strictly through established LoRA/PEFT architectural boundaries.
3. **GRPO Implementation**: Code the mathematical training loop exploiting active advantage computations distributed uniformly across minor programmatic generation groups.
4. **Evaluation**: Quantitatively establish final syntactical adherence limits determining pre- vs post-fine-tuning code Pass@1 metric stability parameters.

#### Extra Objectives
- Directly compare format-stabilization results generated by GRPO protocols against standard rigid Supervised Fine-Tuning mapping differences.
- Integrate an explicitly dualistic reward scaling protocol rewarding intermediate "reasoning" code tags generated prior to returning standard parsing block strings.


<a id='project-24'></a>
### Track 24: Cross-Modal Knowledge Distillation on VGGSound (Audio to Vision)
**Suggested Size**: Large  
**Reference Module**: Knowledge Distillation  

#### Problem Description
In many real-world edge settings, deploying multi-modal sensors is expensive or computationally expensive. This project addresses "modality hallucination" by distilling knowledge from a robust Video-based teacher into an Audio-only student model. This enables the resulting lightweight audio model to internally leverage visual context during inference—even when dealing exclusively with sound clips in production.

#### Dataset
- **VGGSound** (https://www.robots.ox.ac.uk/~vgg/data/vggsound/) (vedere anche qui: https://github.com/hhc1997/vggsound_download)

#### Minimum Objectives
1. **Baseline**: Train a standard audio encoder (e.g., Audio Spectrogram Transformer, AST) using purely audio spectrograms and cross-entropy loss.
2. **Teacher Model**: Fine-tune an image encoder (e.g., ResNet-50) on visual frames to act as the robust expert.
3. **Student Model**: Train the AST audio encoder using distillation loss mapped from the visual embeddings of the teacher, bridging the sensory gap.
4. **Evaluation**: Compare classification accuracy of the vision teacher, the audio baseline, and the distilled audio student. Produce a size (MB) and inference time (ms) comparison.

#### Extra Objectives
- Enforce strict dimensional alignment using Contrastive cross-modal distillation between audio and video embeddings.
- Implement a temporal 3D CNN or a Video Transformer as the visual teacher to account for longitudinal temporal dynamics.
- Explore and benchmark lightweight alternatives to AST.

---

## Groups

| Group Name | Members |
| :--- | :---: |
| LeMeCla | 3 |
| BAT 🦇 (Backpropagation Attention Team) | 3 |
| Deep Team | 3 |
| Justgood AI | 2 |
| FiCo | 3 |
| FlyNow | 3 |
| Overfittony | 3 |
| The Outliers 2.0 | 3 |
| Zero e Uno | 2 |
| DataMinds | 2 |
| TEAM CassiaBranca | 2 |
| Le larunghie | 2 |
| DataLost | 2 |
| EventHorizonTeam | 2 |
| Marte | 2 |
| G16 | 1 |
| G17 | 1 |
| G18 | 1 |
| G19 | 1 |
| G20 | 1 |
| G21 | 1 |
| G22 | 1 |
| G23 | 1 |
| G24 | 1 |
| G25 | 1 |
| G26 | 1 |
| G27 | 1 |
| G28 | 1 |
| G29 | 1 |
| G30 | 1 |
| G31 | 1 |
| G32 | 1 |
