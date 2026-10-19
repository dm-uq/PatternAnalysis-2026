# Multi-Layer GNN for Facebook Page Classification
## COMP3710 Working Prototype Project in the Era of Artificial Intelligence
### Derek McKinney (dm-uq), SN: 45506053

## 1. Project Overview
### 1.1 Problem and Intended User
### 1.2 Algorithm / Working Principle
### 1.3 Project Figure / Overview Diagram

## 2. Feasibility Review
### 2.1 User Need, Scope and Acceptance Criteria
### 2.2 Model Choice, Dataset and Difficulty
### 2.3 Relevant Course Concepts and Literature
### 2.4 Preliminary Feasibility Evidence
### 2.5 Risks, Compute Budget and Fallback
### 2.6 Next Planned Experiment

## 3. Data and Experimental Protocol

### 3.1 Facebook Large Page-Page Network

The project uses the Facebook Large Page-Page Network provided through the
course-linked<br>dataset repository and represented using PyTorch Geometric
(PyG). The graph contains 22,470<br>Facebook pages represented as nodes, with
128-dimensional node features and four page classes:<br>television show,
government, company, and politician.

The processed PyG graph contains 342,004 edge entries and is treated as a
single graph for<br>semi-supervised node classification. Dataset provenance and
the corresponding PyG resources are<br>listed in the References section.

### 3.2 Graph Representation and Features

The dataset is represented as a single PyG `Data` graph containing 22,470
nodes, 342,004 stored<br>edge entries, 128 features per node, and one class
label per node.<br>
After compatibility conversion and loading by PyG, the core dataset tensors
(before experiment-<br>specific masks are added) have shapes:

- `x`: `[22470, 128]`
- `edge_index`: `[2, 342004]`
- `y`: `[22470]`

The graph is undirected, includes self-loops, and contains no isolated nodes.
<br>The supplied node features are stored as 128-dimensional dense
floating-point values, derived from<br>a higher-dimensional sparse bag-of-words
description representation using truncated SVD.

Inspection of the feature matrix gave a mean of approximately 0.0103 and
standard deviation of<br>0.2551, with values ranging from approximately -2.04
to 3.00. No zero-valued feature entries were<br>observed.<br>
Given that the supplied features are already a transformed dense
representation, additional row<br>normalisation was not presumed to be a
required preprocessing step.

### 3.3 Data Audit and Class Distribution

The four node classes are moderately imbalanced but all are well represented:

| Class           | Nodes | Proportion |
|-----------------|------:|-----------:|
| Television show | 6,495 |     28.90% |
| Government      | 6,880 |     30.62% |
| Company         | 5,768 |     25.67% |
| Politician      | 3,327 |     14.81% |

No missing class labels were observed.

Node degree is strongly right-skewed. The observed minimum degree was 1, the
median was 7, the<br>mean was approximately 15.22, and the maximum was 709.
This indicates that most pages have<br>relatively modest connectivity while a
smaller number of highly connected nodes form a long upper tail.

Edge homophily was approximately 0.8854, indicating that connected nodes very
frequently share the<br>same class label. This provides an initial empirical
reason to expect that graph structure may contain<br>useful classification
information beyond the node features alone.

### 3.4 Pre-processing

The original Assignment-Specification-prescribed PyG `FacebookPagePage`
download endpoint was<br>unavailable.
Instead, the course-linked 128-dimensional `facebook.npz` archive was obtained
and<br>then converted to the schema expected by the current PyG loader:

- `X` → `features`
- `y` → `target`
- `edges_x` and `edges_y` → combined `edges`
- `page_name` retained as node-name metadata for later analysis

No feature values, labels, or graph topology were otherwise modified.

### 3.5 Transductive / Inductive Setting

An inductive node-classification setting aims to learn a mapping that can be
applied to nodes not<br>present during training. In contrast, transductive
classification infers labels for nodes already<br>contained within the observed
graph, allowing information from unlabelled nodes to contribute to<br>the task.

This project uses the latter setting. The complete Facebook graph remains
available during training,<br>including the features and connectivity of nodes
assigned to the training, validation, and test splits.<br>The masks instead
control access to their class labels, such that training labels drive 
optimisation,<br>validation labels support model selection, and test labels are
withheld until the final evaluation.

For the GNN, this means that held-out nodes can still contribute neighbourhood
and graph-structure<br>information during message passing, while the prediction
task remains the inference of withheld<br>labels for existing nodes in the same
graph.

### 3.6 Train / Validation / Test Splits

Nodes are divided using a reproducible stratified 60/20/20 split, with a fixed
seed value of 808.<br>
The resulting Boolean masks form a complete, non-overlapping partition of the
data nodes while<br>approximately preserving class proportions.

### 3.7 Leakage Prevention

The train, validation, and test masks are non-overlapping, and together, cover
all graph nodes.<br>
During optimisation, cross-entropy loss is calculated only for nodes selected
by the training mask.

Validation labels do not contribute to gradient updates and are used only for
model selection.<br>
The model checkpoint with the lowest validation loss is retained.<br>
Test labels are excluded from training/validation, and do not inform checkpoint
selection. They are<br>accessed only for final evaluation, once model
development and selection have been completed.

Because this is a transductive task, held-out nodes remain part of the observed
graph, so their<br>features and edges may contribute to message passing. The
relevant leakage boundary is<br>therefore how ground-truth labels are used,
rather than whether the held-out nodes are structurally<br>visible. While
training labels drive optimisation, and validation labels support model 
selection, the<br>test labels remain withheld until final evaluation.

## 4. Methods
### 4.1 Baseline Model

The baseline is a feature-only MLP using the 128-dimensional node features
without graph<br>connectivity.<br>
The model consists of a 128 → 64 linear layer, a ReLU activation, dropout
(`p=0.5`), and a 64 → 4<br>output layer producing raw class logits.

### 4.2 Graph Convolutional Network (GCN)

The graph model is a two-layer Graph Convolutional Network (GCN). It retains
the same<br>128 → 64 → 4 dimensional structure, ReLU activation, and dropout
(`p=0.5`) used by the MLP<br>baseline, but replaces the linear layers with
PyG `GCNConv` layers.

Each graph-convolution layer uses the node features together with
`edge_index`, allowing<br>neighbouring-node information to contribute to the
learned representation. The GCN retains the<br>same number of hidden-layer
units, ReLU activation, and dropout setting as the MLP baseline,<br> while
replacing the feature-only linear transformations with graph convolutions that
incorporate<br>neighbourhood structure.

### 4.3 Training Procedure

The MLP and GCN are both trained using cross-entropy loss and the Adam
optimiser (`lr=0.001`,<br>`weight_decay=0`) under the same 60/20/20 stratified
split. Both models use a hidden width of 64<br>and dropout probability of 0.5.

Training is full-batch, with only training-node labels contributing to
optimisation. Validation is<br>performed after each training epoch, and the
checkpoint with the minimum validation loss is retained.

The MLP training budget was bounded at 2,000 epochs. An initial 2,000-epoch GCN
run attained its<br>lowest validation loss at the final epoch, so the budget
was subsequently extended to 4,000 epochs.<br>This provided sufficient margin
to capture the later training behaviour and the observed validation-<br>loss
minimum.

### 4.4 Confidence / Calibration Method
### 4.5 Reject-Option Decision Rule

## 5. Experimental Results
### 5.1 Training Behaviour
#### 5.1.1 Baseline MLP

The initial MLP training budget was increased progressively while validation
loss was still improving,<br>taking advantage of the relatively low cost of
each full-batch epoch. The maximum was eventually set<br>to 2,000 epochs, which
was sufficient to extend well beyond the observed validation-loss minimum<br>
and therefore capture the relevant training behaviour, with some margin.

A clean local run (employing 2,000 epochs) completed in approximately 19.5
seconds, with the<br>minimum validation cross-entropy loss of 0.55159 occurring
at epoch 1,545.

Training and validation loss both decreased rapidly over the first few hundred
epochs. Thereafter,<br>training loss continued to decrease steadily, while
validation loss improved more gradually, before<br>reaching a broad minimum.
Beyond the best validation checkpoint, training loss continued to fall<br>while
validation loss increased slightly, indicating mild overfitting rather than
severe divergence.

Validation accuracy followed a slightly different pattern, improving rapidly
during early training before<br>increasing more gradually towards approximately
80%. Small improvements then continued, even<br>as validation loss largely
plateaued.

This difference between validation accuracy and validation loss is relevant to
the later confidence<br>investigation. 
Accuracy records only whether the highest-scoring class is correct, whereas
cross-<br>entropy is also sensitive to prediction confidence. The differing
behaviour of these metrics therefore<br>motivates later analysis of how
confidence relates to correctness, and whether improved model<br>performance is
accompanied by better-calibrated predictions. However, the curves alone are not
<br>sufficient to establish model miscalibration or overconfidence.

#### 5.1.2 Initial GCN

The initial GCN run began with the same 2,000-epoch maximum as the MLP,
but its minimum<br>validation loss occurred at epoch 2,000. This indicated that
the useful training region had not yet<br>been bounded, and so the maximum was
increased to 4,000 epochs for a clean follow-up run.

The 4,000-epoch run completed locally in approximately 777.2 seconds
(12.95 minutes). Minimum<br>validation cross-entropy loss was 0.22775 at epoch
3,244, leaving approximately 750 subsequent<br>epochs in which validation loss
no longer improved.
Validation accuracy rapidly rose above 90%<br>during the earlier epochs before
improving more gradually to approximately 93–94%. Beyond<br>the best validation
checkpoint, classification accuracy remained largely stable while training
<br>loss continued to decline, providing sufficient evidence that the
4,000-epoch budget captured the<br>relevant later training behaviour.

### 5.2 Classification Metrics
### 5.3 Baseline vs GCN

Keeping the hidden-layer size, ReLU activation, dropout, optimiser settings,
and data split aligned<br>with the MLP provided a controlled first comparison
in which the principal architectural point of<br>difference was the
incorporation of graph structure. Under these conditions, the GCN produced<br>
substantially stronger validation performance than the feature-only MLP.
Validation accuracy<br>increased from approximately 80% for the MLP to
approximately 93–94% for the GCN, while the<br>selected validation
cross-entropy loss decreased from 0.55159 to 0.22775.

This improvement came at substantially greater computational cost, however, as
the clean MLP<br>run only required approximately 19.5 seconds locally, compared
with 12.95 minutes for the<br>4,000-epoch GCN run. These validation results
therefore indicate a strong preliminary<br>performance advantage for the GCN.
Whether this advantage is sufficient to justify the GCN's<br>additional
computational cost will be assessed through held-out test-set evaluation and
formal<br>comparative resource profiling.

### 5.4 t-SNE / UMAP Embedding Analysis
### 5.5 Confidence and Ambiguity Analysis
### 5.6 Resource Profiling

## 6. Failure Case Analysis
### 6.1 Failure Case 1
### 6.2 Failure Case 2
### 6.3 Failure Case 3
[...up to 5]

## 7. Discussion
### 7.1 Does Graph Context Improve Classification?
### 7.2 Does Confidence Correlate with Correctness?
### 7.3 Cost vs Operational Benefit
### 7.4 Limitations

## 8. Engineering Recommendation

## 9. Usage and Reproducibility
### 9.1 Dependencies and Exact Versions

The implementation was developed using Python 3.12 and PyTorch Geometric
2.8.0.post1.<br>Exact package versions used for the reported experiments are
provided below / in the accompanying<br>environment specification.

### 9.2 Training
### 9.3 Prediction / Evaluation
### 9.4 Example Inputs and Outputs
### 9.5 Reproducing Results

When starting from the course-linked `facebook.npz` archive,
`prepare_facebook_data.py` must be<br>run once to create the raw file layout
and array schema expected by PyG's `FacebookPagePage` loader.<br>Subsequent
loading and preprocessing are performed through `dataset.py`.

## 10. Repository Structure

## 11. Artificial Intelligence Usage Disclosure

### ChatGPT:

Produced an initial README 'skeleton' from my own Assessment
Specification-derived checklist txt file.

Adapted the course-linked data source approach to data file preparation, to
inform the prepare_facebook_data<br>compatibility script for converting the
source data to the same schema used in PyG examples.<br>
Produced a 'terminal py script' command to obtain the correct source data types
for type annotations, within<br>the project code.

Provided cast and assertion suggestions that were adapted for
add_stratified_masks, to allow for the resolution<br>of IDE type-checker
warnings, without compromising on the project's 'rigorous' typing conventions.

Provided feasible argument typing suggestions for the train_mlp_epoch and
evaluate_mlp function signatures,<br>from which, I chose those the most
appropriate for the project's typing conventions.

Ensured correctness of the Markdown formatting 'syntax' for my progressive
README updates, prior to the<br>final refinements by myself.

## 12. References

1. B. Rozemberczki, C. Allen and R. Sarkar,
   "Multi-scale Attributed Node Embedding", 2019.

2. Facebook Page-Page dataset preparation repository:
   https://github.com/gayanku/facebook_page-page_dataset

3. PyTorch Geometric documentation:
   FacebookPagePage dataset.

4. PyTorch Geometric tutorial:
   Node Classification with Graph Neural Networks.

5. PyTorch Geometric tutorial:
   Dataset Splitting.
