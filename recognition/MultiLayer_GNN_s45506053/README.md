# Multi-Layer GNN for Facebook Page Classification
## COMP3710 Working Prototype Project in the Era of Artificial Intelligence
### Derek McKinney (dm-uq), SN: 45506053

## 1. Project Overview
### 1.1 Problem and Intended User
### 1.2 Algorithm / Working Principle
### 1.3 Project Figure / Overview Diagram

## 2. Feasibility Review
### 2.1 User Need, Scope and Acceptance Criteria

The intended user is a social-network analyst or reviewer responsible for
categorising public<br>Facebook pages. Accurate page-type classification can
support downstream trust & safety and<br>network-analysis workflows by helping
analysts identify anomalous or potentially misrepresented<br>pages, prioritise
cases for review, and interpret patterns of coordination across the network.
Manual<br>classification does not scale well to large networks, while fully
automated classification may produce<br>incorrect but highly confident
predictions. The project therefore investigates whether incorporating<br>graph
connectivity through a Graph Convolutional Network (GCN) provides sufficient
practical<br>benefit over a feature-only MLP baseline to justify its additional
complexity and computational cost.

The scope is semi-supervised classification of nodes in the supplied Facebook
Large Page-Page<br>Network into four existing classes: television show,
government, company, and politician. The<br>project uses a transductive
setting, where the complete graph is available during training and access<br>to
class labels is controlled by the train, validation, and test masks. An initial
two-layer GCN<br>provides the starting graph model, after which a limited set
of motivated architectural or training<br>choices may be investigated where
they could improve classification performance, confidence<br>behaviour, or
computational efficiency. Exhaustive hyperparameter optimisation is outside
the<br>project scope.

The prototype does not attempt to discover new classes or demonstrate
generalisation to previously<br>unseen graphs. Resource measurements on the
supplied graph will be used to assess prototype<br>feasibility and inform
discussion of likely scalability constraints, but production-scale
deployment<br>on substantially larger networks is also outside the project
scope.

The GCN will be considered a stronger candidate for further development than
the baseline MLP<br>model if it meets the following acceptance criteria:

1. **Classification benefit:** it provides a material improvement over the MLP
   baseline on held-out<br>classification performance, particularly macro-F1 and
   per-class recall, rather than improving<br>only overall accuracy.

2. **Decision-confidence benefit:** at a comparable level of automated coverage
   or human-review<br>workload, the GCN should achieve a lower error rate among
   automatically accepted predictions<br>than the MLP. Conversely, for a
   specified acceptable error level, it should permit equal or<br>greater
   automated coverage. The reject-option analysis will therefore compare the
   models<br>through their risk–coverage trade-off, rather than simply
   comparing raw confidence scores.

3. **Error concentration:** remaining errors should be preferentially
   concentrated among<br>low-confidence or otherwise ambiguous cases, such
   that routing uncertain predictions to human<br>review captures a meaningful
   proportion of classification errors.

4. **Operational feasibility:** the GCN should provide its classification and
   decision-confidence<br>benefits without a disproportionate increase in
   memory use, inference latency, or model<br>complexity relative to the MLP.
   The supplied graph should remain comfortably tractable on the<br>available
   hardware, while measured resource use and expected scaling behaviour should
   not<br>reveal an obvious barrier to applying the approach to moderately
   larger graphs.

If these criteria are not met, the simpler MLP remains a viable fallback and
the additional<br>graph-model complexity would not be recommended solely on
the basis that a GCN can be trained<br>successfully.

### 2.2 Model Choice, Dataset and Difficulty

The project uses the Facebook Large Page-Page Network, represented as a single
PyTorch<br>Geometric graph containing 22,470 Facebook page nodes, 342,004
stored edge entries,<br>128-dimensional node features, and four target classes:
television show, government, company,<br>and politician. Preliminary data
analysis found edge homophily of approximately 0.885, meaning<br>connected
nodes frequently share the same class label.

The graph model is a multi-layer Graph Convolutional Network (GCN). An initial
two-layer GCN<br>uses the same 128 → 64 → 4 dimensional structure, ReLU
activation, and dropout setting as the<br>feature-only MLP baseline, but
replaces the linear layers with graph-convolution layers that<br>incorporate
neighbourhood information through the graph structure. The observed level
of<br>homophily (0.885) suggests a practical reason to investigate whether this
additional graph<br>information can improve classification over the
feature-only baseline.

The project is classified in the **Normal Difficulty** band. The task goes
beyond straightforward<br>feature-only classification because it uses a
transductive semi-supervised<br>setting and requires graph connectivity to be
exploited without leaking held-out labels. The model<br>must be evaluated not
only for classification performance, but also for confidence behaviour,<br>
reject-option usefulness, resource cost, and representative failure
cases.

### 2.3 Relevant Course Concepts and Literature

The project directly applies course material on graph representation and Graph
Neural Networks. A<br>graph is represented through its nodes, edges, and node
features, while GNN message passing<br>combines a node's own representation
with information aggregated from its neighbours. In a GCN,<br>this allows graph
structure to contribute directly to the learned node representation rather
than<br>relying only on the 128-dimensional page features used by the MLP
baseline.

The task also reflects the course treatment of semi-supervised node
classification. Successive<br>graph-convolution layers increase the receptive
field by incorporating information from progressively<br>more distant
neighbourhoods. However, the course material also identifies limited GNN depth
and<br>oversmoothing as important design considerations. This provides a reason
to begin with a shallow<br>two-layer GCN and to treat any further increase in
depth as a controlled experiment, rather than<br>assuming that a deeper network
will perform better.

The initial GCN design is supported by Kipf and Welling's formulation of
semi-supervised<br>classification with Graph Convolutional Networks. The
planned confidence and reject-option<br>analysis is informed by El-Yaniv and
Wiener's selective-classification framework, which<br>characterises the
trade-off between prediction risk and automated coverage. Geifman and
El-Yaniv<br>apply selective classification to deep neural networks,
providing further support for evaluating the<br>GCN and MLP through their
risk–coverage behaviour. These ideas provide a basis for assessing<br>the
models at matched levels of automated coverage or acceptable prediction risk,
rather than<br>simply comparing raw confidence scores.

### 2.4 Preliminary Feasibility Evidence

Initial data inspection and model experiments indicate that the proposed
approach is feasible on<br>the supplied dataset and available local hardware. The
Facebook graph has been successfully<br>loaded and audited, and the
reproducible stratified train, validation, and test masks are in place.<br>
Edge homophily is approximately 0.885, meaning that most connected node pairs
share the same<br>class label. This is substantially above the roughly 0.265
same-class rate expected from the<br>observed class proportions under simple
random mixing, providing preliminary evidence that<br>graph connectivity
contains useful class-related structure.

The feature-only MLP baseline has been trained successfully on the local GPU.
At its selected<br>minimum-validation-loss checkpoint, it achieved
approximately 79.7% validation accuracy with a<br>validation cross-entropy loss
of 0.55159. A clean 2,000-epoch run completed in approximately<br>18.5 seconds.

The initial two-layer GCN has also been trained successfully under the same
data split, with<br>closely matched architectural settings and the same Adam
optimisation configuration<br>(`lr=0.001`, `weight_decay=0`). At its selected
checkpoint, it achieved approximately 93.7%<br>validation accuracy with a
validation cross-entropy loss of 0.22795. A clean 4,000-epoch run<br>completed
locally in approximately 9.74 minutes.

These preliminary results suggest that incorporating graph structure can
provide a substantial<br>classification benefit over the feature-only baseline,
while remaining computationally practical on<br>the available local hardware.
The results reported here are based on validation performance<br>only. An early
MLP test evaluation was executed, but its results have not informed model<br>
development or selection. Final comparative test evaluation will be
performed only after the<br>remaining model-development decisions have been
completed.

### 2.5 Risks, Compute Budget and Fallback

The main project risks concern model behaviour rather than basic computational
feasibility.<br>Although the initial GCN shows substantially stronger
validation performance than the MLP<br>baseline, this does not guarantee that
higher-confidence predictions are consistently more likely<br>to be correct
than lower-confidence predictions. Graph connectivity may also be misleading
for<br>atypical or heterophilous nodes, where neighbouring pages belong to
different classes. Further<br>increases in GCN depth could additionally
introduce diminishing returns or oversmoothing, so<br>any architectural
refinement will be limited and evaluated experimentally rather than
assuming<br>that a deeper model is preferable.

The current compute requirement is modest. A clean 4,000-epoch GCN run
completed locally in<br>approximately 9.74 minutes, making a small number of
controlled follow-up experiments practical<br>on the available local GPU.
More capable university lab or Rangpur GPUs are available as<br>contingency
resources if later experiments require substantially greater compute.
Formal<br>resource profiling will compare parameter count, peak GPU memory use,
and inference latency,<br>such that the GCN's additional cost can be assessed
against its classification and<br>decision-confidence benefits.

Given the project uses a transductive setting and evaluates a single supplied
graph, its results<br>will not establish generalisation to previously unseen
graphs or production-scale social networks.<br>If further GCN refinement does
not produce a meaningful improvement, the initial two-layer GCN<br>remains a
valid graph model for final evaluation. If the additional
graph-model complexity is not<br>justified by the final classification,
confidence, and resource results, the simpler feature-only MLP<br>provides a
further operational fallback.

### 2.6 Next Planned Experiment

The next experiment will refine the initial GCN through a focused set of
validation-driven design<br>choices. Candidate changes will examine factors
such as network depth, hidden width, dropout,<br>and regularisation where they
can meaningfully affect classification performance, confidence<br>behaviour, or
computational efficiency. Changes in depth will be evaluated in particular
against<br>the trade-off between a larger receptive field and the risk of
oversmoothing.

Candidate GCN configurations will be compared using validation loss and
classification metrics<br>under the same data split and training protocol. Once
the preferred configuration has been<br>selected, the model-development choices
will be frozen before final comparative evaluation on<br>the held-out test set.

The final comparison will extend beyond overall accuracy to include macro-F1,
per-class<br>precision and recall, and AUROC. Confidence behaviour will also be
examined through a<br>reject-option analysis that compares prediction risk and
automated coverage for the MLP and<br>GCN. A threshold will be selected using
validation data before being applied to the final test<br>evaluation.

Resource profiling will compare parameter count, peak GPU memory use, and
full-graph<br>inference latency under the same hardware conditions. A
two-dimensional t-SNE or UMAP<br>visualisation of the learned node
representations will provide a qualitative examination of class<br>structure,
including clustering, overlap, and ambiguous regions in the embedding
space.<br>Together with representative failure-case analysis, these
results will determine whether the<br>GCN's additional complexity provides
sufficient practical benefit over the feature-only MLP<br>baseline.

## 3. Data and Experimental Protocol

### 3.1 Facebook Large Page-Page Network

The project uses the Facebook Large Page-Page Network provided through the
course-linked<br>dataset repository and represented using PyTorch Geometric
(PyG). The graph contains 22,470<br>Facebook pages represented as nodes, with
128-dimensional node features and four Facebook<br>page classes: television
show, government, company, and politician.

The processed PyG graph contains 342,004 edge entries and is treated as a
single graph for<br>semi-supervised node classification. Dataset provenance and
the corresponding PyG resources<br>are listed in the References section.

### 3.2 Graph Representation and Features

The dataset is represented as a graph \(G=(V,E,X)\), where \(V\) denotes the
nodes, \(E\) the graph<br>edges, and \(X\) the node-feature matrix. In PyTorch
Geometric, this is stored as a single `Data`<br>graph containing 22,470 nodes,
342,004 stored edge entries, 128 features per node, and one class<br>label per
node.<br>
After compatibility conversion and loading by PyG, the core dataset tensors
(before<br>experiment-specific masks are added) have shapes:

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

Each `GCNConv` layer performs neighbourhood aggregation through the graph
edges together with a<br>learned feature transformation, allowing each node
representation to incorporate information from<br>connected neighbours.
With two graph-convolution layers, the model can incorporate information<br>from
neighbourhoods extending approximately two hops from each node.

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
to capture the later training behaviour and the observed<br>validation-loss
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

A clean local run (employing 2,000 epochs) completed in approximately 18.5
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
Accuracy records only whether the highest-scoring class is correct, whereas<br>
cross-entropy is also sensitive to prediction confidence. The differing
behaviour of these metrics<br>therefore motivates later analysis of how
confidence relates to correctness, and whether improved<br>model performance is
accompanied by better-calibrated predictions. However, the curves alone are<br>
not sufficient to establish model miscalibration or overconfidence.

#### 5.1.2 Initial GCN

The initial GCN run began with the same 2,000-epoch maximum as the MLP,
but its minimum<br>validation loss occurred at epoch 2,000. This indicated that
the useful training region had not yet<br>been bounded, and so the maximum was
increased to 4,000 epochs for a clean follow-up run.

The 4,000-epoch run completed locally in approximately 584.5 seconds
(9.74 minutes). Minimum<br>validation cross-entropy loss was 0.22795 at epoch
3,614, leaving approximately 386 subsequent<br>epochs in which validation loss
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
At the selected<br>minimum-validation-loss checkpoints, validation accuracy
was approximately 79.7% for the MLP<br>and 93.7% for the GCN, while validation
cross-entropy loss decreased from 0.55159 to 0.22795.

This improvement came at substantially greater computational cost, however, as
the clean MLP<br>run only required approximately 18.5 seconds locally, compared
with 9.74 minutes for the<br>4,000-epoch GCN run. These validation results
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
`prepare_facebook_data.py` must<br>be run once to create the raw file layout
and array schema expected by PyG's `FacebookPagePage`<br>loader. Subsequent
loading and preprocessing are performed through `dataset.py`.

## 10. Repository Structure

## 11. Artificial Intelligence Usage Disclosure

### ChatGPT:

Produced an initial README 'skeleton' from my own Assessment
Specification-derived checklist txt<br>file.

Adapted the course-linked data source approach to data file preparation, to
inform the<br>prepare_facebook_data compatibility script for converting the
source data to the same schema used<br>in PyG examples.<br>
Produced a 'terminal py script' command to obtain the correct source data types
for type annotations,<br>within the project code.

Provided cast and assertion suggestions that were adapted for
add_stratified_masks, to allow for the<br>resolution of IDE type-checker
warnings, without compromising on the project's 'rigorous' typing<br>
conventions.

Provided feasible argument typing suggestions for the train_mlp_epoch and
evaluate_mlp function<br>signatures, from which, I chose those the most
appropriate for the project's typing conventions.

Provided suggestions for harmonising the current plotting approach, based on
examples I had used<br>in data analysis exercises, previously, which I then
adapted into a coherent notebook plotting style<br>for the project.

Provided suggestions for persisting experiment configuration, training history,
selected model<br>checkpoints, and plot outputs, which I adapted for the MLP
and GCN experiment notebooks.

Checked my train.py functionality-generalising refactor for 'hangover errors',
and suggested the use<br>of a forward-pass lambda function parameter, which I
adopted.

Ensured correctness of the Markdown formatting 'syntax' for my progressive
README updates,<br>prior to the final refinements by myself.

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

6. R. El-Yaniv and Y. Wiener,
   "On the Foundations of Noise-Free Selective Classification",
   Journal of Machine Learning Research, 2010.

7. Y. Geifman and R. El-Yaniv,
   "Selective Classification for Deep Neural Networks",
   Advances in Neural Information Processing Systems, 2017.

8. Meta,
   "How We Review Content", 2020.
