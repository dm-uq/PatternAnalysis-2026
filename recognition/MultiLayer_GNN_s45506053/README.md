# Multi-Layer GNN for Facebook Page Classification
## COMP3710 Working Prototype Project in the Era of Artificial Intelligence
### Derek McKinney (dm-uq)

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
### 3.3 Data Audit and Class Distribution
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
### 3.6 Train / Validation / Test Splits
### 3.7 Leakage Prevention

## 4. Methods
### 4.1 Baseline Model
### 4.2 Multi-Layer GNN
### 4.3 Training Procedure
### 4.4 Confidence / Calibration Method
### 4.5 Reject-Option Decision Rule

## 5. Experimental Results
### 5.1 Training Behaviour
### 5.2 Classification Metrics
### 5.3 Baseline vs GNN
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
`prepare_facebook_data.py` must be<br>run once to create the raw file layout and
array schema expected by PyG's `FacebookPagePage` loader.<br>Subsequent loading
and preprocessing are performed through `dataset.py`.

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
