
Project Brief:
# Multiaxial LLM Cascades for Adaptive Reasoning and Knowledge Retrieval

## Project Overview

This project investigates *multiaxial large language model (LLM) cascades*: adaptive inference systems in which escalation decisions occur across multiple capability dimensions rather than a single linear model hierarchy. Traditional cascade systems typically assume a one-dimensional progression from “small/cheap” to “large/expensive” models. In contrast, this work explores whether LLM failure states are *separable across capability axes*, particularly along:

* **Reasoning complexity**
* **Knowledge retrieval / search dependency**

The core hypothesis is that many model failures are not explained solely by overall capability scale, but instead emerge from distinct competency deficits. Under this framing, a sample may require escalation specifically along a reasoning axis, a retrieval/search axis, or both, enabling more efficient and interpretable routing strategies than monolithic cascades.

The project positions itself at the intersection of:

* adaptive inference and cascade systems,
* test-time compute allocation,
* agentic tool-use systems,
* retrieval-augmented reasoning,
* and failure-state modelling in LLM evaluation.

The broader aim is to contribute to future adaptive inference systems capable of selectively allocating reasoning depth, retrieval augmentation, or agentic behaviour according to the structure of individual tasks rather than relying solely on larger general-purpose models. Importantly, the work is aimed at low-to-moderate resource users (businesses, researchers, developers) who use LLMs out-of-the-box with post-training adaptations, rather than large AI companies with the resources to train or fine-tune LLMs themselves. High resource AI teams are capabale of re-training models to use tools in a specific agentic ecosystem, while in this work we train small post-hoc deferral models up to Deberta-v3 to provide the best performance-efficiency trade-off in resource limited environments or inference budgets.

## Experimental Framework

The current experimental pipeline performs exhaustive inference across datasets using multiple models representing positions on different capability axes. Outputs are stored as structured dataframe-based inference artifacts for later cascade simulation and analysis.

This approach enables:

* retrospective evaluation of arbitrary cascade policies,
* controlled ablation of routing strategies,
* synthetic routing experiments,
* analysis of escalation trajectories,
* and detailed modelling of separable failure regions.

The framework currently supports:

* axis-aware model definitions,
* positional hierarchy assignment,
* deterministic replay of cascades,
* confidence-aware escalation,
* and cross-validation over cascade policy configurations.

Experiments are currently executing in cloud notebook runtimes with supporting infrastructure and utility libraries cloned from private repositories.

## Current Experiments

### 1. Multiaxial Cascade Simulation

Offline simulation of cascades in which samples may escalate independently across reasoning and retrieval-oriented axes. This includes evaluation of:

* sequential escalation,
* conditional branching,
* axis-restricted escalation,
* and mixed escalation policies.

### 2. Separable Failure State Analysis

Investigation into whether incorrect responses cluster into identifiable capability regions rather than a single scalar “difficulty” continuum. Current analysis includes:

* cross-axis disagreement patterns,
* confidence calibration behaviour,
* and tie-biased routing experiments designed to preserve axis priors.


## Datasets

Current (non-final) benchmark datasets include:

* MMLU
* ARC-Challenge

These datasets were selected due to:

* broad diversity of question difficulty,
* heterogeneous reasoning requirements,
* differing knowledge dependence,
* and strong suitability for testing the separable failure state hypothesis.
* easy and comparable evaluation

Additional benchmarks may later be incorporated to isolate retrieval-heavy versus reasoning-heavy task structures more explicitly.

## Intended Contribution

The intended contribution is both conceptual and methodological:

1. **Conceptual:** reframing cascade systems as multidimensional adaptive processes rather than single-axis capability ladders.

2. ...

