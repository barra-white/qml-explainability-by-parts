# Component Based Quantum Machine Learning Explainability

## Introduction

Explainable Machine Learning (ML) models are designed to provide transparency and insight into their decision-making process. Explaining how ML models come to their prediction is critical in fields such as healthcare and finance, as it:

- Provides insight into how models can help detect bias in predictions
- Helps comply with GDPR compliance in these fields

Quantum Machine Learning (QML) leverages quantum phenomena such as entanglement and superposition, offering:

- Potential for computational speedup
- Greater insights compared to classical ML

However, QML models also inherit the 'black-box' nature of their classical counterparts, requiring the development of explainability techniques to be applied to these QML models to help understand why and how a particular output was generated.

## Proposed Framework

This paper will explore the idea of creating a modular, explainable QML framework that splits QML algorithms into their core parts, such as:

- Feature maps
- Variational circuits (ansatz)
- Optimizers
- Kernels
- Quantum-classical loops

Each component will be analyzed using explainability techniques, such as:

1. Leave-One-Out (LOO)
2. Permutation Importance
3. Accumulated Local Effects (ALE)
4. SHapley Additive exPlanations (SHAP)

By combining insights from these parts, the paper aims to infer explainability to the overall QML model.

QML models will be compared to classical ML models, all trained on the Pima Indians Diabetes dataset. The QML models are implemented on IBM's Qiskit platform.
