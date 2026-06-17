# Sentience Salience Probe

### Research Question / Objective

Do language models encode an implicit hierarchy of sentience or moral worth in their internal representations? 

Specifically, do they systematically distinguish between humans, vertebrates, invertebrates, and plants in ways that reflect perceived sentience, and do these representations influence welfare-related outputs?

### Why It Matters

AI systems are increasingly used in domains affecting animals, such as agriculture, conservation, and policymaking. 

Yet AI evaluations rarely examine whether models implicitly discount animal interests. If hidden sentience hierarchies exist, they could shape decisions and recommendations at scale. 

This project explores a neglected intersection of AI safety, interpretability, and animal welfare.

### Proposed Approach

Using TransformerLens and open-source models (e.g., Pythia), I create a dataset containing references to humans, mammals, birds, fish, invertebrates, and plants across welfare-focused and neutral contexts. 

I analyse hidden activations using PCA, UMAP, representational similarity analysis, and linear probes to determine whether a sentience-related signal exists. 

I run lightweight causal tests, such as activation patching, to assess the influence of activation patches on model outputs.

### Potential Impact

This project could provide an initial framework for auditing AI systems for hidden assumptions about whose welfare matters. 

Outputs include a short technical report (4–6 pages), visualisations, reproducible code, all available at this public GitHub repository. 

Findings will be shared through AI safety communities, GitHub, and animal advocacy networks.
