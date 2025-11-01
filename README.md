# Structured Knowledge Accumulation (SKA): Riemannian Neural Fields

### Overview
**Structured Knowledge Accumulation (SKA)** is a forward-only learning framework that models intelligence as the continuous organization of information over space and time.  
This repository introduces the **Riemannian Neural Field** architecture — a geometric extension of SKA — where learning is represented as smooth propagation across an information manifold driven by local entropy and neuron density. As the field evolves, the system continuously restructures its internal organization — discovering architecture dynamically rather than following a fixed network design.

The model extends the SKA Neural Network into a continuous field representation in which the **density of neurons is non-homogeneous**.  
This uneven distribution of neurons creates regions of varying informational activity across the manifold — areas of high density act as zones of complex processing, while lower-density regions provide diffusion and stabilization.  
Learning thus emerges as a structured flow through this non-uniform field, continuously adapting and organizing information in space and time.

---

### Core Idea
Traditional machine-learning systems rely on statistical error minimization.  
SKA reformulates learning as a **physical process**: information flows through a structured field, adapting locally while preserving global consistency.  
The field’s geometry evolves with the data itself, allowing the model to self-organize and **discover architecture dynamically**.

This design leads to:
- **Forward-only dynamics** — knowledge is never unlearned, only refined.  
- **Intrinsic geometry** — the structure of learning emerges from the data’s entropy and neuron density, not from **predefined architectures**.  
- **Dimensional universality** — the same framework operates in 3D, 4D, or higher-dimensional information spaces without modification.

---

### Conceptual Highlights
- **Information Manifold:**  
  Each learning process unfolds on a manifold whose local geometry reflects the organization of knowledge.  
  The manifold evolves with entropy, and the **non-homogeneous neuron density** acts as an internal weighting mechanism that shapes the curvature of learning and the flow of information.

- **Geodesic Learning Paths:**  
  Instead of minimizing a loss function, the system follows the shortest information paths—geodesics—through the evolving knowledge field.  
  These trajectories adapt naturally to the spatial variations in neuron density, concentrating learning where informational gradients are strongest.

- **Entropy-Based Adaptation:**  
  Learning is guided by changes in entropy, producing self-regularizing models that remain stable and interpretable over time.

- **Dynamic Architecture Discovery:**
  The geometry of the field enables the spontaneous formation of new structural patterns.
  Network organization emerges naturally from the evolving information manifold, eliminating the need for manual design or layer definition.

- **Riemannian Field Architecture:**  
  Neural activations are treated as a continuous field.  
  Connectivity and structure arise naturally from the information landscape and neuron distribution rather than from manual design.

- **The Duality of Local and Spatial Learning Paths:**  
  The SKA Neural Fields framework reveals a profound duality in how knowledge accumulates, governed by two complementary principles.  
  Locally, knowledge evolves **temporally** following the *first law of entropic least action* — minimizing entropy through time.  
  Globally, knowledge propagates **spatially** following the *second law of spatial least action* — optimizing the flow of information across entropy and density landscapes.  
  These dual principles operate in parallel, coupling time-based learning with space-based self-organization, uniting temporal cognition and spatial intelligence under a single formalism.


- **Scalability Across Dimensions:**  
  The same mechanism applies to spatial (3D), or higher-order cognitive fields (4D+), offering a unified view of learning across physical and informational domains.

---

### Why It Matters
Modern AI systems learn by statistical approximation; they require massive data and periodic retraining.  
SKA proposes a **continuous-time, geometry-aware alternative** that maintains stability and interpretability while adapting in real time.  
This could impact:
- **Autonomous reasoning systems**  
- **Financial and physical modeling**  
- **Dynamic cognitive simulations**  
- **Real-time learning agents**


---

###  **Python Libraries Used in the Paper**

| Library         | Role / Purpose                                                                                                                                                                                                                                            |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`pydec`**     | Core library for **Discrete Exterior Calculus (DEC)**. Used to represent the neural field as a simplicial complex, compute discrete differential forms (entropy gradients, cochains), and calculate **discrete geodesics** via weighted graph operations. |
| **`FEniCS`**    | Handles **finite element method (FEM)** operations for computing the **entropy field** and its gradient (\nabla h). Provides continuous interpolation and Jacobian-based mesh operations.                                                                 |
| **`geomstats`** | Provides **Riemannian metric tensor** operations (g_{ij}(r)), geodesic equations, and manifold geometry routines for validation and high-level metric computations.                                                                                       |
| **`noise`**     | Generates **Simplex noise** to create realistic neuron density fields (\rho(r)), introducing biological variability.                                                                                                                                      |
| **`PyTorch`**   | Performs **tensor computations** (elementwise ⊙, outer products ⊗, matrix operations) for SKA update equations like (Z⊙D′⊗X). Used for time-stepped entropy and weight updates.                                                                           |

---

#### **Key Insight**

> The implementation **requires all five libraries** — no single one can replace the others.
>
> * `pydec` provides discrete geometry (geodesic & mesh topology).
> * `FEniCS` provides continuous FEM interpolation and gradients.
> * `geomstats` defines the Riemannian metric and curvature operations.
> * `noise` generates spatial heterogeneity (ρ fields).
> * `torch` handles numerical SKA tensor updates.



These libraries together bridge **information geometry**, **finite element simulation**, and **entropy-driven neural learning**.

---


### Repository Contents
This repository provides:
- Conceptual overview of the SKA Riemannian Neural Field paradigm  
- Reference implementation structure (to be released separately)  
- Visualization framework outline for entropy, neuron density, and knowledge evolution  
- Research roadmap and references to ongoing work

No equations or proprietary algorithms are included here.  
For mathematical derivations and the complete theoretical background, see the forthcoming paper:

> *Bouarfa Mahi (2026). “Structured Knowledge Accumulation: Geodesic Learning Paths and Architecture Discovery in Riemannian Neural Fields.”*

---

### Status
- **Phase 1:** Theoretical formulation (completed)  
- **Phase 2:** 3D/4D prototype simulations (in progress)  
- **Phase 3:** Cross-domain applications (planned for 2026)

---

### Contact
For collaboration or research inquiries:

**Bouarfa Mahi**  
Quantiota Research Lab  
📧 info@quantiota.org  
🌐 [https://github.com/quantiota](https://github.com/quantiota)

---

### License
All materials are released under **CC BY-NC-ND 4.0** —  
free for academic and non-commercial use with attribution, no derivative works.

---

> *This repository represents a new direction in geometric AI: learning not as optimization, but as the natural flow of knowledge through structured information space with non-homogeneous neuron density and evolving geometry.*
