# Why Simplex Noise for Neuron Density Fields

## Overview

The Riemannian SKA Neural Fields framework requires spatially coherent density fields ρ(r) in arbitrary dimensions. Simplex noise is the **unique** choice that enables this.

## The Problem

Building a density substrate that:
- Has smooth gradients ∇ρ for metric tensor construction
- Shows no directional artifacts
- Scales computationally to higher dimensions
- Maintains spatial coherence in 4D and 5D

## Why Simplex Noise?

Ken Perlin invented Simplex noise specifically because **Perlin noise fails in high dimensions**.

| Property | Perlin Noise | Simplex Noise |
|----------|--------------|---------------|
| Grid structure | Hypercube | Simplex |
| Complexity | O(2ⁿ) | O(n²) |
| Directional artifacts | Yes (worsens with D) | No |
| Gradient continuity | Degrades | Preserved |

## Simplex Geometry

| Dimension | Simplex | Vertices |
|-----------|---------|----------|
| 2D | Triangle | 3 |
| 3D | Tetrahedron | 4 |
| 4D | 5-cell (pentachoron) | 5 |
| 5D | 5-simplex | 6 |
| nD | n-simplex | n+1 |

Vertices grow **linearly** with dimension (n+1).

Hypercube vertices grow **exponentially** (2ⁿ).

## Alternatives Considered

| Alternative | Why It Fails |
|-------------|--------------|
| Perlin noise | Exponential cost, artifacts above 3D |
| Random field | No spatial coherence → meaningless ∇ρ |
| Gaussian random field | No bounded, artifact-free gradient control |
| Uniform density | No structure to drive geodesics |

## Implications for RNF

The density field ρ(r) must provide:
- Smooth gradients ∇ρ for metric tensor g_ij
- Artifact-free structure for geodesic computation
- Dimensional scalability for 4D/5D experiments
- Biological plausibility (30,000–180,000 neurons/mm³)

**Only Simplex noise satisfies all requirements.**

## Result

Coherent density fields in 3D, 4D, and 5D — while most high-dimensional constructs collapse.

This is not a trivial choice. It **enables** the framework's dimensional generality.

## Reference

Perlin, K. (2002). "Improving Noise." SIGGRAPH 2002.