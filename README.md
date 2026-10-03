# Music Dismantling

> An experimental research project for discovering and separating the underlying acoustic components of mixed music recordings using signal processing, machine learning, self-supervised learning, and latent-space discovery.

---

## 1. Project Overview

The goal of this project is to build a system that can take a normal mixed music recording and progressively **dismantle it into meaningful acoustic components**.

The long-term objective is not simply conventional vocal/instrument separation.

The desired system should eventually be able to:

- Take a mixed song as input.
- Discover its underlying acoustic components.
- Separate instruments without requiring a fixed instrument list.
- Handle different numbers of instruments in different songs.
- Discover recurring acoustic patterns across different songs.
- Potentially distinguish multiple singers.
- Allow the user to preview individual discovered components.
- Allow selected components to be saved.
- Eventually provide a GUI around the complete system.

The project deliberately started from classical signal processing and progressively moved toward machine learning and self-supervised representation learning.

The current research direction is:

```text
Mixed Music
     ↓
Time-Frequency / Acoustic Representation
     ↓
Self-Supervised Latent Learning
     ↓
Latent Acoustic Space
     ↓
Clustering / Pattern Discovery
     ↓
Weak Semantic Labels
     ↓
Instrument-Aware Separation
     ↓
Individual Sources
