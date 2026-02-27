# Jarvis-v1 (SNN Playground)
 
 This repository currently contains a minimal, educational **Spiking Neural Network (SNN)** implementation in **PyTorch**.
 
 SNNs are often described as the “third generation” of neural networks: instead of continuous activations, neurons communicate via **discrete spikes** over **time**. This codebase focuses on the practical core pieces needed to train an SNN with modern deep-learning tooling.

 ## The workflow (how it works end-to-end)

 The high-level loop we run is analogous to a brain-like perception/decision cycle:

 1) **Sense / input**
 
    We start with regular tensors (e.g., features, images). In biology, this corresponds to sensory neurons receiving signals.

 2) **Neural coding (convert values → spikes over time)**

    SNNs don’t pass continuous activations between layers; they pass spike events across timesteps.

    - **Rate / Poisson coding** (`poisson_encode`): larger input → higher spike probability per timestep.
    - **Time-to-first-spike (TTFS)** (`ttfs_encode`): larger input → earlier spike time.

 3) **Synaptic integration (weighted currents)**

    Linear layers (e.g., `nn.Linear`) act like **synapses**:

    - Inputs (spikes) are multiplied by **weights** (synaptic strengths).
    - The result is a current-like value injected into the next neuron population.

 4) **Neuron dynamics (LIF membrane potential)**

    The `LIFLayer` is the neuron body (soma):

    - **Membrane potential** `v` integrates incoming current.
    - **Leak**: `v` decays each timestep (controlled by `tau`).
    - **Threshold**: if `v >= threshold` → emit a **spike**.
    - **Reset/Refractory**: after firing, `v` resets and can optionally be held back for a few steps.

 5) **Readout / decision**

    We convert spiking activity into a decision signal (logits). In the toy demo we:

    - accumulate output contributions across timesteps
    - average over time to produce class logits

    Other common readouts include spike counts, last-timestep membrane potential, or filtered traces.

 6) **Learning (credit assignment over spikes)**

    Real spikes are binary, so their derivative is not useful for gradient descent.
    We therefore use **surrogate gradients**:

    - Forward pass: hard threshold spike (0/1)
    - Backward pass: a smooth derivative approximation near the threshold

    This enables standard optimizers (Adam/SGD) to train spiking networks.
 
 ## Mimicking the brain: what’s similar vs what’s not

 This project aims to capture *some* key brain-inspired principles while staying trainable and practical.

 Similarities (high-level):

 - **Event-based communication**: neurons mostly stay silent; information flows as sparse spikes.
 - **Stateful neurons**: the neuron has memory via its membrane potential `v`.
 - **Time matters**: computation unfolds over timesteps, not just layers.

 Differences / limitations (important):

 - Surrogate gradients are a **training trick**, not how biological neurons learn.
 - Real brains have many cell types, dendritic computation, neuromodulators, structural plasticity, etc.
 - “Real intelligence” is not guaranteed by SNNs alone; it also depends on objectives, data, embodiment, memory, planning, and world models.

 ## Roadmap toward more brain-like intelligence (suggested)

 If your goal is to push toward more brain-like behavior, these are sensible next steps to implement in this repo:

 - **Richer neuron models**
   - adaptive thresholds, conductance-based synapses, bursting neurons
 - **Biologically-inspired learning**
   - STDP (local plasticity), reward-modulated STDP (3-factor learning)
 - **Memory + recurrence**
   - recurrent spiking layers, attractor dynamics, working memory tasks
 - **Event-driven perception**
   - event camera datasets / DVS, audio spike encoding
 - **Neuromorphic constraints**
   - sparse operations, quantization, energy-aware training

 ## What’s implemented
 
 - **Spiking neuron model (LIF)**
   - A **Leaky Integrate-and-Fire (LIF)** neuron integrates input current into a membrane potential `v`.
   - `v` **leaks/decays** over time (`tau`).
   - When `v` crosses a **threshold**, the neuron emits a **spike** and `v` is **reset**.
   - Optional **refractory steps** prevent immediate re-firing.
 
 - **Neural coding / encoding**
   - **Poisson (rate) encoding**: converts continuous inputs in `[0,1]` into spike trains over `T` timesteps.
   - **TTFS (time-to-first-spike) encoding**: stronger inputs spike earlier.
 
 - **Learning mechanism (surrogate gradients)**
   - Spikes are binary (0/1), so their true gradient is zero almost everywhere.
   - We use a **surrogate gradient**: a smooth approximation in the backward pass while keeping hard spikes in the forward pass.
 
 ## Project structure
 
 - **`snn/`**
   - **`surrogate.py`**: `SurrogateSpike` autograd function/module.
   - **`lif.py`**: `LIFLayer` + state (`LIFState`).
   - **`encoding.py`**: `poisson_encode`, `ttfs_encode`.
 - **`demos/`**
   - **`train_toy.py`**: trains a small SNN classifier on a synthetic dataset (no downloads required).
 
 ## Setup
 
 1) Create and activate a virtual environment
 
 ```bash
 python3 -m venv .venv
 source .venv/bin/activate
 ```
 
 2) Install dependencies
 
 ```bash
 pip install -r requirements.txt
 ```
 
 Note: `torch` must be installed successfully for the demo to run.
 
 ## Run the toy demo
 
 ```bash
 python3 demos/train_toy.py --epochs 10
 ```
 
 Useful flags:
 
 - `--timesteps`: number of simulation steps (e.g. 10–50)
 - `--hidden`: hidden layer size
 - `--lr`: learning rate
 
 ## Where to go next
 
 - Add an MNIST demo (requires `torchvision`) using the same encoders + LIF layers.
 - Add multi-layer spiking networks and alternative readouts (spike count vs membrane potential).
 - Add STDP (unsupervised) as a separate learning module.
