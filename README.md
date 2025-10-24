# resilient-rl-cbf

**Resilient RL + CBF for Multi‑Robot Safety & Adversary-Aware Exploration**

A Robotarium-based research codebase that combines Reinforcement Learning (DDPG-style actor–critic) with Control Barrier Functions (CBF) and consensus-style adversary estimation to enable resilient multi‑robot exploration and leader–follower tracking under sensing/communication Adversarial. The project was developed for research in multi‑robot localization, resilient control, and adversarial-aware exploration.

---

## Key features

* Robotarium integration and visualization (uses `rps.robotarium`).
* Leader–follower architecture with spline/Stanley path tracking for the leader.
* Control Barrier Functions (CBF) for collision avoidance and steering away from adversarial zones.
* RL actor/critic networks (PyTorch) for exploration and policy learning (select_action_exploration / select_action_exploitation).
* Adversarial region modelling and mini‑batch consensus updates for sensing and communication threat estimation.
* Simple DDPG-style training loop with experience replay and in‑simulation logging (CSV outputs).

## Repository name

`resilient-rl-cbf`

## Requirements

* Python 3.8+ (tested on 3.8–3.11)
* PyTorch
* NumPy, SciPy, Pandas, Matplotlib
* Robotarium Python package (`rps`) and its utilities (barrier certificates, controllers, transformations)

Install with pip (example):

```bash
pip install numpy scipy pandas matplotlib torch
# Install Robotarium / rps utilities according to your Robotarium setup
```

## Quick start

1. Ensure Robotarium Python tools are installed and configured.
2. Place `Resilient_RL_CBF.py` at the repository root.
3. Run the simulation:

```bash
python Resilient_RL_CBF.py
```

The script will launch the Robotarium visualization, run the simulation for the configured number of steps, and export CSV logs (`control_input.csv`, `sensing_data.csv`, `communication_data.csv`, `ground_truth.csv`, `simulation_results.csv`).

## Configuration

Most parameters are located in the `CONFIG` dictionary at the top of `Resilient_RL_CBF.py` (total robots, sensing/communication ranges, noise levels, CBF gains, learning rates, batch sizes, etc.). Edit these to match hardware, Robotarium simulation scale, or experiment design.

## File structure (suggested)

```
resilient-rl-cbf/
├─ Resilient_RL_CBF.py
├─ README.md
├─ requirements.txt
├─ scripts/ (optional helpers)
├─ data/ (output CSVs)
└─ notebooks/ (analysis / visualization)
```

## Notes & caveats

* The code assumes the `rps` Robotarium utilities are available and that the Robotarium simulator is correctly set up. Adjust `initial_conditions`, `number_of_robots`, and plotting limits to match your environment.
* Adversarial modelling and RL hyperparameters are provided as research starting points — tune them for stability in your experiments.

## License

MIT License — see `LICENSE` (or add one if needed).

## Authors

Tohid Kargar Tasooji — Postdoctoral Researcher, University of Georgia. Contact via your preferred channel.

---

If you want, I can: (1) add a `requirements.txt`, (2) craft a minimal `LICENSE` file, (3) create a `run.sh` script to start experiments, or (4) produce a short usage tutorial/figure for the README. Which should I do next?

