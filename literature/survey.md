# Literature Survey: PVRPWDP Time Window

## PARCO

- URL: https://arxiv.org/abs/2409.03811
- Relevance: PARCO provides the parallel autoregressive multi-agent construction
  framework, communication layers, multi-pointer decoder, and conflict handler.
- Gap for this project: PARCO's routing benchmarks do not directly solve
  VRPTW-style hard temporal feasibility, so PVRPWDP needs extra slack features,
  verifier tooling, and time-aware conflict handling.

## OR-Tools VRPTW

- URL: https://developers.google.com/optimization/routing/vrptw
- Relevance: Uses a time dimension and slack variables to model waiting and time
  windows.
- Takeaway: Separate schedule feasibility from route structure; report solution
  windows and waiting explicitly.

## OR-Tools Dropped Visits

- URL: https://developers.google.com/optimization/routing/penalties
- Relevance: Uses disjunction penalties to let a solver drop nodes when full
  service is impossible or too costly.
- Takeaway: This maps to optional-customer PVRPWDP with adaptive big-M penalties.

## PyVRP / Hybrid Genetic Search

- URL: https://pyvrp.github.io/v0.2.1/examples/vrptw.html
- API: https://pyvrp.readthedocs.io/en/stable/api/pyvrp.html
- Relevance: Provides VRPTW diagnostics such as wait time, time warp, slack, and
  route schedule information.
- Takeaway: HGS-style soft infeasible search plus repair is promising for tight
  PVRPWDP time windows.

## Vidal HGSADC for VRPTW

- URL: https://www.sciencedirect.com/science/article/pii/S0305054812001645
- Relevance: Hybrid genetic search with adaptive diversity management is a
  strong classical baseline for VRP variants with time windows.
- Takeaway: Feasible/infeasible subpopulations and time-warp penalties can guide
  search before final hard-feasible repair.

## RL for VRPTW

- RL stochastic VRPTW: https://arxiv.org/abs/2402.09765
- Dynamic CVRP with TW: https://arxiv.org/abs/2102.12088
- RL-AVNS multiple TW: https://arxiv.org/abs/2505.23098
- Relevance: Reinforcement learning is commonly used with feasibility masks,
  heuristic improvement operators, or group rollouts.
- Takeaway: For this repo, RL ideas should first be evaluated as isolated
  experiments after the dataset verifier and GA baseline are stable.

## Extended Related Work Map

- File: `literature/pvrpwdp_timewindow_related_work.md`
- Scope: VRPTW, PDPTW/VRPSPDTW, perishable VRPTW, truck-drone VRPTW,
  optional/profitable VRPTW, HGS/ALNS patterns, and concrete transfer to PVRPWDP.
- Main conclusion: PVRPWDP should first decide hard-cover vs optional-customer
  semantics, then enforce that choice through data generation, verifier tooling,
  reward, GA baseline, and PARCO features.
