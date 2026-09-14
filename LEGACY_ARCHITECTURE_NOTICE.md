# Legacy architecture notice

The following files and directories predate the current IQ architecture research and are retained only as historical experiments/reference material:

- `QUANTUM_LLM_IMPLEMENTATION_PLAN.md`
- `QUANTUM_AI_COMPARISON_2026.md`
- `nif_sovereign/`
- legacy CUDA-Q / neutrino / Ising / manually named MoE designs

They are **not** authoritative inputs for new IQ architecture work.

The active architecture source of truth on this branch is:

1. `IQ_ARCHITECTURE.md`
2. `iq_model/config.py`
3. `iq_model/mixers.py`
4. `iq_model/components.py`
5. `iq_model/reasoning.py`
6. `iq_model/model.py`

If historical files conflict with the active architecture, the active files above win.
