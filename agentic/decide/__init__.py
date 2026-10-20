"""Decision infrastructure: every decision variable (four tiers: TileLang knobs, TileLang controls not offered for
tuning, skeleton structures, hardware features TileLang cannot state) declared once in registry.yaml with its domain,
binder, constraints and the model term that prices it; a generic MILP over them; a pipeline from the MILP's shortlist
to the simulator and the GPU."""
