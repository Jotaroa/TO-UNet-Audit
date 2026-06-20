"""
attn_audit
==========
Does attention attend to mechanics? A confound-controlled audit of the spatial
attention gates in NN4TopOptUNet (AG-ResU-Net) for topology-optimization
prediction.

Pipeline:
    dataset.py  : generate SIMP data + 2-channel training pairs (retrain model)
    train_model : (top-level script) train NN4TopOptUNet with your loss
    audit.py    : evaluate attention vs sensitivity (Q1/Q2/Q3 + baselines)
    sanity.py   : cascading weight-randomization faithfulness check
"""
from .simp import SIMPSolver, SIMPConfig, BoundaryConditions, mbb_beam, cantilever
from .dataset import (generate_objects, make_training_pairs, sample_problem,
                      save_dataset, load_audit_objects, build_net_input,
                      merge_datasets)
from .attention import extract_attention_masks, AttentionExtractor
from .alignment import align_maps, Alignment
from .baselines import competing_targets, competition, shuffle_control
from .audit import analyze_sample, aggregate
from . import sanity, viz

__version__ = "0.2.0"
