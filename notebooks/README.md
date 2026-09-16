# Notebooks

## `mask_graph_augmentation_demo.ipynb`

Canonical visual walkthrough for the cumulative pipeline:

1. load and display the configured binary mask;
2. extract and overlay the graph;
3. rotate the graph directly;
4. flip the transformed graph directly;
5. geometrically crop full branch paths and rebuild the embedded graph;
6. inspect the full sequence;
7. inspect the auditable run summary / `min_ipd` output.

The notebook has **no behavioral experiment parameters**. It loads
`configs/retinal_augmentation_demo.yaml` and calls the same `run_experiment(...)`
path used by the CLI. Change behavior in a copied YAML, not in notebook cells.

Install notebook support and run:

```bash
python -m pip install -e ".[notebook]"
jupyter lab notebooks/mask_graph_augmentation_demo.ipynb
```

The bundled `data/retinal_binary_mask.gif` is the demo input referenced by the
YAML. For final minimum-segment PSLG experiments, change
`extract.simplify.method: optimal` in the YAML.
