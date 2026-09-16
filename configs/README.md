# Paper-facing configs

Every runnable experiment is one self-contained YAML file.  The canonical CLI is:

```bash
mask2graph configs/retinal_augmentation_demo.yaml
```

No behavioral CLI flags are supported.  Copy a YAML, edit the explicit values,
and keep that resolved file with the resulting run archive.

`retinal_augmentation_demo.yaml` drives the notebook and the command-line smoke
path.  For final minimum-segment PSLG experiments, change the YAML field
`extract.simplify.method` from `rdp` to `optimal`; do not change it in notebook
code.
