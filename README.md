# QXU6028 G29 L2 analysis

This public repository contains the analysis code for the Group 29 L2 virtual lab. The original coursework dataset is not published here. Obtain the group's `L2_data.csv`, `L2_meta.json`, and `results_template.json` through the course's authorized channel; keep them in one directory. The optional `manifest.json` can accompany them.

Install the dependencies and run:

```sh
python3 -m pip install -r requirements.txt
python3 reproduce.py --data /path/to/G29_dataset --output /path/to/empty_L2_output
```

This regenerates the model selection, gap estimate, Fermi-level trajectory, figures and diagnostic files from the supplied data. `--checkpoint` additionally requires a Git checkout containing the raw data and full provenance; it is not supported by this public code-only checkout. `results.json` and `commit_hash.txt` in the separately delivered checkpoint identify the source commit used for the published result; a repository commit alone does not prove a particular group's contribution history.

The standalone version of `reproduce.py` in the submitted L2 package embeds the G29 inputs and is intentionally not mirrored in this public repository.
