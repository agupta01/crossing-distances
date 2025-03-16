# Running tests

## Evaluation suite
We have an evaluation suite that compares crosswalk measurements between two datasets. Before you run it, ensure you have a GeoJSON or shapefile zip for each dataset saved locally. Then:

```bash
python -m tests.evaluate_crosswalks --before-path <before_path> --before <before_name> --after-path <after_path> --after <after_name> --plots-path <plots_path>
```

This will return summary statistics for each dataset, the KL divergence between the datasets summary statistics for paired differences between the datasets, and a histogram comparison of the lengths and paired (absolute) differences.

## Unit tests
Currently we only have one for utils.py in the inference module. To run it:
```bash
MODAL_ENVIRONMENT='main' uv run pytest tests/
```