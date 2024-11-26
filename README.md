# crossing-distances
Pedestrian crossing distances of major US cities

## Data ingestion
### Prerequisites
1. Create the uv environment from the lockfile
2. Set the modal API token for the crossing-distances project
3. Add a CSV file of coordinates to the `input/` folder at the root of this repo. This file should have a schema like so:
```
coords.csv
|- index - this one isn't read, but must be unique
|-     x - longitude in EPSG:3857. Must be to at least 5 decimal points of precision (6 preferred)
|-     y - latitude in EPSG:3857. Must be to at least 5 decimal points of precision (6 preferred)
```
### Usage
Run the functions in modal:
```shell
uv run modal run -d --env sfo data.py --mode remote --input inputs/sf_coords.csv --sample -1
```
Here's what each flag does:
- `-d` daemonize the process so it runs even if you close the terminal session
- `--env` modal environment to run the function in. This should be the airport code of the metro region you're getting images for
- `--mode remote` run modal function in serverless mode. Use `local` for local testing (be warned this is flaky)
- `--input inputs/sf_coords.csv` input coordinates file, starting from this repository's root
- `--sample -1` number of coordinates to randomly sample from the coordinate file. -1 to get all the coordinates in the file
