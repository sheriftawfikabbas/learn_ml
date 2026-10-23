# Datasets

All datasets live in this folder and are referenced from the classes and case
studies via paths relative to the repository root (e.g. `../../Data/...`).

| File / folder | Description | Source |
|---|---|---|
| `GlobalTemperatures.csv` | Monthly global land and land-ocean average temperatures (°C), 1750–2015, from the Berkeley Earth averaging work. | [Berkeley Earth](http://berkeleyearth.org/data/) (mirrored on Kaggle) |
| `time_series_covid19_deaths_global.csv` | Daily **cumulative** COVID-19 death counts by country, Jan 2020 – 15 May 2022. | [JHU CSSE COVID-19 Data Repository](https://github.com/CSSEGISandData/COVID-19) |
| `tmr_bicycle_counts_january2022_december2022.csv` | Hourly bicycle counts from Queensland (Australia) counters, 2022. | [QLD TMR bicycle counts](https://www.data.qld.gov.au/dataset/tmr-bicycle-counts) |
| `ucs_satellite_database_7_1_16.csv` | Union of Concerned Scientists database of operational satellites (1 July 2016 snapshot). Deliberately "messy" — mixed units, missing values, junk columns. | [UCS Satellite Database](https://www.ucsusa.org/resources/satellite-database) (mirrored at [data.world](https://data.world/phalseid/satellite-data)) |
| `cell_images/` | Segmented red-blood-cell images, `Parasitized/` and `Uninfected/`. Multiple cells can come from the same patient (patient IDs are embedded in filenames — see CaseStudy05). | [NIH Malaria Datasets](https://ceb.nlm.nih.gov/repositories/malaria-datasets/) |
| `Molecules/` | 5,000 organic molecules (`dsgdb9nsd_*.xyz`) from the GDB-9 quantum-chemistry dataset. The comment line of each xyz file holds scalar properties; field index 9 is the HOMO–LUMO gap (Hartree). | [GDB-9 / Quantum-Machine.org](https://doi.org/10.6084/m9.figshare.978904) |

## Usage notes

- **Where each dataset is used:** `GlobalTemperatures.csv` → CaseStudy01 (offline
  fallback); `time_series_covid19_deaths_global.csv` → CaseStudy04;
  `ucs_satellite_database_7_1_16.csv` → Class03;
  `cell_images/` → Class04 and CaseStudy05; `Molecules/` → CaseStudy03.
  The bicycle-counts CSV is currently an extra dataset for your own practice.
- The COVID file is a frozen snapshot (15 May 2022). The JHU repository stopped
  updating this series in 2023, so it will not grow further.
- The satellite database and the COVID series are used *because* they are dirty —
  they contain real parsing, dtype and missing-data problems.
