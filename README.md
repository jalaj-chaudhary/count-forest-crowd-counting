# COUNT Forest crowd counting on the Mall dataset

This is a from-scratch reimplementation of **COUNT Forest** (Pham et al., ICCV 2015) for density-based crowd counting, evaluated on the Mall dataset. It is a course project for EE5340 Video Analytics at IIT Guwahati, by Naman Bhalawat, Siddhant Zawar and Jalaj Chaudhary.

- **Project report (website):** https://jalaj-chaudhary.github.io/count-forest-crowd-counting/
- **Code:** [`count_forest_full.ipynb`](count_forest_full.ipynb)
- **Demo video:** see section 9 of the report

| Variant | Test MAE | Test MSE | Test MDE |
|---|---|---|---|
| COUNT Forest (plain) | 4.44 | 27.71 | 0.137 |
| + forest permutation | 4.46 | 27.91 | 0.137 |
| + crowdedness prior | **4.41** | **27.46** | **0.136** |
| Paper, full method (quoted) | 2.50 | 10.0 | 0.080 |

Setup and run instructions are in [`docs/setup.md`](docs/setup.md).

## Repository layout

```
count_forest_full.ipynb    implementation and experiments
ppt/                       presentation slides
docs/                      report website (MkDocs Material)
mkdocs.yml                 site configuration
.github/workflows/         builds and deploys the site to GitHub Pages
```

To preview the site locally: `pip install "mkdocs<2" mkdocs-material`, then `mkdocs serve`.
