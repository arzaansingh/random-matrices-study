# A Study of Random Matrices

Honors thesis manuscript and accompanying simulations.

**Author**: Arzaan Singh
**Advisor**: Professor Gustavo Didier
**Institution**: Department of Mathematics, Tulane University
**Year**: 2026

## What this is

A 30-to-40-page honors thesis on random matrix theory, sample covariance matrices, the spiked covariance model, the BBP phase transition, eigenvector alignment, and the connection to spectral clustering. The scope is intentionally narrow: Gaussian sample covariance matrices in Johnstone's spiked model.

The thesis is modeled on Joseph Genzer's 2025 Tulane honors thesis in format and tone: classical article-style LaTeX, single column, subtle sage-green highlights on first-use vocabulary, no marginalia. The complete style guide is in [`SKILLS.md`](SKILLS.md).

## Build

Requires a recent TeX Live distribution (or MacTeX) with `pdflatex`, `bibtex`, `latexmk`, and the packages `amsmath`, `amssymb`, `amsthm`, `mathtools`, `bm`, `xcolor`, `soul`, `microtype`, `enumitem`, `booktabs`, `caption`, `graphicx`, `natbib`, `hyperref`, `cleveref`, `listings`, `geometry`.

From this directory:

```bash
latexmk -pdf main
```

To clean intermediates (the PDF is kept):

```bash
latexmk -c
```

## Folder structure

```
Thesis/
├── README.md                    project overview, build instructions
├── SKILLS.md                    house rules, palette, audit checklist
├── main.tex                     root document, \input's everything below
├── refs.bib                     bibliography
├── latexmkrc                    latexmk config
├── .gitignore                   ignores LaTeX intermediates
├── style/
│   ├── thesis.sty               custom style package
│   ├── colors.tex               six-color palette, defined once
│   └── macros.tex               math macros (\R, \E, \Var, \Sigmahat, ...)
├── frontmatter/
│   ├── titlepage.tex            Tulane honors thesis title page
│   ├── abstract.tex
│   └── acknowledgements.tex
├── sections/
│   ├── 01-introduction.tex
│   ├── 02-probability-foundations.tex
│   ├── 03-sample-covariance-wishart.tex
│   ├── 04-spiked-covariance-bbp.tex
│   └── 05-conclusion.tex
├── appendix/
│   ├── A-auxiliary-results.tex
│   ├── B-simulation-methodology.tex
│   └── C-code-listings.tex
├── figures/                     PNG outputs from notebooks
└── notebooks/                   .ipynb files, one per simulation theme
```

## Notebooks

Each notebook is self-contained and parameterized at the top. The reader can change `alpha`, `y`, `n`, `trials`, `seed` and re-run.

| # | Notebook | Section it feeds |
|---|---|---|
| 01 | `01_lln_clt_demos.ipynb` | §2 |
| 02 | `02_wishart_and_repulsion.ipynb` | §3.2 |
| 03 | `03_fixed_dim_fluctuations.ipynb` | §3.2 |
| 04 | `04_marchenko_pastur_tracy_widom.ipynb` | §3.3 |
| 05 | `05_spiked_eigenvalues_bbp.ipynb` | §4.2 |
| 06 | `06_fluctuation_qq_plots.ipynb` | §4.3 |
| 07 | `07_eigenvector_overlap.ipynb` | §4.4 |
| 08 | `08_spectral_clustering_toy.ipynb` | §4.5 |

## Source materials

Drawn from (in approximate order of weight):
- `../LaTeX notes/Mathematics_Research_Notes_v5/main.tex` (working notes accumulated over the semester).
- `../Notes/Gustavo Notes *.pdf` (handwritten meeting notes from Professor Didier).
- Yao, Zheng, and Bai (2015), *Large Sample Covariance Matrices and High-Dimensional Data Analysis*, Cambridge University Press. Chapter 11 in particular.
- Féral and Péché (2009), "The largest eigenvalues of sample covariance matrices for a spiked population: diagonal case," *Journal of Mathematical Physics*, 50, 073302. Theorem 1.6 in particular.
- Casella and Berger (2002), *Statistical Inference*.
- Vershynin (2018), *High-Dimensional Probability*.
- Genzer (2025), Tulane honors thesis (format reference only, not content).

Full bibliography in [`refs.bib`](refs.bib).

## Workflow

One section at a time. The order is:
1. Draft from source notes.
2. Compile, audit (see §8 of `SKILLS.md`).
3. Commit with message `section: <slug>: <one-line summary>`.
4. Push to GitHub.

The thesis advisor (Gustavo) is shown the PDF after every two completed sections; his feedback is captured as TODO comments in the source and addressed before the next push.
