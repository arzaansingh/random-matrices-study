# SKILLS.md

House rules for "A Study of Random Matrices". Read before drafting a section, refer to during writing, and verify against before committing. This document is prescriptive. When fast and consistent are in tension, consistent wins.

---

## 1. Voice and audience

- **Voice**: third person, present tense, collective "we". No first-person singular.
- **Audience floor**: a strong high-school senior or first-year undergraduate. Every concept is introduced before use. No bare claims; everything is proved, sketched, or cited.
- **Sentence discipline**: one idea per sentence, two for a long one. No three-clause monsters.
- **No em dashes**: neither `---` (LaTeX) nor "—" (Unicode). Use commas, semicolons, periods, or parentheses. En dashes (`--`) are fine for numerical ranges.
- **Avoid hedging filler**: "it should be noted that", "obviously", "clearly", "trivially" without justification.

## 2. Math notation

All notation lives in `style/macros.tex`. Never redefine inline. If a new macro is needed, add it to `macros.tex` and document it.

| Object | Notation |
|---|---|
| Real numbers | `\R` |
| Expectation | `\E[X]` |
| Variance, covariance | `\Var(X)`, `\Cov(X, Y)` |
| Probability | `\Prob` |
| Population covariance | `\Sigma` |
| Sample covariance | `\Sigmahat` |
| Convergence a.s. / P / d | `\convas`, `\convp`, `\convd` |
| Distribution equality | `\eqd` |
| Vectors / matrices (bold) | `\vect{v}`, `\mat{M}` |
| Transpose | `\transpose` |
| Trace | `\tr` |
| Diagonal of | `\diag` |
| Spike map | `\PsiSpike` |
| Limiting overlap | `\rhoSpike` |
| First-use vocabulary highlight | `\vocab{term}` |

Math typography:
- Inline: `$ ... $`.
- Display unnumbered: `\[ ... \]`.
- Display numbered: `\begin{equation} ... \end{equation}`.
- Aligned: `\begin{aligned} ... \end{aligned}` inside display.
- Never `$$ ... $$` (deprecated).
- No bare `\mathbb{R}` in prose; use `\R`.
- Punctuate display math: equations end with periods or commas as if sentences.

## 3. Theorem culture

Five environments, defined in `style/thesis.sty`. Do not invent new ones.

| Env | Counter | Style |
|---|---|---|
| `definition` | per section, shared with `example` | bold head, sage-highlighted counter |
| `example` | per section, shared with `definition` | bold head, sage-highlighted counter |
| `theorem`, `proposition`, `lemma`, `corollary` | per section, shared | bold head, sage-highlighted counter |
| `intuition` | unnumbered | italic head, no number (a viewpoint, not a claim) |
| `remark` | unnumbered | italic head, no number |

Patterns:
- **Picture before proof.** Every theorem gets a figure, numerical experiment, or low-dimensional worked-out case before its formal statement. The reader sees what is true before being convinced of it.
- **Build then prove.** Establish a claim with examples and simulations before proving it.
- **Forward references are explicit.** "We will see in Section X that..." is fine. A bare "we will return to this later" is not.

For every standard result, choose one of:
1. Prove inline.
2. Sketch inline with a pointer to the full proof in Appendix A or a citation.
3. State and cite cleanly.

## 4. Citations

- Author-year style via `\citep{key}` (for parenthetical) and `\citet{key}` (for textual), backed by `natbib` with the `plainnat` style.
- Every standard result is cited. No bare claims. No broken keys.
- Primary sources to cite by name when relevant: Casella and Berger (2002), Billingsley (1995), Vershynin (2018), Yao, Zheng, and Bai (2015), Marchenko and Pastur (1967), Tracy and Widom (1996), Johnstone (2001), Baik, Ben Arous, and Péché (2005), Féral and Péché (2009), von Luxburg (2007), Couillet and Benaych-Georges (2016), Genzer (2025).
- New entries belong in `refs.bib`, keyed in `AuthorYear` form (e.g., `FeralPeche2009`).

## 5. Color palette (locked)

Six colors total. Defined in `style/colors.tex`. Do not introduce new colors anywhere.

| Role | Hex | Used in |
|---|---|---|
| Vocabulary highlight (sage) | `#D4E8C8` | `\vocab{}`, theorem-counter chip |
| Definition rule (slate navy) | `#3A5A78` | reserved for definition environments and code keywords |
| Theorem rule (charcoal) | `#2D2D2D` | reserved for theorem/proposition/lemma/corollary |
| Intuition tint (warm cream) | `#FBF3D9` | subtle background for intuition remarks, code background |
| Hyperlink (muted blue) | `#3D7BAA` | `hyperref` link, cite, URL |
| Figure / code accent (warm gray) | `#8B8378` | caption rule, listing border, secondary accents |

Plots use only these swatches (with alpha variants permitted). A figure that introduces a non-palette color fails the audit.

## 6. Figures and simulations

- **Two tools only**: matplotlib for data, TikZ for diagrams. No mermaid, no drawio, no hand sketches.
- **Plot styling**: every notebook imports the shared style helper `notebooks/thesis_style.py` (to be created when the first plot lands). No bare `plt.plot(...)` without a style setup call.
- **Reproducibility**: every plot has a corresponding script. Every script seeds the RNG explicitly with `np.random.default_rng(seed)`.
- **File output**: figures are saved as `.png` (raster) into `figures/` for inclusion in the thesis. Notebooks live in `notebooks/`.
- **Captions**: complete sentences, ending in periods. Describe what the reader should see, not the mechanics of the plot.
  - Bad: "Histogram of 5000 simulated top eigenvalues."
  - Good: "The top eigenvalue concentrates near $\Psi(\alpha) \approx 3.33$, well above the bulk edge $(1+\sqrt{y})^2 \approx 2.91$."
- **Figure references**: every figure is referenced from the text before it appears: "see Figure 4.1".
- **Page counting**: figures and tables do not count toward the 30-to-40-page main-text budget.

## 7. Notebooks

- Python 3 only.
- Top of every notebook: a `# === CONFIG ===` block with `alpha`, `y`, `n`, `trials`, `seed`. The user can change these to reproduce variants.
- Standard imports:
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy import stats
  from thesis_style import setup_plot
  setup_plot()
  rng = np.random.default_rng(seed=2026)
  ```
- Vectorize when possible. Avoid `for i in range(len(...))`; use `enumerate` or `zip`.
- Comments explain "why," not "what."

## 8. Audit checklist (before each commit)

Run three passes on the section that was touched.

### 8.1 Visual audit
- [ ] No box overlap, no clipped text, no broken cross-references (no `??`).
- [ ] Color palette compliance (only the six colors).
- [ ] Captions complete; figures numbered.
- [ ] Page breaks happen at clean paragraph boundaries.

### 8.2 Logical audit
- [ ] Every concept defined before first use.
- [ ] Forward references are explicit, never bare "later".
- [ ] Picture-before-proof rule applied to every theorem.
- [ ] No bare claims; every standard result is proved, sketched, or cited.

### 8.3 Content audit
- [ ] Voice is third-person, present-tense.
- [ ] No em dashes anywhere.
- [ ] Notation matches `macros.tex`; no inline redefinitions.
- [ ] All citations resolve (run `bibtex` and check log for missing-key warnings).
- [ ] Spell check passes.

If any item fails, flag with `% AUDIT-FAIL: ...` and fix before committing.

## 9. Build

From `Thesis/`:

```
latexmk -pdf main
```

To clean intermediates (PDF kept):

```
latexmk -c
```

## 10. Workflow

- **One section at a time.** A section is the unit of work.
- **Source-to-draft order**: read the relevant pages of `LaTeX notes v5/main.tex`, the Gustavo handwritten notes (`Notes/Gustavo Notes *.pdf`), and the textbook chapters (Yao-Zheng-Bai 2015 Ch. 11 for spiked covariance; Casella-Berger for foundations). Then draft.
- **Commit message format**: `section: <slug>: <one-line summary>`. Example: `04: BBP transition: state and prove Cor 11.4 for Johnstone`.
- **GitHub sync**: push after each completed section.

---

## 11. The one rule

**Picture before proof. Citation before claim. Audit before commit.**
