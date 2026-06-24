# STYLE_GUIDE.md — Random Matrix Theory Report

The single template for editing every word. If a sentence fails any rule below, fix it before it reaches Gustavo. The goal: he never gives the same note twice.

---

## 1. Voice

Write **engaging and parsimonious, and mathematically correct**. Engaging and parsimonious are not opposites: the engine of engagement is a clear through-line stated concretely, not hype, and not padding. The two profiles behind this guide are Joseph Genzer's thesis (engaging, medium-length sentences) and Didier's papers (precise, results-first); the operational rules below are how you write like them without needing either text open.

**The four pillars, each made testable:**

- **Engaging means a concrete through-line, with positive devices — not hype.** Be engaging by doing, not by adjectives. The four engagement engines, each its own rule below: state the main result concretely up front (R-abstract, R13), open each section with one purposeful orienting sentence (R-open), label marquee results (R18), and close each cluster of results with a result-to-meaning bridge ("Thus,", "In summary,") that says what the math means for the report's story (R39). The hook is the precise surprise itself (the sample eigenvalues stop converging; a sharp threshold at `1 + sqrt(y)`), stated plainly — never an intensifier.
- **Parsimonious means each fact appears once, in the shortest correct phrasing.** Displayed equations do the heavy lifting; prose stays lean (2–4 sentences per paragraph). Every sentence either states a fact, displays an equation, or bridges result-to-meaning with an explicit connective. No sentence asserts significance without adding content (R11).
- **The register is measured mathematical English.** Use "we" for mathematical agency ("we consider", "we show", "we test"). No rhetorical flourish, no jokes, no "very/extremely/remarkable/groundbreaking". The strongest praise allowed is restrained ("central", "renowned").
- **Be correct before being smooth.** A weaker true statement always beats a stronger false one. State proven results flatly; flag simulated ones as simulated and name the estimated value.

**Litmus test for every sentence (made checkable):** *Does this sentence state a fact, number, or definition that does not already appear earlier in this document (search backward for it), AND is it true given only what has been proved, cited, or simulated so far?* If not, cut or fix it.

---

## 2. Correctness and No Overclaiming

**R1. Claim only what is proved, cited, or simulated — and say which.**
*Why:* Overstated rigor is the fastest way to lose the reader's trust.
*Didier:* "each claim is proved" — `\GDcomment{This is not really true. :) }`
Bad: "each claim is proved"
Good: "each claim is proved, sketched with a reference to a complete proof, or cited at the precise statement used, and the most difficult limit laws are stated and simulated rather than proved."

**R2. Name the precise quantity; never "the truth", "the picture", "the behavior".**
*Why:* A vague referent hides whether the claim is even well-posed.
*Didier:* "almost three times the truth" — `\GDcomment{``The truth"?}`
Bad: "the largest sample eigenvalue sits near 2.91, almost three times the truth"
Good: "the largest sample eigenvalue sits near 2.91, almost three times the population eigenvalue 1"

**R3. Name the statement that fails or holds, and the body of theory it belongs to — not a gesture at either.**
*Why:* "The picture changes" tells the reader nothing checkable, and "classical ___ guarantees" leaves the body of theory dangling.
Bad: "many modern data sets do not sit in that regime [and] the picture changes." / "classical theory guarantees the estimate."
Good: "in the high-dimensional regime the sample eigenvalues no longer converge to those of `Sigma`." / "classical statistical theory guarantees the estimate." (Name the body of theory precisely — "classical statistical theory" — not a bare modifier.)

**R4. Phrase spectral facts as limits; attach hypotheses to formulas.**
*Why:* A density without its regime, or a convergence written as an equality, is a different (false) statement.
*Didier:* "asymptotic support" (inserted before the interval `[(1-sqrt y)^2,(1+sqrt y)^2]`).
Bad: "they spread across the support `[(1-sqrt y)^2,(1+sqrt y)^2]`."
Good: "as `p` and `n` grow together with `y = p/n` fixed, they spread across the asymptotic support `[(1-sqrt y)^2,(1+sqrt y)^2]`."

**R5. Distinguish finite-n effects from limit statements.**
*Why:* A near-threshold "merge" is finite-sample resolution, not an asymptotic fact.
Bad: "near the threshold the two regimes merge."
Good: "near `alpha_c` the regimes are hard to separate at finite `n`; the limits still differ for fixed `alpha` off the threshold." (For a sharp crossing, prefer "once `alpha` crosses the threshold" over "when/if".)

**R6. A statistic that converges to a constant has a degenerate limit law until centered and rescaled.**
*Why:* A violin/histogram of such a statistic shows concentration, not a limiting distribution.
Bad: "the violins show the asymptotic distribution of the squared alignment."
Good: "the violins show finite-`n` concentration; the distributional limit requires centering and rescaling (Theorem X)."

**R7. Bound every subspace/aggregate claim by its single-component result.**
*Why:* No plane is recovered better than a direction inside it.
Bad: "the spike plane is recovered."
Good: "the plane is identifiable to the same partial degree `rho < 1` as a single spike direction, no better; directions inside the plane are not identifiable."

**R8. Use the precise word; downgrade a strong false word to a weaker true one.**
*Why:* "independent", "recover", "exact", "reliable" carry precise meanings; check before using.
Bad: "two distinct spikes behave like two independent single spikes."
Good: "two distinct spikes decouple: matched overlaps converge to their one-spike constants and crossed overlaps to zero (literal independence fails, since the eigenvectors are orthogonal by construction)."

**R9. Reserve escape/exit language for genuine outliers.**
*Why:* At the null nothing leaves the bulk; the top eigenvalue settles at the edge.
Bad: "the largest sample eigenvalue exits the bulk."
Good: "the largest sample eigenvalue converges to the upper edge of the bulk."

---

## 3. Concision and No Redundancy

**R10. State each fact once; delete clauses that restate a neighbor.**
*Why:* Repetition reads as padding.
*Didier:* "Every entry ... is a good estimate ..." — `\GDcomment{You already made the previous statement.}`
Bad: "Each entry of the sample covariance matrix is an unbiased average ... [later] Every entry of the sample covariance matrix is a good estimate of the matching entry of `Sigma`; its eigenvalues, taken together, are not."
Good: keep the first; delete the re-assertion.

**R11. Cut sentences that assert significance without content.**
*Why:* "Sharp and can be quantified" says nothing the next pages do not.
*Didier:* `\GDcomment{This sentence does not add information: ``The departure from the classical picture is sharp and can be quantified."}`
Bad: "The departure from the classical picture is sharp and can be quantified."
Good: [omit].

**R12. Delete forward-promises the body already delivers.**
*Why:* A preview duplicates the pages it points to. (This forbids empty "(explained below)" pointers — not the concrete statement of the main result, which belongs up front; see R13.)
Bad: "The dividing threshold depends on `y` in a precise way (explained below)."
Good: [omit; the following pages give the dependence].

**R13. State the main result concretely in the abstract/intro; keep only over-granular procedural detail for the body.**
*Why:* Both models put concrete results up front and refuse to make the reader wait; what does not belong in the intro is fine-grained experimental bookkeeping (sample counts, per-run percentages, step-by-step pipelines), not the result itself.
*Didier:* the MNIST 93.4%/93.8% walkthrough — `\GDcomment{Too many details for an introduction.}`
Bad: "We sort twelve thousand images of `0` and `1` ... places `93.4\%` ... predicted `93.8\%` ..." (over-granular procedure in the intro)
Good: "We give an explicit accuracy formula and confirm it on the MNIST digits in Section X." (The main result — the explicit formula — stated concretely; the image counts and per-run percentages move to the body.)
Note: do NOT over-correct into a teased or vague intro. The main theorem appears in plain language in the abstract and intro; only procedural bookkeeping is deferred.

**R14. Shortest correct phrasing; cut non-informative adjectives and lab jargon.**
*Why:* "exact low-rank", "pipeline", "validated" add noise, not information.
*Operational test for a non-informative adjective:* delete it and re-read the sentence; if no claim weakens and no named regime changes, the adjective is non-informative — cut it.
*Lab-jargon ban-list (replace each with its plain mathematical equivalent):* pipeline/workflow → procedure; validated/sanity-check → verified; tune → choose/set.
Bad: "the pipeline is the one validated in Figure X."
Good: "the procedure is the one verified in Figure X."

---

## 4. Clarity and Structure

**R-abstract. Open the abstract with the funnel: field → object → what-we-do → what-we-find → method, results stated concretely.**
*Why:* This is the positive structure both models use to be engaging without hype; teasing the result is the failure mode it prevents.
*Joseph (model):* "Random matrix theory ... provides a powerful framework ... This thesis examines the spectral behavior of sample covariance matrices ... We then extend the analysis ... Our results show that dependence fundamentally reshapes the spectral behavior ..."
Bad: "We study a surprising phenomenon in high-dimensional statistics, with results that may interest the reader." (teased, no object, no concrete finding)
Good: a single dense paragraph that names the object (sample covariance matrices), states what we do (analyze the top eigenvalue and eigenvector), and states what we find concretely (a sharp threshold at `1 + sqrt(y)`), closing on the contribution.

**R-open. Open each section with exactly one purposeful orienting sentence; an infinitive of intent is welcome.**
*Why:* One orienting sentence gives the reader the section's destination; throat-clearing or a history dump does not.
*Joseph (model):* "To lay the mathematical foundations for our analysis, we begin by reviewing key concepts in probability theory."
Bad: [section dives straight into a definition] or [section opens with three sentences of motivational filler].
Good: "To pin down where the top eigenvalue lands, we first record the limiting spectral distribution of the bulk."

**R15. Aim for medium sentences; split only overlong chains.**
*Why:* Both models write balanced claim-plus-consequence sentences; chopping every sentence to staccato flattens that rhythm into monotone, the dull failure mode.
*Joseph (model):* medium sentences (~15–30 words), one clause of setup plus one clause of consequence.
*Didier (model):* a compact claim followed by an elaborating subordinate clause — "The emergence of a fractal is usually the signature of a physical mechanism that generates scale invariance."
*Operational test:* flag any sentence over ~35 words OR containing two or more coordinating "and"/"but" clauses, and split it at a colon or semicolon into balanced halves (or attach the condition/mechanism as one subordinate clause). The grammatical subject and main verb should appear within roughly the first 12 words. Do NOT chop every sentence to staccato; vary length.
Bad: "When the data fall into several groups, that grouping shows up as a few strong directions, one per group, standing out against the noise, and the directions can be estimated, and the estimates are consistent." (overlong, three chained clauses)
Good: "When the data fall into several groups, each group contributes one strong direction that stands out against the noise. These directions can be estimated consistently."

**R16. Keep distinct phenomena in distinct sentences/paragraphs.**
*Why:* Blending unrelated mechanisms confuses both.
*Didier:* `\GDcomment{Here, one is mixing up two different issues: the geometry of spiked eigenvalues and the MP phenomenon.}`
Bad: the no-signal Marchenko-Pastur bulk-spreading story and the spiked-direction story in one breath.
Good: one paragraph for bulk spreading at the null; a separate paragraph for the spike.

**R17. Replace pronouns and elided nouns with their referent; name each member of a grouping word.**
*Why:* "it", "this", "the variables", "both", "the two" must each have one visible antecedent.
*Didier:* "compares the variables" — `\GDcomment{The variables?}`
Bad: "the sample covariance matrix is `p x p` and compares the variables."
Good: "the sample covariance matrix is `p x p`; its `(i,j)` entry is the sample covariance between the `i`-th and `j`-th measured features."

**R18. Orient before formalizing; one-sentence lead-in per theorem/definition, and label marquee results.**
*Why:* A theorem dropped cold loses the reader; naming the stakes of a main result is how Joseph keeps the reader invested.
*Joseph (model):* "The following theorem is the first of the two main mathematical results of this thesis."
*Operational test:* each theorem/definition is immediately preceded by exactly one sentence that names what the result determines or where it is used (it contains a verb naming the result's output, e.g. "pins down", "determines", "gives"). Marquee results may be explicitly labeled as such.
Bad: [Theorem appears with no introduction.]
Good: "The next theorem pins down where the top sample eigenvalue lands at the null. [Theorem]"

**R19. Heuristics follow the result they explain; prefer prose transitions to bare headings inside a derivation.**
*Why:* A scaling heuristic motivates best after the theorem; a connecting sentence keeps the argument a story.
*Operational test:* every heuristic/scaling argument appears AFTER the displayed result it motivates, never before.
Bad: "**Matrix form.**"
Good: "The entrywise formula repackages conveniently in matrix form."

**R19b. One display per formula.**
*Why:* Two displays for one quantity (e.g. supercritical and subcritical written separately) force the reader to reassemble what is logically a single object.
*Didier:* `\GDcomment{Why are they displayed separately? Display the two limits to the right of a left curly brace.}`
Put algebraic re-expressions in prose, not a fresh display; merge the two regimes of one quantity into a single cases brace.
Bad: `rho = [supercritical formula]` (displayed), "and `rho` converges to 0 below threshold." (a second display or tacked-on regime)
Good: `rho(alpha, y) = { [supercritical formula], alpha > alpha_c;  0, otherwise }` (one cases display).

**R20. Problems have goals; matrices and data do not act; rewrite anthropomorphism and initial negations.**
*Why:* "data tell directions apart" and "nothing is assumed" are imprecise and overstate.
*Scope:* this bans figuration that CARRIES a mathematical claim or hides a mechanism (data "tells", matrix "wants", eigenvalue "tries"). Restrained figurative language that names its exact content is allowed (Didier's "the signature of a mechanism"), since it states its shape rather than hiding one.
*Operational test:* scan for verbs of agency or perception (tell, want, try, decide, see, know, fight, escape, recover) with a non-agent subject (data/matrix/eigenvalue/signal). Each occurrence fails unless the subject is a person, "we", or "the problem/method". Rewrite with the mechanism as subject.
Bad: "the data tell the directions apart" / "nothing is assumed about the data."
Good: "the leading eigenvectors separate the groups" / "we make assumptions about the model, not the data."

**R21. State both what a quantity is for and what it is not; say what is retained on a viewpoint shift.**
*Why:* Restricted roles and carried-over objects must be explicit.
Bad: "the true labels are used only to evaluate."
Good: "the true labels are used only to score the result, never to form it."

---

## 5. Terminology and Definitions

**R22. Define every nonstandard or load-bearing term of art inline at first use; cite a reference for standard results rather than re-deriving or re-glossing them.**
*Why:* The reader needs the field's load-bearing vocabulary pinned down, but glossing every standard term bloats the prose and reads as condescending throat-clearing — both models cite standard facts instead.
*Didier (model):* "A *wavelet* is an oscillatory function in `L2(R)` with unit norm." (nonstandard, load-bearing → defined inline)
*Joseph (model):* cites "Casella and Berger, 2002, p.136" rather than re-deriving a standard textbook fact.
Bad: "the eigenvalues spread across the bulk." (load-bearing term "bulk" left undefined)
Good: "the eigenvalues spread across the bulk (the connected range they fill in the limit)."
Also define an operational term whose MEANING (not just label) is opaque: gloss "accuracy" as "the fraction of points sorted correctly" and "prediction" as "the value of the formula", rather than leaving the procedure's meaning implicit.

**R23. Read each load-bearing formula back in words — tightly — and define the informal word where it carries weight.**
*Why:* A verbal reading anchors the symbol; "signal" must be pinned down where it matters; and a clunkier paraphrase defeats the purpose, so the reading must be the shortest correct rendering.
Bad (no reading): "a fraction `d^2` of `u-hat` lands on the spike, and the signal is recovered."
Bad (clunky reading): "a fraction equal to the quantity `d^2`, which is `d` multiplied by itself, of the total unit-length amount of the estimated vector `u-hat` is found to lie along ..."
Good (tight reading): "a fraction `d^2` of the unit length of `u-hat` lands on the true spike direction, where the signal is the spike `alpha`, the excess population variance along that direction."

**R24. Name each specialization step before using its consequences; announce every change of model or assumption (and display the new object).**
*Why:* Silent specialization hides where generality is lost.
*Didier:* "It seems like we are focusing on Johnstone's model ... that was not clear from the narrative."
Bad (model change): [text silently starts using `B_p = I`.]
Good (model change): "From here we specialize to Johnstone's model, taking `B_p = I` and `H = delta_1`."
Bad (parameter specialization): [text silently sets the number of spikes to one and uses `e_1`.]
Good (parameter specialization): "We first treat the simple spike, `m = 1`, so the signal block is the single direction `e_1`; the general `m` follows in Section X."

**R25. Introduce generality only where it is used; give an analogy its exact mathematical content.**
*Why:* General machinery after a specific choice is a digression; a loose analogy must name its shape.
Bad: "the structure now sits in the means rather than the variances."
Good: "it has the same shape as the spiked model, a low-rank perturbation of a noise matrix, with the signal in the column means rather than a covariance."

**R26. One symbol per concept across the whole document.**
*Why:* A competing notation (`kappa` vs `K`) forces the reader to re-learn.
Bad: "the number of clusters `kappa`" (when theorems use `K`).
Good: "the number of clusters `K`."

---

## 6. Numbers, Regimes, and Figures

**R27. Anchor every number to its regime at the point it appears.**
*Why:* `[0.09, 2.91]` is meaningless without `y = 1/2`, `p = 500`, `n = 1000`; never attach it to a different setting (e.g. MNIST).
Bad: "the support is `[0.09, 2.91]`."
Good: "for `y = 1/2` (`p = 500`, `n = 1000`) the asymptotic support is `[0.09, 2.91]`."

**R28. Report the named statistic, not a paraphrase of what it measures.**
*Why:* "agree to 0.004 on average" hides which statistic.
Bad: "the two triangles agree to 0.004 on average."
Good: "the mean absolute difference between the two triangles is 0.004."

**R29. Captions are read in isolation: name every plotted quantity, the averaging set, prediction-vs-fit, and reproducibility parameters.**
*Why:* A caption must stand alone.
*Joseph (model):* "generated using the code in listing 1, with `n = 10,000` samples ..."
Bad: "Simulated means."
Good: "Simulated mean squared alignments (each point averages the `R` replications); the curve is the prediction from `zeta-hat`, not a fit. Generated by `notebooks/...`."

**R30. Walk a parameter-sweep figure through its regimes in axis order.**
*Why:* Out-of-order narration fights the figure.
Bad: "the distribution is broad above threshold and degenerate below."
Good: "Below the threshold the distribution is concentrated; just above it, broad and skewed upward; eventually it concentrates near 1."

---

## 7. Attribution and Context

**R31. State precisely what a cited author proved, at the right strength; bound novelty claims with epistemic fencing.**
*Why:* Understated credit weakens the contribution; unfenced priority overstates it. These — not name-origin etymology (a minor mechanics note, R-mech-names) — are Didier's actual attribution patterns.
*Didier (model):* "to the best of our knowledge this paper provides the first mathematical study ..." and the "first proof" framing.
Bad: "Johnstone supplied the normalization sequences." / "we are the first to study this."
Good: "Johnstone (with Johansson) gave the first proof of the Tracy-Widom limit for SCMs." / "to the best of our knowledge, this is the first proof of the threshold."

**R32. Separate history from mathematics; replace a re-narration with a cross-reference.**
*Why:* Interleaved history clutters the math; "cf. Example X" beats repeating it.
Bad: "[full re-narration of the spinning-eigenvector example]."
Good: "(cf. Example X)."

**R33. Anchor motivation in named application domains and place a striking result in its wider context — in one cited sentence, no machinery.**
*Why:* An honest pointer situates the result without a digression; naming the application domain makes an abstract task concrete (Didier anchors motivation across econometrics, neuroscience, turbulence, climate).
*Operational trigger and test:* when a result is stated without proof and linked to a named general phenomenon (e.g. universality), that connection must be exactly one sentence, must carry a citation, and must introduce no symbol or equation solely for the connection.
Good (wider context): "random-matrix predictions calibrated on Gaussian models track real-data performance, an instance of universality (Couillet & Liao 2022)."
Good (named application domain): "the machine learning task of sorting data into groups" (framing the abstract task by its application), rather than "the abstract clustering problem."

**R34. Justify a claimed mechanism by naming the theorem behind it.**
*Why:* Assert the conclusion only after citing the law that gives it.
*Didier:* `\GD{...by the classical strong law of large numbers (SLLN),...}`
Bad: "classical theory guarantees the sample covariance is close to `Sigma`."
Good: "by the strong law of large numbers (SLLN), the sample covariance matrix is close to `Sigma`."

---

## 8. Mechanics

**R35. No em dashes.** Use a comma, semicolon, colon, or a new sentence. (House rule.)

**R36. US spelling throughout** ("behavior", "neighbor", "normalize"). (House rule.)

**R37. Reference every display by number; keep every label unique.**
*Why:* A bare "the entrywise formula" and duplicate labels break compilation and traceability.
Bad: "the entrywise formula."
Good: "the entrywise formula of (eq:scm-entrywise)."

**R38. Prefix asides with a signpost from the allowed set so they are not read as part of a definition.**
*Why:* A bare remark abutting a definition reads as part of it.
*Operational test:* any sentence adjacent to a definition/theorem that is not part of it must begin with one of: "Note that", "Remark:", "Aside:", "For intuition,". A bare remark abutting a definition fails.
Bad: "[squaring remark stated bare, against the definition]."
Good: "Note that [squaring remark]."

**R39. Use explicit connectives to mark logical moves, including the result-to-meaning bridge.**
*Why:* "Nevertheless", "Moreover", "In fact", "By contrast" tell the reader the turn before it happens; "Thus,"/"In summary," after a cluster of results says what the math means for the report's story — the plain-spoken engagement device both models use.
*Didier:* inserted `\GD{Nevertheless,}` at the classical-to-modern pivot and `\GD{Moreover,}` between the two conventions; uses "By contrast"/"whereas" to pair what holds against what fails.
*Joseph:* "Thus, while the diagonal entries of `M(h)` alone cannot recover the full dependence structure, the eigendomain is able to do so."
Good: pair a result that holds against one that does not with "By contrast"/"whereas"; close a results cluster with a "Thus,"/"In summary," sentence stating the takeaway.

**R40. Keep the load-bearing feature prominent, not buried in a list.**
*Why:* Downstream results rest on it (e.g. the square-root edge behind Tracy-Widom).
*Operational test:* any property a later theorem cites by name (e.g. the square-root edge behind Tracy-Widom) appears in its own sentence, not as a non-final item in a bulleted or comma-separated list, and is referenced by the downstream theorem with a numbered cross-reference.
Bad: [square-root edge as one minor bullet among many].
Good: [square-root edge highlighted in its own sentence as the feature behind the Tracy-Widom limit].

**R-mech-names. Verify name-origin and etymology claims (minor copy-edit note).**
*Why:* A false "named after ___" is easy to refute, but this is a mechanics check, not a headline rule.
Bad: "named after universality."
Good: "named after Tracy and Widom."

---

## 9. Pre-Submission Checklist

Run over **every sentence**. Each must be a clear "yes".

1. Claims only what is proved/cited/simulated, and says which? (R1)
2. Every quantity named precisely (no "the truth"/"the picture"/"the behavior"), and the body of theory named precisely (no bare "classical ___ guarantees")? (R2, R3)
3. Each fact stated once, no clause restating its own start or a neighbor? (R10, R11)
4. Every "it/this/these/both/the two" and definite article has one already-visible antecedent, each grouping member named? (R17)
5. Every nonstandard/load-bearing term defined inline at first use, standard results cited not re-glossed, and every load-bearing formula read back in the shortest correct words? (R22, R23)
6. Every number anchored to its regime (`y`, `p`, `n`, fixed vs growing) at the point it appears? (R27)
7. Spectral facts as limits; every density/support carries its regime qualifier? (R4)
8. Finite-`n` effects kept distinct from limit statements; no concentrating statistic sold as a limit law? (R5, R6)
9. Each theorem/definition immediately preceded by exactly one sentence naming what the result determines or where it is used; every heuristic placed AFTER the result it motivates? (R18, R19)
10. Distinct phenomena in distinct sentences/paragraphs? (R16)
11. Precise words checked; strong-false downgraded to weak-true; escape/exit only for genuine outliers? (R8, R9)
12. Subspace/aggregate claims bounded by their single-component result? (R7)
13. No verb of agency/perception (tell, want, try, decide, see, know, fight, escape, recover) with a non-agent subject unless figuration names its exact content; initial negations rewritten positive? (R20)
14. No sentence over ~35 words or with two-plus coordinating "and"/"but" clauses; subject and main verb within ~12 words; medium rhythm, not staccato? (R15)
15. No non-informative adjective (deletes without weakening a claim), no lab jargon (pipeline/validated/tune); no forward-promise the body already delivers; the main result still stated concretely up front? (R12, R13, R14)
16. Captions name quantity, averaging set, prediction-vs-fit, and reproducibility parameters, readable alone? (R29)
17. Attribution at correct strength; novelty fenced ("to the best of our knowledge", "the first proof"); history separated from mathematics? (R31, R32)
18. Engagement engines present: abstract follows the funnel with the result stated concretely; each section opens with one orienting sentence; marquee results labeled; each results cluster closes with a "Thus,"/"In summary," bridge? (R-abstract, R-open, R18, R39)
19. One display per formula (regimes merged into a single cases brace, algebraic re-expressions in prose)? (R19b)
20. No em dashes; US spelling; one symbol per concept; unique labels; displays cited by number; name-origins verified? (R35, R36, R26, R37, R-mech-names)
