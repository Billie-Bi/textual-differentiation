# Computational Analysis of Textual Differentiation in *To the Lighthouse*

This repository contains the complete reproducibility materials for the study **“Computational analysis of textual differentiation across the tripartite structure of Virginia Woolf’s *To the Lighthouse*.”**

The repository name is `textual-differentiation`. It includes the six analysis scripts used in the article, the exact cleaned text files supplied to the pipeline, the final Python environment specification, and the complete sequence of generated outputs reported or used in the manuscript.

## Study overview

The study examines how eight computational textual indicators vary across the three parts of *To the Lighthouse*—“The Window,” “Time Passes,” and “The Lighthouse.” The paragraph is the primary analytical unit. Chapter-level aggregation is used for sensitivity analyses, boxplots, and longitudinal visualization.

The indicators are treated as reproducible textual proxies with explicitly limited operational meanings. They are not treated as direct equivalents of narratological categories, psychological states, aesthetic value, or authorial intention.

## Analytical indicators

### Narrative-semantic indicators

- **Event Density**: proportion of sentences satisfying a rule-based definition of externally anchored event cues.
- **Perspective Shift**: sum of semantic distance from the preceding paragraph and within-paragraph pronoun-transition rate.
- **Psychological Density**: frequency of lemmatized items in a WordNet-expanded psychological lexical set, normalized by alphabetic token count.
- **Temporal Density**: frequency of lemmatized items in a WordNet-expanded temporal lexical set, normalized by alphabetic token count.

### Lexicogrammatical indicators

- **POS Ratio of Adjectives and Adverbs to Verbs**: adjective and adverb count divided by lexical verb count.
- **Average Dependency Depth**: mean sentence-level maximum dependency-tree depth normalized by alphabetic sentence length.
- **Repeated Words per 100 Tokens**: occurrences of recurring content-word forms per 100 alphabetic tokens.
- **Parallel Ratio**: coordinating conjunctions governed by verbs divided by sentence count.

## Reference framework

The target text is contextualized using:

- Virginia Woolf’s *Mrs. Dalloway* as an intra-author reference;
- *Pride and Prejudice*, *Jane Eyre*, *Great Expectations*, *Middlemarch*, and *Tess of the d’Urbervilles* as a selected 19th-century reference set.

The five 19th-century novels provide a descriptive minimum-to-maximum reference band. They are not treated as a statistically representative sample of 19th-century fiction, a confidence interval, or a population estimate.

## Repository structure

```text
textual-differentiation/
├── README.md
├── environment.yml
├── LICENSE
├── DATA_LICENSE.md
├── CITATION.cff
├── .gitignore
├── code/
│   ├── text_segmentation.py
│   ├── reference_corpus_extractor.py
│   ├── narrative-semantic_indicators.py
│   ├── lexicogrammatical_indicators.py
│   ├── chapter_level_robustness.py
│   └── generate_publication_tables.py
├── data/
│   ├── README.md
│   ├── To_the_Lighthouse.txt
│   ├── Mrs_Dalloway.txt
│   ├── Pride_and_Prejudice.txt
│   ├── Jane_Eyre.txt
│   ├── Great_Expectations.txt
│   ├── Middlemarch.txt
│   └── Tess_of_the_dUrbervilles.txt
└── Textual_Indicators_output/
    ├── text_segmentation/
    ├── reference_corpus/
    ├── narrative-semantic_indicators/
    ├── lexicogrammatical_indicators/
    ├── chapter_level_robustness/
    └── publication_tables/
```

The output directories intentionally retain intermediate, statistical, visualization, and publication-stage files. Some numerical information is duplicated across raw indicators, chapter-level aggregates, figure source data, and publication tables so that every stage of the workflow can be inspected independently.

## Included text data

The `data/` directory contains the seven cleaned plain-text files used in the final analysis. These files preserve the exact analytical inputs, including the part and chapter headings required by the segmentation procedure. Repository headers, navigation material, tables of contents, legal notices, and other non-novel front or back matter were removed before analysis.

The source editions, access information, file preparation, and licensing limitations are documented in [`data/README.md`](data/README.md). The repository’s CC BY 4.0 license for author-generated data does not replace or modify the terms or public-domain status applicable to the source novels.

## Environment setup

The final analysis was conducted in Python 3.10.20. Create the environment from the supplied file:

```bash
conda env create -f environment.yml
conda activate textual_indicators
python -m nltk.downloader wordnet omw-1.4
```

The principal package versions are:

| Package | Version |
|---|---:|
| spaCy | 3.8.14 |
| en_core_web_sm | 3.8.0 |
| sentence-transformers | 5.6.0 |
| PyTorch | 2.13.0 |
| NLTK | 3.10.0 |
| NumPy | 1.25.2 |
| pandas | 2.3.3 |
| SciPy | 1.15.3 |
| statsmodels | 0.14.6 |
| scikit-learn | 1.7.1 |
| umap-learn | 0.5.12 |
| Matplotlib | 3.10.9 |
| seaborn | 0.13.2 |

## Running the pipeline

Clone the repository and run all commands from its root directory:

```bash
git clone https://github.com/USERNAME/textual-differentiation.git
cd textual-differentiation
```

Replace `USERNAME` with the repository owner’s GitHub username.

The scripts write their outputs to `Textual_Indicators_output/`. Rerunning the pipeline will regenerate or overwrite the corresponding committed output files.

### 1. Segment *To the Lighthouse*

```bash
python code/text_segmentation.py
```

Primary output:

```text
Textual_Indicators_output/text_segmentation/to_the_lighthouse_paragraphs.csv
```

### 2. Extract reference-text indicators

```bash
python code/reference_corpus_extractor.py
```

Primary output:

```text
Textual_Indicators_output/reference_corpus/reference_corpus_textual_indicators.csv
```

### 3. Extract target-text indicators and generate the primary analyses

```bash
python code/narrative-semantic_indicators.py
python code/lexicogrammatical_indicators.py
```

These scripts generate paragraph-level indicators, descriptive statistics, Spearman correlations, Kruskal–Wallis tests, Mann–Whitney U tests, chapter-level boxplots, chapter-trend figures, source data for the figures, and the exploratory chapter-level UMAP projection.

The UMAP projection uses `n_neighbors=10`, `min_dist=0.3`, `n_components=2`, and `random_state=42`; all remaining settings use package defaults.

### 4. Run chapter-level sensitivity analyses

```bash
python code/chapter_level_robustness.py
```

This script aggregates the 496 paragraph observations into 43 chapter observations: 19 for “The Window,” 10 for “Time Passes,” and 14 for “The Lighthouse.” It repeats the descriptive, correlation, omnibus, and pairwise analyses using chapter means.

### 5. Generate publication tables and Supporting Information files

```bash
python code/generate_publication_tables.py
```

Primary outputs:

```text
Textual_Indicators_output/publication_tables/
├── Table 1. Summary of narrative-semantic indicators and reference comparisons.csv
├── Table 2. Summary of lexicogrammatical indicators and reference comparisons.csv
├── S1 File. Comprehensive descriptive statistics for narrative-semantic and lexicogrammatical indicators.csv
├── S2 File. Kruskal-Wallis test statistics for the eight indicators across the three parts.csv
├── S3 File. Pairwise Mann-Whitney U tests for the eight indicators with Benjamini-Hochberg FDR correction.csv
└── S4 File. Spearman correlations among the computational textual indicators.csv
```

## Output directories

### `text_segmentation/`

Contains the 496 paragraph observations extracted from *To the Lighthouse*, with part, chapter, paragraph-order, and text fields.

### `reference_corpus/`

Contains the novel-level means of the eight indicators for *Mrs. Dalloway* and the five selected 19th-century novels.

### `narrative-semantic_indicators/`

Contains raw paragraph-level narrative-semantic indicators, descriptive statistics, correlation results, omnibus and pairwise tests, two figures, and the source data for both figures.

### `lexicogrammatical_indicators/`

Contains raw paragraph- and chapter-level lexicogrammatical indicators, descriptive statistics, correlation results, omnibus and pairwise tests, three figures, and the source data for all three figures.

### `chapter_level_robustness/`

Contains the combined chapter-level means for all eight indicators, chapter-level descriptive and inferential results, and an additional standardized diagnostic boxplot with its source data.

### `publication_tables/`

Contains Tables 1 and 2 and the four Supporting Information files supplied with the article.

## Statistical procedures

The paragraph-level and chapter-level analyses use Kruskal–Wallis omnibus tests, two-sided Mann–Whitney U pairwise tests, rank-biserial correlations, Cohen’s *d*, epsilon squared, and Spearman rank correlations. Benjamini–Hochberg false discovery rate correction is applied separately by analysis level and indicator family. Omnibus, pairwise, and correlation testing families are corrected separately.

Chapter-level analyses are sensitivity analyses designed to assess the influence of local dependence among paragraphs. Nonsignificant chapter-level results are not interpreted as evidence that local variation is absent.

## Reproducibility

The committed output files are the results generated for the associated article. They include:

- exact cleaned textual inputs;
- paragraph-level indicator data used in the primary tests;
- chapter-level indicator means used in the sensitivity analyses;
- reference-text novel-level indicator values;
- complete descriptive, correlation, omnibus, and pairwise statistical outputs;
- source data for every reported figure;
- Tables 1 and 2 and S1–S4 Files;
- the final environment specification and all six analysis scripts.

## Scope

This repository contains only the analyses reported in the associated article. Random Forest feature-importance analysis, graph neural networks, and other exploratory procedures not reported in the final manuscript are intentionally excluded from this reproducibility release.

## Citation

If you use this code or the released data, please cite the associated article and the archived repository release. Initial citation metadata are provided in `CITATION.cff`. Add the final GitHub URL, Zenodo DOI, and article DOI when they become available.

## License

The analysis code is licensed under the MIT License. Author-generated numerical outputs are licensed under CC BY 4.0 as described in `DATA_LICENSE.md`. Those licenses do not replace or modify the terms or public-domain status applicable to the source novels in `data/`.

## Authors

- Yan Pan, College of Foreign Languages and Literature, Jilin Normal University
- Liqi Bi, College of Foreign Languages and Literature, Jilin Normal University

Correspondence: liqi.bi@mails.jlnu.edu.cn
