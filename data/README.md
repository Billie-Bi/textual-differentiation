# Source texts and data preparation

## Purpose

This directory contains the seven cleaned plain-text files used as the exact inputs to the computational pipeline associated with the study **“Computational analysis of textual differentiation across the tripartite structure of Virginia Woolf’s *To the Lighthouse*.”**

The files are included to make the reported analysis reproducible from the segmentation stage onward. They are third-party literary texts, not author-generated research data.

## Included files and sources

| Repository file | Text | Source edition | Access date |
|---|---|---|---|
| `To_the_Lighthouse.txt` | Virginia Woolf, *To the Lighthouse* | [Project Gutenberg Australia](https://gutenberg.net.au/ebooks01/0100101h.html) | 20 September 2025 |
| `Mrs_Dalloway.txt` | Virginia Woolf, *Mrs. Dalloway* | [Project Gutenberg, eBook 71865](https://www.gutenberg.org/ebooks/71865) | 19 May 2025 |
| `Pride_and_Prejudice.txt` | Jane Austen, *Pride and Prejudice* | [Project Gutenberg, eBook 1342](https://www.gutenberg.org/ebooks/1342) | 3 April 2026 |
| `Jane_Eyre.txt` | Charlotte Brontë, *Jane Eyre* | [Project Gutenberg, eBook 1260](https://www.gutenberg.org/ebooks/1260) | 9 April 2026 |
| `Great_Expectations.txt` | Charles Dickens, *Great Expectations* | [Project Gutenberg, eBook 1400](https://www.gutenberg.org/ebooks/1400) | 9 April 2026 |
| `Middlemarch.txt` | George Eliot, *Middlemarch* | [Project Gutenberg, eBook 145](https://www.gutenberg.org/ebooks/145) | 3 April 2026 |
| `Tess_of_the_dUrbervilles.txt` | Thomas Hardy, *Tess of the d’Urbervilles* | [Project Gutenberg, eBook 110](https://www.gutenberg.org/ebooks/110) | 9 April 2026 |

These sources correspond to the editions cited in the associated article.

## Text preparation

The downloaded texts were converted to UTF-8 plain text and prepared consistently before analysis. Repository headers, legal notices, navigation material, tables of contents, and other non-novel front or back matter were removed. The complete literary text used in the study was retained.

For `To_the_Lighthouse.txt`, the headings `THE WINDOW`, `TIME PASSES`, and `THE LIGHTHOUSE` and the chapter-number lines were retained because `code/text_segmentation.py` uses them to identify the novel’s three parts and chapters.

The six reference novels are processed as complete cleaned texts by `code/reference_corpus_extractor.py`. Their novel-level indicator means are written to:

```text
Textual_Indicators_output/reference_corpus/reference_corpus_textual_indicators.csv
```

## Required filenames

The scripts use the filenames listed above as relative paths. Do not rename the text files unless the corresponding configuration paths in the scripts are also updated.

## Licensing and reuse

The source novels are included solely to document the exact analytical inputs and support computational reproducibility. They were obtained from the public repositories listed above.

The MIT License for the repository’s code and the CC BY 4.0 license for author-generated numerical outputs do not apply to these source novels and do not supersede the terms, notices, or public-domain determinations applicable to the original works, source editions, repositories, or users’ jurisdictions.

Users who redistribute or reuse the source texts are responsible for confirming that such use is permitted in their jurisdiction and for complying with the applicable source-repository terms.

## Generated data

Running the pipeline produces intermediate and final data under `Textual_Indicators_output/`, including:

- paragraph segmentation with part and chapter identifiers;
- paragraph-level narrative-semantic and lexicogrammatical indicators;
- chapter-level aggregates;
- reference-text novel-level means;
- statistical test outputs;
- figure source data;
- publication tables and Supporting Information files.

The committed output directories preserve the files used in the associated article.
