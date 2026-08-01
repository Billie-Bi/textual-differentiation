import os
import re
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

INPUT_FILEPATH = "data/To_the_Lighthouse.txt"
OUTPUT_DIRECTORY = "Textual_Indicators_output/text_segmentation"
OUTPUT_FILEPATH = os.path.join(OUTPUT_DIRECTORY, "to_the_lighthouse_paragraphs.csv")

TEXT_ENCODING = "utf-8"
CSV_ENCODING = "utf-8"
PART_PATTERN = r"(THE WINDOW|TIME PASSES|THE LIGHTHOUSE)"
CHAPTER_PATTERN = r"^\s*(\d+)\s*$"
PARAGRAPH_PATTERN = r"\n\s*\n"
OUTPUT_COLUMNS = ["part", "chapter_id", "para_idx", "text"]


def extract_novel_parts(novel_text):
    split_components = re.split(PART_PATTERN, novel_text)
    return [
        (split_components[index].title(), split_components[index + 1].strip())
        for index in range(1, len(split_components), 2)
        if split_components[index + 1].strip()
    ]


def extract_part_chapters(part_text):
    split_components = re.split(CHAPTER_PATTERN, part_text, flags=re.MULTILINE)
    return [
        (int(split_components[index]), split_components[index + 1].strip())
        for index in range(1, len(split_components), 2)
        if split_components[index + 1].strip()
    ]


def build_paragraph_dataframe(novel_text):
    paragraph_records = []
    for part_name, part_text in extract_novel_parts(novel_text):
        for chapter_id, chapter_text in extract_part_chapters(part_text):
            paragraphs = [paragraph.strip() for paragraph in re.split(PARAGRAPH_PATTERN, chapter_text) if paragraph.strip()]
            paragraph_records.extend(
                {
                    "part": part_name,
                    "chapter_id": chapter_id,
                    "para_idx": paragraph_index,
                    "text": paragraph_text,
                }
                for paragraph_index, paragraph_text in enumerate(paragraphs, start=1)
            )
    return pd.DataFrame(paragraph_records, columns=OUTPUT_COLUMNS)


def run_pipeline():
    if not os.path.exists(INPUT_FILEPATH):
        raise FileNotFoundError(f"[ERROR] Input file missing: {INPUT_FILEPATH}")

    print("[PROCESS] Segmenting target novel...")
    with open(INPUT_FILEPATH, encoding=TEXT_ENCODING) as input_file:
        paragraph_dataframe = build_paragraph_dataframe(input_file.read())

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    paragraph_dataframe.to_csv(OUTPUT_FILEPATH, index=False, encoding=CSV_ENCODING)
    print(f"[RESULT] Saved {len(paragraph_dataframe)} paragraphs to: {OUTPUT_FILEPATH}")


if __name__ == "__main__":
    run_pipeline()
