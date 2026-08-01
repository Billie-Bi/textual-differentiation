import os
import re
import unicodedata
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import spacy
from nltk.corpus import wordnet as nltk_wordnet
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

OUTPUT_DIRECTORY = "Textual_Indicators_output/reference_corpus"
OUTPUT_FILEPATH = os.path.join(OUTPUT_DIRECTORY, "reference_corpus_textual_indicators.csv")

SPACY_MODEL_NAME = "en_core_web_sm"
SPACY_TAGGER_DISABLED_COMPONENTS = ["ner", "parser"]
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
TEXT_ENCODING = "utf-8"
CSV_ENCODING = "utf-8"
CSV_FLOAT_FORMAT = "%.12g"
PARAGRAPH_PATTERN = r"\n\s*\n"
EVENT_DEPENDENCY_LABELS = {"ROOT", "xcomp", "ccomp", "advcl"}
EVENT_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC"}
CONTENT_WORD_POS_LABELS = {"NOUN", "VERB", "ADJ", "ADV"}
PROGRESS_BAR_WIDTH = 80

REFERENCE_NOVEL_FILEPATHS = {
    "Mrs. Dalloway": "data/Mrs_Dalloway.txt",
    "Pride and Prejudice": "data/Pride_and_Prejudice.txt",
    "Middlemarch": "data/Middlemarch.txt",
    "Jane Eyre": "data/Jane_Eyre.txt",
    "Great Expectations": "data/Great_Expectations.txt",
    "Tess of the d'Urbervilles": "data/Tess_of_the_dUrbervilles.txt",
}

PSYCHOLOGICAL_BASE_WORDS = [
    "mind",
    "thought",
    "feeling",
    "memory",
    "introspection",
    "consciousness",
    "awareness",
    "soul",
    "emotion",
    "perception",
    "imagination",
    "reflection",
    "contemplation",
    "intellect",
    "mood",
]
TEMPORAL_BASE_WORDS = [
    "time",
    "year",
    "season",
    "past",
    "future",
    "day",
    "night",
    "moment",
    "hour",
    "age",
    "minute",
    "century",
    "eternity",
    "duration",
    "instant",
    "clock",
    "afternoon",
    "morning",
]

FEATURE_COLUMN_NAMES = [
    "event_density",
    "perspective_shift",
    "psychological_density",
    "temporal_density",
    "pos_ratio_adj_adv_to_verb",
    "avg_dependency_depth",
    "repeated_words_per_100_tokens",
    "parallel_ratio",
]


def build_synonym_lexicon(base_words):
    synonyms = set()
    for word in base_words:
        synsets = nltk_wordnet.synsets(word)
        if synsets:
            for lemma in synsets[0].lemma_names():
                synonyms.add(lemma.lower())
    return synonyms


def clean_paragraph_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"[;—]+", "; ", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def calculate_dependency_tree_depth(token):
    return 1 + max((calculate_dependency_tree_depth(child) for child in token.children), default=0)


def extract_textual_features(
    spacy_document, text_embedding, previous_embedding, spacy_tagger_model, psychological_synonyms, temporal_synonyms
):
    sentences = list(spacy_document.sents)
    num_sentences = max(len(sentences), 1)
    num_words = max(sum(1 for token in spacy_document if token.is_alpha), 1)

    events = [
        sent.text
        for sent in sentences
        if any(token.pos_ == "VERB" and token.dep_ in EVENT_DEPENDENCY_LABELS for token in sent)
        and any(entity.label_ in EVENT_ENTITY_LABELS for entity in sent.ents)
    ]
    event_density = len(events) / num_sentences

    perspective_shift = 0.0
    if previous_embedding is not None:
        distance = cosine(text_embedding, previous_embedding)
        perspective_shift += 0.0 if pd.isna(distance) else float(distance)

    for sent in sentences:
        text_without_quotes = re.sub(r'["“”].*?["“”]|(?<!\w)[\'‘].*?[\'’](?!\w)', "", sent.text)
        sentence_doc = spacy_tagger_model(text_without_quotes)
        pronouns = [token.text.lower() for token in sentence_doc if token.pos_ == "PRON"]
        if len(pronouns) > 1:
            pronoun_shifts = sum(1 for index in range(1, len(pronouns)) if pronouns[index] != pronouns[index - 1])
            perspective_shift += pronoun_shifts / num_sentences

    psychological_density = sum(1 for token in spacy_document if token.lemma_.lower() in psychological_synonyms) / num_words
    temporal_density = sum(1 for token in spacy_document if token.lemma_.lower() in temporal_synonyms) / num_words

    num_adj_adv = sum(1 for token in spacy_document if token.pos_ in ["ADJ", "ADV"])
    num_verbs = max(sum(1 for token in spacy_document if token.pos_ == "VERB"), 1)
    pos_ratio = num_adj_adv / num_verbs

    dependency_depths = [
        calculate_dependency_tree_depth(sent.root) / max(sum(1 for token in sent if token.is_alpha), 1) for sent in sentences
    ]
    avg_dependency_depth = sum(dependency_depths) / max(len(dependency_depths), 1)

    content_words = [token.text.lower() for token in spacy_document if token.pos_ in CONTENT_WORD_POS_LABELS and token.is_alpha]
    word_counts = pd.Series(content_words, dtype="object").value_counts()
    repeated_words = sum(count for count in word_counts if count > 1)
    repeated_words_score = (repeated_words / num_words) * 100

    parallel_conjunctions = sum(1 for token in spacy_document if token.dep_ == "cc" and token.head.pos_ == "VERB")
    parallel_ratio = parallel_conjunctions / num_sentences

    return [
        event_density,
        perspective_shift,
        psychological_density,
        temporal_density,
        pos_ratio,
        avg_dependency_depth,
        repeated_words_score,
        parallel_ratio,
    ]


def extract_novel_features(
    novel_name,
    filepath,
    spacy_full_model,
    spacy_tagger_model,
    embedding_model,
    psychological_synonyms,
    temporal_synonyms,
):
    if not os.path.exists(filepath):
        print(f"[WARN] File not found: {filepath}")
        return None

    print(f"[PROCESS] Processing {novel_name}")
    with open(filepath, encoding=TEXT_ENCODING) as file:
        raw_text = file.read()

    paragraphs = [paragraph.strip() for paragraph in re.split(PARAGRAPH_PATTERN, raw_text) if paragraph.strip()]

    features_list = []
    previous_embedding = None

    for paragraph in tqdm(paragraphs, desc=novel_name, unit="para", ncols=PROGRESS_BAR_WIDTH):
        clean_text = clean_paragraph_text(paragraph)
        if not clean_text:
            continue

        spacy_document = spacy_full_model(clean_text)
        text_embedding = embedding_model.encode(clean_text)
        features = extract_textual_features(
            spacy_document=spacy_document,
            text_embedding=text_embedding,
            previous_embedding=previous_embedding,
            spacy_tagger_model=spacy_tagger_model,
            psychological_synonyms=psychological_synonyms,
            temporal_synonyms=temporal_synonyms,
        )
        features_list.append(features)
        previous_embedding = text_embedding

    return features_list


def run_pipeline():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    print("[PROCESS] Loading NLP and embedding models...")
    spacy_full_model = spacy.load(SPACY_MODEL_NAME)
    spacy_tagger_model = spacy.load(SPACY_MODEL_NAME, disable=SPACY_TAGGER_DISABLED_COMPONENTS)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("[PROCESS] Generating synonym lexicons...")
    psychological_synonyms = build_synonym_lexicon(PSYCHOLOGICAL_BASE_WORDS)
    temporal_synonyms = build_synonym_lexicon(TEMPORAL_BASE_WORDS)

    corpus_summary_records = []

    for novel_name, filepath in REFERENCE_NOVEL_FILEPATHS.items():
        features_list = extract_novel_features(
            novel_name=novel_name,
            filepath=filepath,
            spacy_full_model=spacy_full_model,
            spacy_tagger_model=spacy_tagger_model,
            embedding_model=embedding_model,
            psychological_synonyms=psychological_synonyms,
            temporal_synonyms=temporal_synonyms,
        )
        if features_list:
            novel_means = pd.DataFrame(features_list, columns=FEATURE_COLUMN_NAMES).mean()
            result_row = {"Novel": novel_name}
            result_row.update(novel_means.to_dict())
            corpus_summary_records.append(result_row)

    if not corpus_summary_records:
        raise RuntimeError("[ERROR] No reference corpus results were generated.")

    summary_dataframe = pd.DataFrame(corpus_summary_records)
    summary_dataframe.to_csv(OUTPUT_FILEPATH, index=False, encoding=CSV_ENCODING, float_format=CSV_FLOAT_FORMAT)
    print(f"[RESULT] Reference corpus indicators saved to: {OUTPUT_FILEPATH}")


if __name__ == "__main__":
    run_pipeline()
