import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 14
sns.set_theme(style="whitegrid")

INPUT_NARRATIVE_SEMANTIC_FILEPATH = "Textual_Indicators_output/narrative-semantic_indicators/01_Narrative-Semantic_Indicators_Raw_Paragraph_Indicators.csv"
INPUT_LEXICOGRAMMATICAL_FILEPATH = "Textual_Indicators_output/lexicogrammatical_indicators/01a_Lexicogrammatical_Indicators_Raw_Paragraph_Indicators.csv"
OUTPUT_DIRECTORY = "Textual_Indicators_output/integrated_analysis"

OUTPUT_IMPORTANCE_SCORES_CSV = os.path.join(OUTPUT_DIRECTORY, "01_Textual_Indicator_Importance_Scores.csv")
OUTPUT_MODEL_DIAGNOSTICS_CSV = os.path.join(OUTPUT_DIRECTORY, "02_Random_Forest_Model_Diagnostics.csv")
OUTPUT_IMPORTANCE_FIGURE = os.path.join(OUTPUT_DIRECTORY, "Fig1_Textual_Indicator_Importance.png")
OUTPUT_IMPORTANCE_FIGURE_DATA = os.path.join(OUTPUT_DIRECTORY, "Fig1_Textual_Indicator_Importance_Source_Data.csv")

MERGE_KEY_COLUMNS = ["part", "chapter_id", "para_idx"]

NARRATIVE_SEMANTIC_TARGET_COLUMNS = MERGE_KEY_COLUMNS + [
    "event_density",
    "perspective_shift",
    "psychological_density",
    "temporal_density",
]

LEXICOGRAMMATICAL_TARGET_COLUMNS = MERGE_KEY_COLUMNS + [
    "pos_ratio_adj_adv_to_verb",
    "avg_dependency_depth",
    "repeated_words_per_100_tokens",
    "parallel_ratio",
]

TARGET_FEATURE_COLUMNS = [
    "event_density",
    "perspective_shift",
    "psychological_density",
    "temporal_density",
    "pos_ratio_adj_adv_to_verb",
    "avg_dependency_depth",
    "repeated_words_per_100_tokens",
    "parallel_ratio",
]

METRIC_DISPLAY_LABELS = {
    "event_density": "Event Density",
    "perspective_shift": "Perspective Shift",
    "psychological_density": "Psychological Density",
    "temporal_density": "Temporal Density",
    "pos_ratio_adj_adv_to_verb": "POS Ratio of Adjectives and Adverbs to Verbs",
    "avg_dependency_depth": "Average Dependency Depth",
    "repeated_words_per_100_tokens": "Repeated Words per 100 Tokens",
    "parallel_ratio": "Parallel Ratio",
}


def load_dataframe_with_required_columns(filepath, required_columns):
    dataframe = pd.read_csv(filepath)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"[ERROR] Missing columns in {filepath}: {missing_columns}")
    return dataframe[required_columns].copy()


def load_and_validate_merged_dataset(narrative_semantic_filepath, lexicogrammatical_filepath):
    narrative_semantic_dataframe = load_dataframe_with_required_columns(
        narrative_semantic_filepath, NARRATIVE_SEMANTIC_TARGET_COLUMNS
    )
    lexicogrammatical_dataframe = load_dataframe_with_required_columns(
        lexicogrammatical_filepath, LEXICOGRAMMATICAL_TARGET_COLUMNS
    )

    merged_dataframe = pd.merge(
        narrative_semantic_dataframe,
        lexicogrammatical_dataframe,
        on=MERGE_KEY_COLUMNS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched_records = merged_dataframe[merged_dataframe["_merge"] != "both"]
    if not unmatched_records.empty:
        merge_counts = unmatched_records["_merge"].value_counts().to_dict()
        raise ValueError(f"[ERROR] Narrative-semantic and lexicogrammatical paragraph rows do not match one-to-one: {merge_counts}")

    merged_dataframe = merged_dataframe.drop(columns="_merge")
    missing_values_summary = merged_dataframe[["part"] + TARGET_FEATURE_COLUMNS].isna().sum()
    missing_values_summary = missing_values_summary[missing_values_summary > 0]
    
    if not missing_values_summary.empty:
        raise ValueError(f"[ERROR] Missing values found in the model input columns: {missing_values_summary.to_dict()}")

    return merged_dataframe


def compute_random_forest_feature_importance(merged_dataframe):
    feature_data = merged_dataframe[TARGET_FEATURE_COLUMNS]
    target_data = merged_dataframe["part"]

    random_forest_model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        bootstrap=True,
        oob_score=True,
        n_jobs=1,
    )
    random_forest_model.fit(feature_data, target_data)

    importance_dataframe = pd.DataFrame(
        {
            "Indicator": [METRIC_DISPLAY_LABELS[feature] for feature in TARGET_FEATURE_COLUMNS],
            "Importance": random_forest_model.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)
    importance_dataframe.insert(0, "Rank", range(1, len(importance_dataframe) + 1))

    out_of_bag_probabilities = random_forest_model.oob_decision_function_
    valid_out_of_bag_mask = np.isfinite(out_of_bag_probabilities).all(axis=1) & (
        out_of_bag_probabilities.sum(axis=1) > 0
    )
    
    if valid_out_of_bag_mask.any():
        out_of_bag_predictions = random_forest_model.classes_[
            np.argmax(out_of_bag_probabilities[valid_out_of_bag_mask], axis=1)
        ]
        out_of_bag_balanced_accuracy = balanced_accuracy_score(
            target_data.to_numpy()[valid_out_of_bag_mask], out_of_bag_predictions
        )
    else:
        out_of_bag_balanced_accuracy = np.nan

    diagnostics_dataframe = pd.DataFrame(
        [
            {
                "Number of paragraphs": len(merged_dataframe),
                "Number of indicators": len(TARGET_FEATURE_COLUMNS),
                "Number of trees": random_forest_model.n_estimators,
                "Class weighting": "balanced",
                "OOB accuracy": random_forest_model.oob_score_,
                "OOB balanced accuracy": out_of_bag_balanced_accuracy,
            }
        ]
    )
    return importance_dataframe, diagnostics_dataframe


def save_dataframe_to_numeric_csv(dataframe, filepath):
    dataframe.to_csv(filepath, index=False, encoding="utf-8", float_format="%.12g")


def render_feature_importance_plot(importance_dataframe, plot_filepath):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Indicator", data=importance_dataframe, palette="magma")

    plt.xlabel("Mean Decrease in Impurity", fontsize=13)
    plt.ylabel("")
    plt.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
    plt.close()


def run_pipeline():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    print("[PROCESS] Loading and validating textual indicator data...")
    merged_dataset_dataframe = load_and_validate_merged_dataset(
        INPUT_NARRATIVE_SEMANTIC_FILEPATH, INPUT_LEXICOGRAMMATICAL_FILEPATH
    )

    print("[PROCESS] Computing textual indicator importance...")
    importance_results_dataframe, model_diagnostics_dataframe = compute_random_forest_feature_importance(merged_dataset_dataframe)
    
    out_of_bag_accuracy = model_diagnostics_dataframe.loc[0, "OOB accuracy"]
    out_of_bag_balanced_accuracy = model_diagnostics_dataframe.loc[0, "OOB balanced accuracy"]
    print(f"[PROCESS] Random-forest out-of-bag accuracy: {out_of_bag_accuracy:.4f}")
    print(f"[PROCESS] Random-forest out-of-bag balanced accuracy: {out_of_bag_balanced_accuracy:.4f}")

    print("[PROCESS] Exporting importance scores and model diagnostics...")
    save_dataframe_to_numeric_csv(importance_results_dataframe, OUTPUT_IMPORTANCE_SCORES_CSV)
    save_dataframe_to_numeric_csv(importance_results_dataframe, OUTPUT_IMPORTANCE_FIGURE_DATA)
    save_dataframe_to_numeric_csv(model_diagnostics_dataframe, OUTPUT_MODEL_DIAGNOSTICS_CSV)

    print("[PROCESS] Rendering textual indicator importance visualization...")
    render_feature_importance_plot(importance_results_dataframe, OUTPUT_IMPORTANCE_FIGURE)

    print(f"[RESULT] Analysis completed successfully. Outputs saved to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    run_pipeline()