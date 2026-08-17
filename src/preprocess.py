from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
# Matches lowercase alphabetic words with an optional internal apostrophe.
#
# Examples:
#   king       -> kept
#   don't      -> kept
#   king's     -> kept
#   hello!     -> "hello"
#
# Punctuation is otherwise discarded.
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """
    Return the repository root.

    preprocess.py lives in:
        <project_root>/src/preprocess.py

    so parents[1] is the project root.
    """
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    """
    Load the frozen experiment configuration from JSON.

    The config is the single source of truth for preprocessing and
    experiment parameters. We avoid duplicating values such as the seed,
    block size, or train/test fractions inside Python.
    """
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """
    Fail early if the experimental configuration is inconsistent.

    Catching errors here is much better than silently producing a dataset
    with different assumptions from those described in the paper.
    """
    required = {
        "seed",
        "train_fraction",
        "validation_fraction",
        "test_fraction",
        "training_subset_fractions",
        "min_token_frequency",
        "training_block_tokens",
        "max_history_length",
    }

    missing = required - config.keys()

    if missing:
        raise ValueError(
            f"Missing required config fields: {sorted(missing)}"
        )

    total_split = (
        config["train_fraction"]
        + config["validation_fraction"]
        + config["test_fraction"]
    )

    if not math.isclose(total_split, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "train_fraction + validation_fraction + "
            "test_fraction must equal 1."
        )

    if config["min_token_frequency"] < 1:
        raise ValueError("min_token_frequency must be >= 1.")

    if config["training_block_tokens"] < 1:
        raise ValueError("training_block_tokens must be >= 1.")


# ---------------------------------------------------------------------------
# Corpus loading + tokenisation
# ---------------------------------------------------------------------------

def load_text(path: Path) -> str:
    """
    Load the raw corpus exactly as stored on disk.
    """
    with path.open("r", encoding="utf-8") as file:
        return file.read()


def tokenize(text: str) -> list[str]:
    """
    Convert raw text into the fixed word-level representation.

    Processing:
        1. lowercase
        2. extract alphabetic words
        3. retain internal apostrophes

    This function must remain frozen once experiments begin because
    tokenisation directly changes vocabulary and context sparsity.
    """
    return TOKEN_PATTERN.findall(text.lower())


# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------

def split_tokens(
    tokens: list[str],
    train_fraction: float,
    validation_fraction: float,
) -> tuple[list[str], list[str], list[str]]:
    """
    Sequentially split the corpus.

    IMPORTANT:
    We do not randomly shuffle individual tokens before splitting.

    This preserves local language structure and prevents nearby fragments
    from being scattered across training and held-out sets.
    """
    n_tokens = len(tokens)

    train_end = int(n_tokens * train_fraction)
    validation_end = train_end + int(
        n_tokens * validation_fraction
    )

    train_tokens = tokens[:train_end]
    validation_tokens = tokens[train_end:validation_end]
    test_tokens = tokens[validation_end:]

    return train_tokens, validation_tokens, test_tokens


# ---------------------------------------------------------------------------
# Fixed vocabulary
# ---------------------------------------------------------------------------

def build_vocabulary(
    train_tokens: list[str],
    min_frequency: int = 2,
) -> dict[str, int]:
    """
    Build vocabulary STRICTLY from the full training partition.

    Words with frequency below min_frequency are not included and will
    later map to <UNK>.

    The vocabulary is built once from the full D_train and then frozen for
    every training-size condition.

    This prevents vocabulary size |V| from changing when m changes.
    """
    counts = Counter(train_tokens)

    # Reserve integer ID 0 for the unknown token.
    vocabulary = {"<UNK>": 0}

    # Sorting produces deterministic integer IDs across runs.
    for token in sorted(counts):
        if counts[token] >= min_frequency:
            vocabulary[token] = len(vocabulary)

    return vocabulary


def apply_vocabulary(
    tokens: list[str],
    vocabulary: dict[str, int],
) -> list[str]:
    """
    Apply the fixed vocabulary while retaining readable token strings.

    Rare or unseen words become the literal string "<UNK>".

    These readable versions are written to train.txt, val.txt, and test.txt.
    """
    return [
        token if token in vocabulary else "<UNK>"
        for token in tokens
    ]


def encode_tokens(
    tokens: list[str],
    vocabulary: dict[str, int],
) -> list[int]:
    """
    Convert already vocabulary-normalised token strings to integer IDs.

    Integer IDs are used in dataset_blocks.json and by the model because
    they are compact and unambiguous.
    """
    return [vocabulary[token] for token in tokens]


# ---------------------------------------------------------------------------
# Training blocks
# ---------------------------------------------------------------------------

def make_blocks(
    tokens: list[int],
    block_size: int,
) -> list[list[int]]:
    """
    Split training tokens into contiguous blocks.

    Blocks preserve local word order while allowing the blocks themselves
    to be shuffled.

    Keeping blocks separate is scientifically important:
    n-grams must NEVER cross boundaries between artificially joined blocks.
    """
    return [
        tokens[i:i + block_size]
        for i in range(0, len(tokens), block_size)
    ]


def shuffle_blocks(
    blocks: list[list[int]],
    seed: int,
) -> list[list[int]]:
    """
    Deterministically shuffle training blocks.

    A local Random instance is used rather than random.seed(), so this
    operation does not modify Python's global random state.
    """
    shuffled = blocks.copy()

    rng = random.Random(seed)
    rng.shuffle(shuffled)

    return shuffled


def calculate_subset_block_counts(
    total_blocks: int,
    fractions: list[float],
) -> dict[str, int]:
    """
    Determine how many shuffled blocks belong to each nested subset.

    We store only these block counts rather than duplicating the actual
    blocks for every 5%, 10%, 20%, ... subset in dataset_blocks.json.

    Example:
        5%  -> first 5 blocks
        10% -> first 10 blocks

    Therefore:
        D_5% ⊆ D_10% ⊆ ... ⊆ D_100%
    """
    subset_counts = {}

    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError(
                "Training subset fractions must lie in (0, 1]."
            )

        if total_blocks == 0:
            count = 0
        elif math.isclose(fraction, 1.0):
            # Guarantee that 100% really means every block.
            count = total_blocks
        else:
            count = max(
                1,
                int(round(total_blocks * fraction)),
            )

        label = f"{int(round(fraction * 100))}%"
        subset_counts[label] = count

    return subset_counts


def get_training_subset(
    dataset_payload: dict,
    subset_label: str,
) -> list[list[int]]:
    """
    Reconstruct one nested training subset from dataset_blocks.json.

    Example:
        get_training_subset(payload, "20%")
    """
    if subset_label not in dataset_payload["subset_block_counts"]:
        raise KeyError(
            f"Unknown training subset: {subset_label}"
        )

    n_blocks = dataset_payload["subset_block_counts"][subset_label]

    return dataset_payload["train_blocks"][:n_blocks]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_token_file(
    path: Path,
    tokens: list[str],
) -> None:
    """
    Write tokens as human-readable whitespace-separated text.

    These files exist for auditing and inspection, not as the canonical
    machine-readable experimental representation.
    """
    path.write_text(
        " ".join(tokens),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = get_project_root()

    config_path = project_root / "experiments" / "config.json"

    raw_data_path = (
        project_root
        / "data"
        / "raw"
        / "Shakespeare's complete works.txt"
    )

    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1. Load frozen experiment configuration
    # ---------------------------------------------------------------------
    config = load_config(config_path)

    seed = config["seed"]
    min_frequency = config["min_token_frequency"]
    block_size = config["training_block_tokens"]
    subset_fractions = config["training_subset_fractions"]

    # ---------------------------------------------------------------------
    # 2. Load + tokenize complete raw corpus
    # ---------------------------------------------------------------------
    print("Loading raw corpus...")

    raw_text = load_text(raw_data_path)
    tokens = tokenize(raw_text)

    print(f"Total tokens extracted: {len(tokens):,}")

    # Save the cleaned/tokenised corpus BEFORE <UNK> mapping.
    write_token_file(
        processed_dir / "cleaned_sequences.txt",
        tokens,
    )

    # ---------------------------------------------------------------------
    # 3. Sequential train / validation / test split
    # ---------------------------------------------------------------------
    raw_train, raw_val, raw_test = split_tokens(
        tokens,
        train_fraction=config["train_fraction"],
        validation_fraction=config["validation_fraction"],
    )

    print(
        "Raw split counts -> "
        f"Train: {len(raw_train):,} | "
        f"Val: {len(raw_val):,} | "
        f"Test: {len(raw_test):,}"
    )

    # ---------------------------------------------------------------------
    # 4. Build vocabulary ONLY from D_train
    # ---------------------------------------------------------------------
    print(
        f"Building vocabulary from D_train "
        f"(C(w) >= {min_frequency})..."
    )

    vocabulary = build_vocabulary(
        raw_train,
        min_frequency=min_frequency,
    )

    print(
        f"Frozen vocabulary size: "
        f"{len(vocabulary):,} including <UNK>"
    )

    with (processed_dir / "vocab.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vocabulary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------------------
    # 5. Apply frozen vocabulary to ALL partitions
    # ---------------------------------------------------------------------
    train_tokens = apply_vocabulary(raw_train, vocabulary)
    val_tokens = apply_vocabulary(raw_val, vocabulary)
    test_tokens = apply_vocabulary(raw_test, vocabulary)

    # Human-readable audit files.
    write_token_file(
        processed_dir / "train.txt",
        train_tokens,
    )

    write_token_file(
        processed_dir / "val.txt",
        val_tokens,
    )

    write_token_file(
        processed_dir / "test.txt",
        test_tokens,
    )

    # ---------------------------------------------------------------------
    # 6. Encode tokens as integer IDs
    # ---------------------------------------------------------------------
    train_ids = encode_tokens(train_tokens, vocabulary)
    val_ids = encode_tokens(val_tokens, vocabulary)
    test_ids = encode_tokens(test_tokens, vocabulary)

    # ---------------------------------------------------------------------
    # 7. Create + deterministically shuffle TRAINING blocks
    # ---------------------------------------------------------------------
    train_blocks = make_blocks(
        train_ids,
        block_size=block_size,
    )

    shuffled_train_blocks = shuffle_blocks(
        train_blocks,
        seed=seed,
    )

    # ---------------------------------------------------------------------
    # 8. Define nested subset sizes
    # ---------------------------------------------------------------------
    subset_block_counts = calculate_subset_block_counts(
        total_blocks=len(shuffled_train_blocks),
        fractions=subset_fractions,
    )

    # ---------------------------------------------------------------------
    # 9. Save canonical machine-readable dataset
    # ---------------------------------------------------------------------
    #
    # Validation and test are intentionally retained as one contiguous
    # sequence each. They are not shuffled and do not need artificial
    # 1000-token boundaries.
    #
    # Training requires blocks only because blocks are used to create
    # deterministic nested training-size subsets.
    # ---------------------------------------------------------------------
    dataset_payload = {
        "metadata": {
            "seed": seed,
            "train_fraction": config["train_fraction"],
            "validation_fraction": config["validation_fraction"],
            "test_fraction": config["test_fraction"],
            "min_token_frequency": min_frequency,
            "training_block_tokens": block_size,
            "vocabulary_size": len(vocabulary),
            "raw_token_count": len(tokens),
            "train_token_count": len(train_ids),
            "validation_token_count": len(val_ids),
            "test_token_count": len(test_ids),
        },

        # Shuffled exactly once, using the fixed seed.
        "train_blocks": shuffled_train_blocks,

        # Example: {"5%": 34, "10%": 68, ...}
        "subset_block_counts": subset_block_counts,

        # Held-out sets remain sequential.
        "val_sequences": [val_ids],
        "test_sequences": [test_ids],
    }

    with (processed_dir / "dataset_blocks.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(dataset_payload, file)

    # ---------------------------------------------------------------------
    # 10. Summary for sanity checking
    # ---------------------------------------------------------------------
    train_unk_count = train_tokens.count("<UNK>")
    val_unk_count = val_tokens.count("<UNK>")
    test_unk_count = test_tokens.count("<UNK>")

    def unk_rate(count: int, total: int) -> float:
        return count / total if total else 0.0

    print("\n--- Preprocessing Summary ---")

    print(f"Vocabulary size: {len(vocabulary):,}")

    print(
        f"Train <UNK> rate: "
        f"{unk_rate(train_unk_count, len(train_tokens)):.4%}"
    )

    print(
        f"Val <UNK> rate:   "
        f"{unk_rate(val_unk_count, len(val_tokens)):.4%}"
    )

    print(
        f"Test <UNK> rate:  "
        f"{unk_rate(test_unk_count, len(test_tokens)):.4%}"
    )

    print(
        f"Training blocks: "
        f"{len(shuffled_train_blocks):,}"
    )

    print("\nNested training subsets:")

    for label, n_blocks in subset_block_counts.items():
        subset = shuffled_train_blocks[:n_blocks]

        token_count = sum(
            len(block)
            for block in subset
        )

        print(
            f"  {label:>4}: "
            f"{n_blocks:>4} blocks | "
            f"{token_count:>8,} tokens"
        )

    print(
        "\nProcessed outputs written to:",
        processed_dir,
    )


if __name__ == "__main__":
    main()