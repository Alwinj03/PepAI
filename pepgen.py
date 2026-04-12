"""
pepgen_core.py
==============
PepGen AI – Transformer-Based Conditional Sequence Generator
Core module for ESM-2-based peptide binder generation.

Usage:
    from pepgen_core import load_model, generate_peptides, compute_scores, format_results

    model, tokenizer = load_model()
    results = generate_peptides(model, tokenizer, "MKTAYIAKQRQISFVKSHFSRQ", top_k=50, length=12, num_outputs=5)
    df = format_results(results)
"""

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────

import math
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"

AVAILABLE_MODELS = {
    "ESM2-8M  (fast, low accuracy)":   "facebook/esm2_t6_8M_UR50D",
    "ESM2-35M  (balanced)":            "facebook/esm2_t12_35M_UR50D",
    "ESM2-150M (recommended)":         "facebook/esm2_t30_150M_UR50D",
    "ESM2-650M (high accuracy, slow)": "facebook/esm2_t33_650M_UR50D",
}


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────

def load_model(
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
) -> Tuple[EsmForMaskedLM, AutoTokenizer]:
    """
    Load an ESM-2 masked language model and its tokenizer.

    Args:
        model_name: HuggingFace model identifier. Defaults to ESM2-8M.
        device: 'cuda', 'cpu', or None (auto-detect).

    Returns:
        (model, tokenizer) tuple ready for inference.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()
    model.to(device)

    return model, tokenizer


# ─────────────────────────────────────────────
# Input Processing
# ─────────────────────────────────────────────

def preprocess_input(
    sequence: str,
    tokenizer: AutoTokenizer,
    device: str = "cpu",
) -> dict:
    """
    Tokenize a protein sequence for ESM-2 inference.

    Args:
        sequence: Single-letter amino acid sequence (uppercase).
        tokenizer: Loaded ESM-2 tokenizer.
        device: Device to place tensors on.

    Returns:
        Dictionary of PyTorch tensors (input_ids, attention_mask).
    """
    sequence = sequence.upper().strip()
    tokens = tokenizer(
        sequence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    return {k: v.to(device) for k, v in tokens.items()}


# ─────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────

def _mask_sequence_for_peptide(
    protein_seq: str,
    peptide_length: int,
    tokenizer: AutoTokenizer,
    device: str,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Build a masked input: full protein + fully masked peptide suffix.

    Returns:
        (input_ids tensor, list of masked positions)
    """
    mask_token = tokenizer.mask_token  # <mask>
    masked_peptide = " ".join([mask_token] * peptide_length)
    combined = protein_seq + " " + masked_peptide

    tokens = tokenizer(
        combined,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    input_ids = tokens["input_ids"].to(device)

    # Positions of mask tokens in the sequence (1-indexed, skip [CLS])
    mask_positions = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].tolist()
    return input_ids, mask_positions


def _decode_positions(
    model: EsmForMaskedLM,
    input_ids: torch.Tensor,
    mask_positions: List[int],
    top_k: int,
    tokenizer: AutoTokenizer,
) -> List[str]:
    """
    Iteratively fill each masked position by sampling from top-k logits.

    Returns:
        List of single amino-acid tokens, one per masked position.
    """
    current_ids = input_ids.clone()
    aa_token_ids = tokenizer.convert_tokens_to_ids(AMINO_ACIDS)

    with torch.no_grad():
        for pos in mask_positions:
            logits = model(current_ids).logits          # (1, seq_len, vocab)
            pos_logits = logits[0, pos, :]

            # Zero out non-amino-acid tokens for clean sampling
            mask = torch.full_like(pos_logits, float("-inf"))
            mask[aa_token_ids] = pos_logits[aa_token_ids]

            top_k_logits, top_k_indices = torch.topk(mask, k=min(top_k, len(aa_token_ids)))
            probs = torch.softmax(top_k_logits, dim=-1)
            chosen_idx = top_k_indices[torch.multinomial(probs, num_samples=1).item()]

            current_ids[0, pos] = chosen_idx

    chosen_tokens = [tokenizer.convert_ids_to_tokens(current_ids[0, p].item()) for p in mask_positions]
    return chosen_tokens


def generate_peptides(
    model: EsmForMaskedLM,
    tokenizer: AutoTokenizer,
    protein_sequence: str,
    top_k: int = 50,
    length: int = 12,
    num_outputs: int = 5,
    device: str | None = None,
) -> List[str]:
    """
    Generate candidate peptide binders conditioned on a protein sequence.

    Args:
        model:            Loaded ESM-2 model.
        tokenizer:        Corresponding tokenizer.
        protein_sequence: Target protein as a single-letter AA string.
        top_k:            Number of top-scoring tokens to sample from at each position.
        length:           Length of the generated peptide (number of residues).
        num_outputs:      How many distinct peptide sequences to generate.
        device:           Compute device ('cuda'/'cpu'). Auto-detected if None.

    Returns:
        List of generated peptide strings.
    """
    if device is None:
        device = next(model.parameters()).device.type

    peptides: List[str] = []
    for _ in range(num_outputs):
        input_ids, mask_positions = _mask_sequence_for_peptide(
            protein_sequence, length, tokenizer, device
        )
        tokens = _decode_positions(model, input_ids, mask_positions, top_k, tokenizer)
        peptides.append("".join(tokens))

    return peptides


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def compute_pseudo_perplexity(
    model: EsmForMaskedLM,
    tokenizer: AutoTokenizer,
    peptide: str,
    device: str | None = None,
) -> float:
    """
    Compute pseudo-perplexity (PPL) for a peptide via masked marginals.

    Lower PPL → model finds the sequence more likely / natural.

    Args:
        model:     Loaded ESM-2 model.
        tokenizer: Corresponding tokenizer.
        peptide:   Amino acid string to score.
        device:    Compute device.

    Returns:
        Pseudo-perplexity as a float.
    """
    if device is None:
        device = next(model.parameters()).device.type

    input_ids = tokenizer(peptide, return_tensors="pt")["input_ids"].to(device)
    seq_len = input_ids.shape[1] - 2  # exclude [CLS] and [EOS]

    total_log_prob = 0.0

    with torch.no_grad():
        for pos in range(1, seq_len + 1):          # 1-indexed due to [CLS]
            masked = input_ids.clone()
            masked[0, pos] = tokenizer.mask_token_id

            logits = model(masked).logits
            log_probs = torch.log_softmax(logits[0, pos], dim=-1)

            true_token = input_ids[0, pos].item()
            total_log_prob += log_probs[true_token].item()

    avg_nll = -total_log_prob / seq_len
    return math.exp(avg_nll)


def compute_scores(
    model: EsmForMaskedLM,
    tokenizer: AutoTokenizer,
    peptides: List[str],
    device: str | None = None,
) -> List[float]:
    """
    Score a list of peptides by pseudo-perplexity.

    Args:
        model, tokenizer: Loaded model and tokenizer.
        peptides:         List of peptide strings.
        device:           Compute device.

    Returns:
        List of PPL scores aligned with input peptides.
    """
    return [
        compute_pseudo_perplexity(model, tokenizer, pep, device)
        for pep in peptides
    ]


# ─────────────────────────────────────────────
# Output Formatting
# ─────────────────────────────────────────────

def format_results(
    peptides: List[str],
    scores: List[float] | None = None,
    protein_sequence: str | None = None,
) -> pd.DataFrame:
    """
    Format generated peptides (and optional scores) into a DataFrame.

    Args:
        peptides:         List of peptide strings.
        scores:           Optional PPL scores. If provided, results are sorted ascending.
        protein_sequence: Optional source protein sequence (added as metadata column).

    Returns:
        pandas DataFrame with columns: rank, peptide, length, [score], [protein].
    """
    data = {
        "rank":    list(range(1, len(peptides) + 1)),
        "peptide": peptides,
        "length":  [len(p) for p in peptides],
    }

    if scores is not None:
        data["pseudo_perplexity"] = scores
        df = pd.DataFrame(data).sort_values("pseudo_perplexity").reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
    else:
        df = pd.DataFrame(data)

    if protein_sequence is not None:
        df["target_protein"] = protein_sequence

    return df


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def get_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Validate that a string is a legal amino-acid sequence.

    Returns:
        (is_valid: bool, message: str)
    """
    sequence = sequence.upper().strip()
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    invalid = set(sequence) - valid_aa
    if not sequence:
        return False, "Sequence is empty."
    if invalid:
        return False, f"Invalid characters found: {sorted(invalid)}"
    return True, "Valid sequence."


def list_available_models() -> dict:
    """Return a dict of human-readable model names → HuggingFace IDs."""
    return AVAILABLE_MODELS


# ─────────────────────────────────────────────
# Example Usage (run as script)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    TARGET = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"

    print("Loading model …")
    model, tokenizer = load_model("facebook/esm2_t6_8M_UR50D")
    device = get_device()

    valid, msg = validate_sequence(TARGET)
    if not valid:
        raise ValueError(msg)

    print("Generating peptides …")
    peptides = generate_peptides(
        model, tokenizer, TARGET,
        top_k=50, length=12, num_outputs=5, device=device
    )

    print("Scoring peptides …")
    scores = compute_scores(model, tokenizer, peptides, device=device)

    df = format_results(peptides, scores, protein_sequence=TARGET[:20] + "…")
    print("\n── Results ──────────────────────────────")
    print(df.to_string(index=False))
