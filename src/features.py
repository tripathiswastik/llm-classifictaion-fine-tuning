import math
import re
import zlib
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Precompiled regex patterns for speed
RE_WORDS = re.compile(r"\b\w+\b")
RE_CODE_BLOCKS = re.compile(r"```")
RE_BULLETS = re.compile(r"(?:^|\n)\s*[-*+]\s+")
RE_NUMBERED = re.compile(r"(?:^|\n)\s*\d+\.\s+")
RE_HEADERS = re.compile(r"(?:^|\n)\s*#{1,6}\s+")
RE_BOLD = re.compile(r"\*\*[^*]+\*\*")
RE_LATEX = re.compile(r"\$[^\$]+\$")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_APOLOGY = re.compile(r"\b(sorry|apologize|apologies|pardon)\b", re.IGNORECASE)
RE_AI_REF = re.compile(r"\b(as an ai|language model|ai assistant|artificial intelligence)\b", re.IGNORECASE)
RE_REFUSAL = re.compile(r"\b(cannot fulfill|unable to|i cannot|i can't|violates|not able to)\b", re.IGNORECASE)
RE_SENTENCE_SPLIT = re.compile(r"[.!?\n]+")
RE_VOWELS = re.compile(r"[aeiouy]+", re.IGNORECASE)
RE_NUM_CONSTRAINT = re.compile(
    r"\b(?:give me|list|write|provide)?\s*(\d{1,2})\s*(?:reasons|points|tips|ways|examples|items|steps|ideas)\b",
    re.IGNORECASE,
)
RE_TABLE = re.compile(r"(?:^|\n)\s*\|.*?\|.*?\|(?:\n\s*\|[-:\s|]+\|)", re.MULTILINE)
RE_SYNTAX_CODE = re.compile(r"```[a-zA-Z0-9_+#-]+")
RE_SYCOPHANCY = re.compile(
    r"\b(great question|good question|you are right|absolutely right|certainly|delighted to help|happy to assist|pleasure to help|as an expert)\b",
    re.IGNORECASE,
)
RE_CODE_PROMPT = re.compile(r"\b(code|python|java|c\+\+|javascript|function|algorithm|implement|script|sql|html|css|bug|error|exception)\b", re.IGNORECASE)
RE_MATH_PROMPT = re.compile(r"\b(calculate|math|equation|solve|integral|derivative|formula|probability|matrix|algebra|theorem)\b", re.IGNORECASE)
RE_COMPARE_PROMPT = re.compile(r"\b(compare|comparison|versus|vs|table|difference between|pros and cons)\b", re.IGNORECASE)
RE_FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself|we|us|our|ours)\b", re.IGNORECASE)
RE_SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself)\b", re.IGNORECASE)
RE_COMPLEX_PUNCT = re.compile(r"[;—\(\)\[\]\{\}]")
RE_CONNECTIVES = re.compile(
    r"\b(furthermore|moreover|therefore|however|nevertheless|consequently|specifically|in particular|for instance|for example|in summary|in conclusion)\b",
    re.IGNORECASE,
)
RE_STEP = re.compile(r"(?:^|\n)\s*(?:step\s*\d+|\d+\.)\s+", re.IGNORECASE)
RE_PARAGRAPHS = re.compile(r"\n\s*\n")
RE_NEG_PROMPT = re.compile(r"\b(do not|don't|never|avoid|without|no|must not)\s+(\w+)", re.IGNORECASE)

RE_UPPER = re.compile(r"[A-Z]")
RE_DIGIT = re.compile(r"[0-9]")

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
}


def compute_entropy(words: List[str]) -> float:
    """Computes Shannon entropy of word distribution."""
    if len(words) <= 1:
        return 0.0
    counts = Counter(words)
    total = len(words)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def compute_compression_ratio(text: str) -> float:
    """Computes zlib compression ratio (information density / Kolmogorov complexity)."""
    if not text:
        return 0.0
    raw_bytes = text.encode("utf-8", errors="replace")
    compressed = zlib.compress(raw_bytes, level=6)
    return len(compressed) / max(len(raw_bytes), 1)


def count_syllables(word: str) -> int:
    """Estimates syllables in an English word for readability indices."""
    w = word.lower().strip()
    if len(w) <= 3:
        return 1
    if w.endswith("e") and not w.endswith("le"):
        w = w[:-1]
    matches = RE_VOWELS.findall(w)
    return max(len(matches), 1)


def compute_text_stats(text: str) -> Dict[str, float]:
    """
    Computes structural, formatting, lexical, readability, and repetition metrics for a single text response.
    """
    char_len = len(text)
    if char_len == 0:
        return {
            "char_len": 0.0,
            "word_count": 0.0,
            "unique_words": 0.0,
            "avg_word_len": 0.0,
            "code_blocks": 0.0,
            "bullets": 0.0,
            "numbered_lists": 0.0,
            "headers": 0.0,
            "bold_spans": 0.0,
            "latex_math": 0.0,
            "urls": 0.0,
            "newlines": 0.0,
            "qmarks": 0.0,
            "excls": 0.0,
            "colons": 0.0,
            "uppercase_ratio": 0.0,
            "digit_count": 0.0,
            "apology_count": 0.0,
            "ai_ref_count": 0.0,
            "refusal_count": 0.0,
            "trigram_rep_rate": 0.0,
            "type_token_ratio": 1.0,
            "avg_sent_len": 0.0,
            "flesch_reading_ease": 0.0,
            "code_density": 0.0,
            "has_table": 0.0,
            "syntax_code_blocks": 0.0,
            "sycophancy": 0.0,
            "first_person": 0.0,
            "second_person": 0.0,
            "complex_punct": 0.0,
        }

    words = RE_WORDS.findall(text)
    word_count = len(words)
    words_lower = [w.lower() for w in words]
    unique_words = len(set(words_lower))
    total_word_len = sum(len(w) for w in words)
    avg_word_len = total_word_len / max(word_count, 1)

    code_blocks = text.count("```") // 2
    bullets = len(RE_BULLETS.findall(text))
    numbered = len(RE_NUMBERED.findall(text))
    headers = len(RE_HEADERS.findall(text))
    bold = text.count("**") // 2
    latex = text.count("$") // 2
    urls = len(RE_URL.findall(text))

    newlines = text.count("\n")
    qmarks = text.count("?")
    excls = text.count("!")
    colons = text.count(":")
    digits = len(RE_DIGIT.findall(text))
    uppers = len(RE_UPPER.findall(text))
    uppercase_ratio = uppers / char_len

    apology = len(RE_APOLOGY.findall(text))
    ai_ref = len(RE_AI_REF.findall(text))
    refusal = len(RE_REFUSAL.findall(text))

    # Repetition and Diversity
    ttr = unique_words / max(word_count, 1)
    sentences = max(len(RE_SENTENCE_SPLIT.findall(text)), 1)
    avg_sent_len = word_count / sentences

    if word_count < 5:
        rep_rate = 0.0
    else:
        trigrams = [tuple(words_lower[i : i + 3]) for i in range(word_count - 2)]
        rep_rate = 1.0 - (len(set(trigrams)) / max(len(trigrams), 1))

    # Readability: Flesch Reading Ease
    if word_count == 0:
        fre = 0.0
    else:
        syl = sum(count_syllables(w) for w in words) / max(word_count, 1)
        fre = 206.835 - 1.015 * avg_sent_len - 84.6 * syl

    # Code density
    code_chars = sum(len(c) for c in re.findall(r"```[\s\S]*?```", text))
    code_density = code_chars / max(char_len, 1)

    # Tables, syntax code fences, sycophancy
    has_table = float(bool(RE_TABLE.search(text)))
    syntax_code_blocks = float(len(RE_SYNTAX_CODE.findall(text)))
    sycophancy = float(len(RE_SYCOPHANCY.findall(text)))

    # Pronouns & punctuation
    first_person = float(len(RE_FIRST_PERSON.findall(text)))
    second_person = float(len(RE_SECOND_PERSON.findall(text)))
    complex_punct = float(len(RE_COMPLEX_PUNCT.findall(text)))

    return {
        "char_len": float(char_len),
        "word_count": float(word_count),
        "unique_words": float(unique_words),
        "avg_word_len": float(avg_word_len),
        "code_blocks": float(code_blocks),
        "bullets": float(bullets),
        "numbered_lists": float(numbered),
        "headers": float(headers),
        "bold_spans": float(bold),
        "latex_math": float(latex),
        "urls": float(urls),
        "newlines": float(newlines),
        "qmarks": float(qmarks),
        "excls": float(excls),
        "colons": float(colons),
        "uppercase_ratio": float(uppercase_ratio),
        "digit_count": float(digits),
        "apology_count": float(apology),
        "ai_ref_count": float(ai_ref),
        "refusal_count": float(refusal),
        "trigram_rep_rate": float(rep_rate),
        "type_token_ratio": float(ttr),
        "avg_sent_len": float(avg_sent_len),
        "flesch_reading_ease": float(fre),
        "code_density": float(code_density),
        "has_table": has_table,
        "syntax_code_blocks": syntax_code_blocks,
        "sycophancy": sycophancy,
        "first_person": first_person,
        "second_person": second_person,
        "complex_punct": complex_punct,
    }


def compute_pairwise_features(
    prompt: str,
    resp_a: str,
    resp_b: str,
    turns_a: int = 1,
    turns_b: int = 1,
    prompt_turns: Optional[List[str]] = None,
    resp_a_turns: Optional[List[str]] = None,
    resp_b_turns: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Computes comparative difference and interaction features between response_a and response_b.
    """
    stats_a = compute_text_stats(resp_a)
    stats_b = compute_text_stats(resp_b)

    p_words = [w.lower() for w in RE_WORDS.findall(prompt)]
    a_words = [w.lower() for w in RE_WORDS.findall(resp_a)]
    b_words = [w.lower() for w in RE_WORDS.findall(resp_b)]

    p_set = set(p_words)
    a_set = set(a_words)
    b_set = set(b_words)

    # Prompt overlap
    inter_pa = len(p_set & a_set)
    union_pa = max(len(p_set | a_set), 1)
    jaccard_pa = inter_pa / union_pa

    inter_pb = len(p_set & b_set)
    union_pb = max(len(p_set | b_set), 1)
    jaccard_pb = inter_pb / union_pb

    cov_pa = inter_pa / max(len(p_set), 1)
    cov_pb = inter_pb / max(len(p_set), 1)

    # Content word overlap (ignoring stopwords)
    content_p = set(w for w in p_words if w not in STOPWORDS)
    content_a = set(w for w in a_words if w not in STOPWORDS)
    content_b = set(w for w in b_words if w not in STOPWORDS)

    if content_p:
        c_cov_pa = len(content_p & content_a) / len(content_p)
        c_cov_pb = len(content_p & content_b) / len(content_p)
        c_union_pa = max(len(content_p | content_a), 1)
        c_union_pb = max(len(content_p | content_b), 1)
        c_jacc_pa = len(content_p & content_a) / c_union_pa
        c_jacc_pb = len(content_p & content_b) / c_union_pb
    else:
        c_cov_pa = c_cov_pb = c_jacc_pa = c_jacc_pb = 0.0

    # Inter-response overlap (Crucial for tie detection)
    inter_ab = len(a_set & b_set)
    union_ab = max(len(a_set | b_set), 1)
    jaccard_ab = inter_ab / union_ab
    char_len_ab_diff = abs(stats_a["char_len"] - stats_b["char_len"])
    char_len_ab_min = min(stats_a["char_len"], stats_b["char_len"])
    char_len_ab_max = max(stats_a["char_len"], stats_b["char_len"])
    char_len_ratio = (char_len_ab_min + 1.0) / (char_len_ab_max + 1.0)

    # Differences (A - B)
    char_diff = stats_a["char_len"] - stats_b["char_len"]
    word_diff = stats_a["word_count"] - stats_b["word_count"]
    log_char_diff = math.log(stats_a["char_len"] + 1.0) - math.log(stats_b["char_len"] + 1.0)
    log_word_diff = math.log(stats_a["word_count"] + 1.0) - math.log(stats_b["word_count"] + 1.0)

    code_diff = stats_a["code_blocks"] - stats_b["code_blocks"]
    bullet_diff = stats_a["bullets"] - stats_b["bullets"]
    num_diff = stats_a["numbered_lists"] - stats_b["numbered_lists"]
    header_diff = stats_a["headers"] - stats_b["headers"]
    bold_diff = stats_a["bold_spans"] - stats_b["bold_spans"]
    newline_diff = stats_a["newlines"] - stats_b["newlines"]
    latex_diff = stats_a["latex_math"] - stats_b["latex_math"]
    url_diff = stats_a["urls"] - stats_b["urls"]
    qmark_diff = stats_a["qmarks"] - stats_b["qmarks"]
    excl_diff = stats_a["excls"] - stats_b["excls"]
    colon_diff = stats_a["colons"] - stats_b["colons"]

    apology_diff = stats_a["apology_count"] - stats_b["apology_count"]
    ai_ref_diff = stats_a["ai_ref_count"] - stats_b["ai_ref_count"]
    refusal_diff = stats_a["refusal_count"] - stats_b["refusal_count"]

    turns_diff = float(turns_a - turns_b)

    # Numerical constraint adherence (e.g. prompt asks for N items)
    match_constraint = RE_NUM_CONSTRAINT.search(prompt)
    if match_constraint:
        target_n = int(match_constraint.group(1))
        bullets_a = len(RE_BULLETS.findall(resp_a)) + len(RE_NUMBERED.findall(resp_a))
        bullets_b = len(RE_BULLETS.findall(resp_b)) + len(RE_NUMBERED.findall(resp_b))
        err_a = abs(bullets_a - target_n)
        err_b = abs(bullets_b - target_n)
        diff_constraint_err = float(err_b - err_a)
    else:
        diff_constraint_err = 0.0

    # Prompt Intent Classification
    is_code_p = float(bool(RE_CODE_PROMPT.search(prompt)))
    is_math_p = float(bool(RE_MATH_PROMPT.search(prompt)))
    is_compare_p = float(bool(RE_COMPARE_PROMPT.search(prompt)))

    # Prompt conditioned interactions
    diff_table = stats_a["has_table"] - stats_b["has_table"]
    diff_prompt_code_inter = is_code_p * code_diff
    diff_prompt_math_inter = is_math_p * latex_diff
    diff_prompt_table_inter = is_compare_p * diff_table

    # Shannon Lexical Entropy
    ent_a = compute_entropy(a_words)
    ent_b = compute_entropy(b_words)
    diff_entropy = ent_a - ent_b
    sym_mean_entropy = 0.5 * (ent_a + ent_b)

    # Zlib Compression Ratio (information density & redundancy)
    cr_a = compute_compression_ratio(resp_a)
    cr_b = compute_compression_ratio(resp_b)
    diff_cr = cr_a - cr_b
    sym_mean_cr = 0.5 * (cr_a + cr_b)

    # Content Word Bigram Coverage
    p_content_list = [w for w in p_words if w not in STOPWORDS]
    a_content_list = [w for w in a_words if w not in STOPWORDS]
    b_content_list = [w for w in b_words if w not in STOPWORDS]
    p_bigrams = set(zip(p_content_list[:-1], p_content_list[1:])) if len(p_content_list) >= 2 else set()
    a_bigrams = set(zip(a_content_list[:-1], a_content_list[1:])) if len(a_content_list) >= 2 else set()
    b_bigrams = set(zip(b_content_list[:-1], b_content_list[1:])) if len(b_content_list) >= 2 else set()
    if p_bigrams:
        cov_bigram_a = len(p_bigrams & a_bigrams) / len(p_bigrams)
        cov_bigram_b = len(p_bigrams & b_bigrams) / len(p_bigrams)
    else:
        cov_bigram_a = cov_bigram_b = 0.0
    diff_bigram_cov = cov_bigram_a - cov_bigram_b
    sym_mean_bigram_cov = 0.5 * (cov_bigram_a + cov_bigram_b)

    # Discourse Connectives & Cohesion Markers
    conn_a = float(len(RE_CONNECTIVES.findall(resp_a)))
    conn_b = float(len(RE_CONNECTIVES.findall(resp_b)))
    diff_connectives = conn_a - conn_b
    sym_connectives = 0.5 * (conn_a + conn_b)

    # Structural Paragraphs & Step Progression
    para_a = float(len(RE_PARAGRAPHS.findall(resp_a)))
    para_b = float(len(RE_PARAGRAPHS.findall(resp_b)))
    diff_para = para_a - para_b
    sym_para = para_a + para_b

    steps_a = float(len(RE_STEP.findall(resp_a)))
    steps_b = float(len(RE_STEP.findall(resp_b)))
    diff_steps = steps_a - steps_b
    sym_steps = steps_a + steps_b

    # Negative Prompt Constraint Adherence
    neg_matches = RE_NEG_PROMPT.findall(prompt)
    diff_neg_violation = 0.0
    if neg_matches:
        for _, forbidden_word in neg_matches:
            w_lower = forbidden_word.lower()
            if len(w_lower) >= 3:
                in_a = 1.0 if w_lower in a_set else 0.0
                in_b = 1.0 if w_lower in b_set else 0.0
                diff_neg_violation += (in_b - in_a)

    # Multi-turn final-turn signals
    p_tl = prompt_turns if (prompt_turns is not None and len(prompt_turns) > 0) else [prompt]
    a_tl = resp_a_turns if (resp_a_turns is not None and len(resp_a_turns) > 0) else [resp_a]
    b_tl = resp_b_turns if (resp_b_turns is not None and len(resp_b_turns) > 0) else [resp_b]

    is_multi = 1.0 if len(p_tl) > 1 else 0.0
    n_turns = float(len(p_tl))

    final_p = p_tl[-1] if p_tl else ""
    final_a = a_tl[-1] if a_tl else ""
    final_b = b_tl[-1] if b_tl else ""

    len_fa = float(len(final_a))
    len_fb = float(len(final_b))
    diff_final_len = len_fa - len_fb
    diff_final_log_len = float(math.log(len_fa + 1.0) - math.log(len_fb + 1.0))
    sym_final_len = len_fa + len_fb

    words_fp = set(w.lower() for w in RE_WORDS.findall(final_p))
    words_fa = set(w.lower() for w in RE_WORDS.findall(final_a))
    words_fb = set(w.lower() for w in RE_WORDS.findall(final_b))
    jacc_fa = len(words_fp & words_fa) / max(len(words_fp | words_fa), 1)
    jacc_fb = len(words_fp & words_fb) / max(len(words_fp | words_fb), 1)
    diff_final_jacc = float(jacc_fa - jacc_fb)
    sym_final_jacc = 0.5 * float(jacc_fa + jacc_fb)

    # Non-linear length ratio cliffs
    len_a = float(stats_a["char_len"])
    len_b = float(stats_b["char_len"])
    ratio_15 = float(len_a > 1.5 * len_b) - float(len_b > 1.5 * len_a)
    ratio_20 = float(len_a > 2.0 * len_b) - float(len_b > 2.0 * len_a)
    ratio_30 = float(len_a > 3.0 * len_b) - float(len_b > 3.0 * len_a)
    sym_extreme_len_imbalance = float((len_a > 2.0 * len_b) or (len_b > 2.0 * len_a))

    # Exact match indicator
    is_exact = float(resp_a.strip() == resp_b.strip())

    features: Dict[str, float] = {
        # Anti-symmetric difference features
        "diff_char_len": char_diff,
        "diff_word_count": word_diff,
        "diff_log_char": log_char_diff,
        "diff_log_word": log_word_diff,
        "diff_avg_word_len": stats_a["avg_word_len"] - stats_b["avg_word_len"],
        "diff_code_blocks": code_diff,
        "diff_bullets": bullet_diff,
        "diff_numbered": num_diff,
        "diff_headers": header_diff,
        "diff_bold": bold_diff,
        "diff_latex": latex_diff,
        "diff_urls": url_diff,
        "diff_newlines": newline_diff,
        "diff_qmarks": qmark_diff,
        "diff_excls": excl_diff,
        "diff_colons": colon_diff,
        "diff_uppercase_ratio": stats_a["uppercase_ratio"] - stats_b["uppercase_ratio"],
        "diff_apology": apology_diff,
        "diff_ai_ref": ai_ref_diff,
        "diff_refusal": refusal_diff,
        "diff_jaccard_p": jaccard_pa - jaccard_pb,
        "diff_cov_p": cov_pa - cov_pb,
        "diff_turns": turns_diff,
        "diff_trigram_rep_rate": stats_a["trigram_rep_rate"] - stats_b["trigram_rep_rate"],
        "diff_type_token_ratio": stats_a["type_token_ratio"] - stats_b["type_token_ratio"],
        "diff_avg_sent_len": stats_a["avg_sent_len"] - stats_b["avg_sent_len"],
        "diff_flesch_reading_ease": stats_a["flesch_reading_ease"] - stats_b["flesch_reading_ease"],
        "diff_code_density": stats_a["code_density"] - stats_b["code_density"],
        "diff_constraint_adherence": diff_constraint_err,
        "diff_markdown_table": diff_table,
        "diff_syntax_code_blocks": stats_a["syntax_code_blocks"] - stats_b["syntax_code_blocks"],
        "diff_sycophancy": stats_a["sycophancy"] - stats_b["sycophancy"],
        "diff_prompt_code_inter": diff_prompt_code_inter,
        "diff_prompt_math_inter": diff_prompt_math_inter,
        "diff_prompt_table_inter": diff_prompt_table_inter,
        "diff_first_person": stats_a["first_person"] - stats_b["first_person"],
        "diff_second_person": stats_a["second_person"] - stats_b["second_person"],
        "diff_complex_punct": stats_a["complex_punct"] - stats_b["complex_punct"],
        "diff_content_cov": float(c_cov_pa - c_cov_pb),
        "diff_content_jacc": float(c_jacc_pa - c_jacc_pb),
        "diff_entropy": diff_entropy,
        "diff_compression_ratio": diff_cr,
        "diff_bigram_cov": diff_bigram_cov,
        "diff_connectives": diff_connectives,
        "diff_para": diff_para,
        "diff_steps": diff_steps,
        "diff_neg_violation": diff_neg_violation,
        "diff_final_len": diff_final_len,
        "diff_final_log_len": diff_final_log_len,
        "diff_final_jacc": diff_final_jacc,
        "diff_len_ratio_15": ratio_15,
        "diff_len_ratio_20": ratio_20,
        "diff_len_ratio_30": ratio_30,
        # Symmetric tie & similarity features
        "sym_jaccard_ab": jaccard_ab,
        "sym_abs_char_diff": char_len_ab_diff,
        "sym_char_ratio": char_len_ratio,
        "sym_total_char_len": stats_a["char_len"] + stats_b["char_len"],
        "sym_prompt_char_len": float(len(prompt)),
        "sym_prompt_word_len": float(len(p_set)),
        "sym_mean_jaccard_p": 0.5 * (jaccard_pa + jaccard_pb),
        "sym_code_presence": float((stats_a["code_blocks"] > 0) or (stats_b["code_blocks"] > 0)),
        "sym_latex_presence": float((stats_a["latex_math"] > 0) or (stats_b["latex_math"] > 0)),
        "sym_mean_ttr": 0.5 * (stats_a["type_token_ratio"] + stats_b["type_token_ratio"]),
        "sym_mean_flesch_reading_ease": 0.5 * (stats_a["flesch_reading_ease"] + stats_b["flesch_reading_ease"]),
        "sym_mean_code_density": 0.5 * (stats_a["code_density"] + stats_b["code_density"]),
        "sym_markdown_table": 0.5 * (stats_a["has_table"] + stats_b["has_table"]),
        "sym_syntax_code_blocks": 0.5 * (stats_a["syntax_code_blocks"] + stats_b["syntax_code_blocks"]),
        "sym_sycophancy": 0.5 * (stats_a["sycophancy"] + stats_b["sycophancy"]),
        "sym_is_code_prompt": is_code_p,
        "sym_is_math_prompt": is_math_p,
        "sym_first_person": 0.5 * (stats_a["first_person"] + stats_b["first_person"]),
        "sym_second_person": 0.5 * (stats_a["second_person"] + stats_b["second_person"]),
        "sym_complex_punct": 0.5 * (stats_a["complex_punct"] + stats_b["complex_punct"]),
        "sym_mean_content_cov": 0.5 * float(c_cov_pa + c_cov_pb),
        "sym_mean_entropy": sym_mean_entropy,
        "sym_mean_compression_ratio": sym_mean_cr,
        "sym_mean_bigram_cov": sym_mean_bigram_cov,
        "sym_connectives": sym_connectives,
        "sym_para": sym_para,
        "sym_steps": sym_steps,
        "sym_final_len": sym_final_len,
        "sym_final_jacc": sym_final_jacc,
        "sym_is_multi_turn": is_multi,
        "sym_num_turns": n_turns,
        "sym_extreme_len_imbalance": sym_extreme_len_imbalance,
        "sym_is_exact_match": is_exact,
    }
    return features


class FeatureExtractor:
    """
    Transforms DataFrame of battles into a clean NumPy feature matrix.
    Supports hand-crafted pairwise features, sparse TF-IDF prompt-response alignment,
    and dense Latent Semantic Analysis (TruncatedSVD) dimensions.
    """

    def __init__(
        self,
        use_tfidf: bool = True,
        max_tfidf_features: int = 15000,
        use_lsa: bool = True,
        n_lsa_components: int = 24,
        use_char_lsa: bool = True,
        n_char_lsa_components: int = 16,
    ):
        self.use_tfidf = use_tfidf
        self.max_tfidf_features = max_tfidf_features
        self.use_lsa = use_lsa
        self.n_lsa_components = n_lsa_components
        self.use_char_lsa = use_char_lsa
        self.n_char_lsa_components = n_char_lsa_components
        self.base_feature_names_: List[str] = []
        self.feature_names_: List[str] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.lsa_feature_names_: List[str] = []
        self.char_vectorizer: Optional[TfidfVectorizer] = None
        self.char_svd: Optional[TruncatedSVD] = None
        self.char_lsa_feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":
        sample_feat = compute_pairwise_features(
            prompt=df["prompt_text"].iloc[0],
            resp_a=df["response_a_text"].iloc[0],
            resp_b=df["response_b_text"].iloc[0],
        )
        self.base_feature_names_ = sorted(list(sample_feat.keys()))

        if self.use_tfidf:
            corpus = (
                list(df["prompt_text"].values)
                + list(df["response_a_text"].values)
                + list(df["response_b_text"].values)
            )
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_tfidf_features,
                sublinear_tf=True,
                stop_words="english",
                ngram_range=(1, 2),
            )
            tfidf_mat = self.tfidf_vectorizer.fit_transform(corpus)
            tfidf_feature_names = [
                "diff_tfidf_prompt_sim",
                "sym_tfidf_ab_sim",
                "sym_tfidf_mean_prompt",
            ]
            self.feature_names_ = self.base_feature_names_ + tfidf_feature_names

            if self.use_lsa:
                n_features = tfidf_mat.shape[1]
                n_samples = tfidf_mat.shape[0]
                actual_k = min(self.n_lsa_components, max(1, min(n_samples, n_features) - 1))
                self.svd = TruncatedSVD(n_components=actual_k, random_state=42)
                self.svd.fit(tfidf_mat)
                self.lsa_feature_names_ = [
                    "diff_lsa_prompt_cos",
                    "sym_lsa_ab_cos",
                    "sym_lsa_mean_prompt_cos",
                    "diff_lsa_norm",
                    "sym_lsa_norm",
                    "sym_lsa_dist_ab",
                    "diff_lsa_dist_p",
                ]
                for i in range(self.n_lsa_components):
                    self.lsa_feature_names_.append(f"diff_lsa_dim_{i}")
                    self.lsa_feature_names_.append(f"sym_lsa_dim_{i}")
                self.feature_names_ = self.feature_names_ + self.lsa_feature_names_

            if self.use_char_lsa:
                self.char_vectorizer = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=12000,
                    sublinear_tf=True,
                )
                char_mat = self.char_vectorizer.fit_transform(corpus)
                n_char_feats = char_mat.shape[1]
                n_char_samps = char_mat.shape[0]
                actual_k_char = min(self.n_char_lsa_components, max(1, min(n_char_samps, n_char_feats) - 1))
                self.char_svd = TruncatedSVD(n_components=actual_k_char, random_state=42)
                self.char_svd.fit(char_mat)
                self.char_lsa_feature_names_ = [
                    "diff_char_lsa_prompt_cos",
                    "sym_char_lsa_ab_cos",
                    "sym_char_lsa_mean_prompt_cos",
                    "diff_char_lsa_norm",
                    "sym_char_lsa_norm",
                    "sym_char_lsa_dist_ab",
                ]
                for i in range(self.n_char_lsa_components):
                    self.char_lsa_feature_names_.append(f"diff_char_lsa_dim_{i}")
                    self.char_lsa_feature_names_.append(f"sym_char_lsa_dim_{i}")
                self.feature_names_ = self.feature_names_ + self.char_lsa_feature_names_
        else:
            self.feature_names_ = list(self.base_feature_names_)

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.feature_names_:
            raise ValueError("FeatureExtractor must be fitted before transform")

        rows = []
        p_texts = df["prompt_text"].values
        a_texts = df["response_a_text"].values
        b_texts = df["response_b_text"].values
        p_turns = df["prompt_turns"].values if "prompt_turns" in df.columns else None
        a_turns = df["response_a_turns"].values if "response_a_turns" in df.columns else None
        b_turns = df["response_b_turns"].values if "response_b_turns" in df.columns else None

        for i, (p, a, b) in enumerate(zip(p_texts, a_texts, b_texts)):
            ptl = p_turns[i] if p_turns is not None else None
            atl = a_turns[i] if a_turns is not None else None
            btl = b_turns[i] if b_turns is not None else None
            ta = len(atl) if atl is not None else 1
            tb = len(btl) if btl is not None else 1
            feat_dict = compute_pairwise_features(p, a, b, ta, tb, ptl, atl, btl)
            rows.append([feat_dict[k] for k in self.base_feature_names_])

        X_base = np.asarray(rows, dtype=np.float32)

        if self.use_tfidf:
            if self.tfidf_vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted")
            p_vecs = self.tfidf_vectorizer.transform(df["prompt_text"])
            a_vecs = self.tfidf_vectorizer.transform(df["response_a_text"])
            b_vecs = self.tfidf_vectorizer.transform(df["response_b_text"])

            sim_pa = np.asarray(p_vecs.multiply(a_vecs).sum(axis=1)).flatten()
            sim_pb = np.asarray(p_vecs.multiply(b_vecs).sum(axis=1)).flatten()
            sim_ab = np.asarray(a_vecs.multiply(b_vecs).sum(axis=1)).flatten()

            diff_prompt_sim = sim_pa - sim_pb
            sym_ab_sim = sim_ab
            mean_prompt_sim = 0.5 * (sim_pa + sim_pb)

            X_tfidf = np.column_stack([diff_prompt_sim, sym_ab_sim, mean_prompt_sim]).astype(np.float32)

            if self.use_lsa and self.svd is not None:
                p_svd = self.svd.transform(p_vecs)
                a_svd = self.svd.transform(a_vecs)
                b_svd = self.svd.transform(b_vecs)

                k = self.n_lsa_components
                if a_svd.shape[1] < k:
                    pad = k - a_svd.shape[1]
                    p_svd = np.pad(p_svd, ((0, 0), (0, pad)), mode="constant")
                    a_svd = np.pad(a_svd, ((0, 0), (0, pad)), mode="constant")
                    b_svd = np.pad(b_svd, ((0, 0), (0, pad)), mode="constant")

                def l2_norm(mat: np.ndarray) -> np.ndarray:
                    norms = np.linalg.norm(mat, axis=1, keepdims=True)
                    return mat / np.maximum(norms, 1e-9)

                p_norm = l2_norm(p_svd)
                a_norm = l2_norm(a_svd)
                b_norm = l2_norm(b_svd)

                cos_pa = np.sum(p_norm * a_norm, axis=1, keepdims=True)
                cos_pb = np.sum(p_norm * b_norm, axis=1, keepdims=True)
                cos_ab = np.sum(a_norm * b_norm, axis=1, keepdims=True)

                diff_cos_p = (cos_pa - cos_pb).astype(np.float32)
                sym_cos_ab = cos_ab.astype(np.float32)
                sym_mean_cos_p = (0.5 * (cos_pa + cos_pb)).astype(np.float32)

                # LSA Norms and Distances
                norm_a = np.linalg.norm(a_svd, axis=1, keepdims=True)
                norm_b = np.linalg.norm(b_svd, axis=1, keepdims=True)
                diff_lsa_norm = (norm_a - norm_b).astype(np.float32)
                sym_lsa_norm = (0.5 * (norm_a + norm_b)).astype(np.float32)

                dist_ab = np.linalg.norm(a_svd - b_svd, axis=1, keepdims=True).astype(np.float32)
                dist_pa = np.linalg.norm(p_svd - a_svd, axis=1, keepdims=True)
                dist_pb = np.linalg.norm(p_svd - b_svd, axis=1, keepdims=True)
                diff_dist_p = (dist_pb - dist_pa).astype(np.float32)

                diff_svd = (a_svd - b_svd).astype(np.float32)
                sym_svd = (0.5 * (a_svd + b_svd)).astype(np.float32)

                dim_cols = []
                for i in range(self.n_lsa_components):
                    dim_cols.append(diff_svd[:, i : i + 1])
                    dim_cols.append(sym_svd[:, i : i + 1])

                X_lsa = np.hstack([diff_cos_p, sym_cos_ab, sym_mean_cos_p, diff_lsa_norm, sym_lsa_norm, dist_ab, diff_dist_p] + dim_cols)
                parts = [X_base, X_tfidf, X_lsa]
            else:
                parts = [X_base, X_tfidf]

            if self.use_char_lsa and self.char_vectorizer is not None and self.char_svd is not None:
                p_char = self.char_vectorizer.transform(df["prompt_text"])
                a_char = self.char_vectorizer.transform(df["response_a_text"])
                b_char = self.char_vectorizer.transform(df["response_b_text"])

                p_csvd = self.char_svd.transform(p_char)
                a_csvd = self.char_svd.transform(a_char)
                b_csvd = self.char_svd.transform(b_char)

                k_c = self.n_char_lsa_components
                if a_csvd.shape[1] < k_c:
                    pad_c = k_c - a_csvd.shape[1]
                    p_csvd = np.pad(p_csvd, ((0, 0), (0, pad_c)), mode="constant")
                    a_csvd = np.pad(a_csvd, ((0, 0), (0, pad_c)), mode="constant")
                    b_csvd = np.pad(b_csvd, ((0, 0), (0, pad_c)), mode="constant")

                def l2_norm_c(mat: np.ndarray) -> np.ndarray:
                    norms = np.linalg.norm(mat, axis=1, keepdims=True)
                    return mat / np.maximum(norms, 1e-9)

                p_cnorm = l2_norm_c(p_csvd)
                a_cnorm = l2_norm_c(a_csvd)
                b_cnorm = l2_norm_c(b_csvd)

                cos_cpa = np.sum(p_cnorm * a_cnorm, axis=1, keepdims=True)
                cos_cpb = np.sum(p_cnorm * b_cnorm, axis=1, keepdims=True)
                cos_cab = np.sum(a_cnorm * b_cnorm, axis=1, keepdims=True)

                diff_cos_cp = (cos_cpa - cos_cpb).astype(np.float32)
                sym_cos_cab = cos_cab.astype(np.float32)
                sym_mean_cos_cp = (0.5 * (cos_cpa + cos_cpb)).astype(np.float32)

                # Char LSA Norms and Distances
                cnorm_a = np.linalg.norm(a_csvd, axis=1, keepdims=True)
                cnorm_b = np.linalg.norm(b_csvd, axis=1, keepdims=True)
                diff_char_norm = (cnorm_a - cnorm_b).astype(np.float32)
                sym_char_norm = (0.5 * (cnorm_a + cnorm_b)).astype(np.float32)
                cdist_ab = np.linalg.norm(a_csvd - b_csvd, axis=1, keepdims=True).astype(np.float32)

                diff_csvd = (a_csvd - b_csvd).astype(np.float32)
                sym_csvd = (0.5 * (a_csvd + b_csvd)).astype(np.float32)

                dim_char_cols = []
                for i in range(self.n_char_lsa_components):
                    dim_char_cols.append(diff_csvd[:, i : i + 1])
                    dim_char_cols.append(sym_csvd[:, i : i + 1])

                X_char_lsa = np.hstack([diff_cos_cp, sym_cos_cab, sym_mean_cos_cp, diff_char_norm, sym_char_norm, cdist_ab] + dim_char_cols)
                parts.append(X_char_lsa)

            return np.hstack(parts)

        return X_base

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
