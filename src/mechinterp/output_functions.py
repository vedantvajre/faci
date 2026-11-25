"""
Task-Faithful Output Functions for Tier 3

This module provides task-faithful output functions that measure
actual task performance, not generic confidence proxies.
"""

import torch
import torch.nn.functional as F
from functools import partial
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
import random


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Tier3ExamplePair:
    """A pair of examples for Tier 3 with all needed info."""
    positive_input: str
    negative_input: str
    correct_tokens: List[int]
    incorrect_tokens: List[int]


# =============================================================================
# SENTIMENT OUTPUT FUNCTIONS
# =============================================================================

def sentiment_encoder_fn(outputs, inputs, model, tokenizer, **kwargs):
    """Encoder with classification head."""
    logits = outputs.logits
    if hasattr(model.config, 'label2id'):
        label2id = model.config.label2id
        # Handle various naming conventions
        pos_idx = label2id.get('POSITIVE', label2id.get('positive', 
                  label2id.get('LABEL_1', label2id.get('1', 1))))
        neg_idx = label2id.get('NEGATIVE', label2id.get('negative',
                  label2id.get('LABEL_0', label2id.get('0', 0))))
    else:
        pos_idx, neg_idx = 1, 0
    return (logits[:, pos_idx] - logits[:, neg_idx]).mean()


def sentiment_decoder_fn(outputs, inputs, model, tokenizer, **kwargs):
    """Decoder LM - score positive vs negative tokens."""
    logits = outputs.logits
    
    # Get token IDs with space prefix for GPT-2
    pos_tokens = tokenizer.encode(' positive', add_special_tokens=False)
    neg_tokens = tokenizer.encode(' negative', add_special_tokens=False)
    pos_token = pos_tokens[0] if pos_tokens else tokenizer.encode('positive', add_special_tokens=False)[0]
    neg_token = neg_tokens[0] if neg_tokens else tokenizer.encode('negative', add_special_tokens=False)[0]
    
    last_logits = logits[:, -1, :]
    log_probs = F.log_softmax(last_logits, dim=-1)
    return (log_probs[:, pos_token] - log_probs[:, neg_token]).mean()


def sentiment_encdec_fn(outputs, inputs, model, tokenizer, **kwargs):
    """Encoder-decoder - score at first decoder position."""
    logits = outputs.logits
    
    pos_tokens = tokenizer.encode('positive', add_special_tokens=False)
    neg_tokens = tokenizer.encode('negative', add_special_tokens=False)
    pos_token = pos_tokens[0]
    neg_token = neg_tokens[0]
    
    first_logits = logits[:, 0, :]
    log_probs = F.log_softmax(first_logits, dim=-1)
    return (log_probs[:, pos_token] - log_probs[:, neg_token]).mean()


# =============================================================================
# NLI OUTPUT FUNCTIONS
# =============================================================================

def nli_encoder_fn(outputs, inputs, model, tokenizer, **kwargs):
    """Encoder with classification head - use config for label indices."""
    logits = outputs.logits
    # Empirically verified label indices:
    NLI_INDEX_OVERRIDES = {
        "textattack/bert-base-uncased-MNLI": (1, 0),
        "textattack/roberta-base-MNLI": (2, 0),
    }
    _name = getattr(getattr(model, "config", None), "_name_or_path", "")
    if _name in NLI_INDEX_OVERRIDES:
        ent_idx, con_idx = NLI_INDEX_OVERRIDES[_name]
        return (logits[:, ent_idx] - logits[:, con_idx]).mean()
    
    if hasattr(model.config, 'label2id'):
        label2id = model.config.label2id
        # Handle different naming conventions
        ent_idx = None
        con_idx = None
        
        for key, val in label2id.items():
            key_lower = key.lower()
            if 'entail' in key_lower:
                ent_idx = val
            elif 'contradict' in key_lower:
                con_idx = val
        
        if ent_idx is None:
            ent_idx = label2id.get('LABEL_0', 0)
        if con_idx is None:
            con_idx = label2id.get('LABEL_2', 2)
    else:
        ent_idx, con_idx = 0, 2
    
    return (logits[:, ent_idx] - logits[:, con_idx]).mean()


def nli_decoder_fn(outputs, inputs, model, tokenizer, **kwargs):
    """Decoder LM - use actual label tokens (entailment/contradiction, not yes/no)."""
    logits = outputs.logits
    
    # Try entailment/contradiction first, fall back to yes/no
    try:
        ent_tokens = tokenizer.encode(' entailment', add_special_tokens=False)
        con_tokens = tokenizer.encode(' contradiction', add_special_tokens=False)
        if not ent_tokens or not con_tokens:
            raise ValueError("Empty tokens")
        ent_token = ent_tokens[0]
        con_token = con_tokens[0]
    except:
        # Fallback to true/false
        ent_token = tokenizer.encode(' true', add_special_tokens=False)[0]
        con_token = tokenizer.encode(' false', add_special_tokens=False)[0]
    
    last_logits = logits[:, -1, :]
    log_probs = F.log_softmax(last_logits, dim=-1)
    return (log_probs[:, ent_token] - log_probs[:, con_token]).mean()


def nli_encdec_fn(outputs, inputs, model, tokenizer, **kwargs):
    """Encoder-decoder NLI."""
    logits = outputs.logits
    
    try:
        ent_token = tokenizer.encode('entailment', add_special_tokens=False)[0]
        con_token = tokenizer.encode('contradiction', add_special_tokens=False)[0]
    except:
        ent_token = tokenizer.encode('true', add_special_tokens=False)[0]
        con_token = tokenizer.encode('false', add_special_tokens=False)[0]
    
    first_logits = logits[:, 0, :]
    log_probs = F.log_softmax(first_logits, dim=-1)
    return (log_probs[:, ent_token] - log_probs[:, con_token]).mean()


# =============================================================================
# FACTUAL RECALL OUTPUT FUNCTIONS
# =============================================================================

def factual_encoder_mlm_fn(outputs, inputs, model, tokenizer, 
                           correct_tokens: List[int], 
                           incorrect_tokens: List[int], **kwargs):
    """Encoder MLM - score at [MASK] position."""
    logits = outputs.logits
    input_ids = inputs['input_ids']
    
    # Find mask position
    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
    if len(mask_positions[0]) == 0:
        return torch.tensor(0.0, device=logits.device)
    
    batch_idx = mask_positions[0][0]
    mask_pos = mask_positions[1][0]
    mask_logits = logits[batch_idx, mask_pos, :]
    
    log_probs = F.log_softmax(mask_logits, dim=-1)
    
    # Handle multi-token by averaging
    correct_lp = log_probs[correct_tokens].mean() if correct_tokens else torch.tensor(0.0)
    incorrect_lp = log_probs[incorrect_tokens].mean() if incorrect_tokens else torch.tensor(0.0)
    
    return correct_lp - incorrect_lp


def factual_decoder_fn(outputs, inputs, model, tokenizer,
                       correct_tokens: List[int],
                       incorrect_tokens: List[int], **kwargs):
    """Decoder LM - score at last position (next token prediction)."""
    logits = outputs.logits
    last_logits = logits[:, -1, :]
    log_probs = F.log_softmax(last_logits, dim=-1)
    
    correct_lp = log_probs[:, correct_tokens].mean() if correct_tokens else torch.tensor(0.0)
    incorrect_lp = log_probs[:, incorrect_tokens].mean() if incorrect_tokens else torch.tensor(0.0)
    
    return (correct_lp - incorrect_lp).mean()


def factual_encdec_fn(outputs, inputs, model, tokenizer,
                      correct_tokens: List[int],
                      incorrect_tokens: List[int], **kwargs):
    """Encoder-decoder - score at first decoder position."""
    logits = outputs.logits
    first_logits = logits[:, 0, :]
    log_probs = F.log_softmax(first_logits, dim=-1)
    
    correct_lp = log_probs[:, correct_tokens].mean() if correct_tokens else torch.tensor(0.0)
    incorrect_lp = log_probs[:, incorrect_tokens].mean() if incorrect_tokens else torch.tensor(0.0)
    
    return (correct_lp - incorrect_lp).mean()


# =============================================================================
# QA OUTPUT FUNCTIONS (same pattern as factual)
# =============================================================================

qa_encoder_fn = factual_encoder_mlm_fn
qa_decoder_fn = factual_decoder_fn
qa_encdec_fn = factual_encdec_fn


# =============================================================================
# OUTPUT FUNCTION REGISTRY
# =============================================================================

OUTPUT_FN_REGISTRY = {
    ('sentiment', 'encoder'): sentiment_encoder_fn,
    ('sentiment', 'decoder'): sentiment_decoder_fn,
    ('sentiment', 'encoder_decoder'): sentiment_encdec_fn,
    
    ('nli', 'encoder'): nli_encoder_fn,
    ('nli', 'decoder'): nli_decoder_fn,
    ('nli', 'encoder_decoder'): nli_encdec_fn,
    
    ('factual_recall', 'encoder'): factual_encoder_mlm_fn,
    ('factual_recall', 'decoder'): factual_decoder_fn,
    ('factual_recall', 'encoder_decoder'): factual_encdec_fn,
    
    ('qa', 'encoder'): qa_encoder_fn,
    ('qa', 'decoder'): qa_decoder_fn,
    ('qa', 'encoder_decoder'): qa_encdec_fn,
}


def get_output_fn(
    task: str,
    architecture: str,
    model,
    tokenizer,
    correct_tokens: List[int] = None,
    incorrect_tokens: List[int] = None
) -> Callable:
    """
    Get the appropriate output function for a task/architecture combination.
    
    Args:
        task: 'sentiment', 'nli', 'factual_recall', or 'qa'
        architecture: 'encoder', 'decoder', or 'encoder_decoder'
        model: The model (needed for encoder classification)
        tokenizer: The tokenizer
        correct_tokens: Token IDs for correct answer (required for factual/qa)
        incorrect_tokens: Token IDs for incorrect answer (required for factual/qa)
    
    Returns:
        A callable output function
    """
    key = (task, architecture)
    if key not in OUTPUT_FN_REGISTRY:
        raise ValueError(f"No output function for task={task}, architecture={architecture}")
    
    base_fn = OUTPUT_FN_REGISTRY[key]
    
    # Create partial with common args
    if task in ['sentiment', 'nli']:
        return partial(base_fn, model=model, tokenizer=tokenizer)
    else:  # factual_recall or qa
        if correct_tokens is None or incorrect_tokens is None:
            raise ValueError(f"{task} requires correct_tokens and incorrect_tokens")
        return partial(base_fn, model=model, tokenizer=tokenizer,
                      correct_tokens=correct_tokens, incorrect_tokens=incorrect_tokens)


# =============================================================================
# TIER 3 EXAMPLE PREPARATION
# =============================================================================

def prepare_factual_tier3_pairs(
    raw_data: List[Tuple[str, str, str]],  # (subject, relation, object) tuples
    templates: Dict[str, str],
    tokenizer,
    architecture: str,
    num_pairs: int,
    seed: int = None
) -> List[Tier3ExamplePair]:
    """
    Prepare factual recall pairs for Tier 3.
    
    CRITICAL: Input text does NOT include the answer.
    """
    if seed is not None:
        random.seed(seed)
    
    # Sample data
    sampled = random.sample(raw_data, min(num_pairs, len(raw_data)))
    
    # Get all objects for creating negatives
    all_objects = list(set(obj for _, _, obj in raw_data))
    
    pairs = []
    for subject, relation, correct_obj in sampled:
        # Create prompt WITHOUT the answer
        template = templates.get(relation, "{subject} {relation}")
        prompt = template.format(subject=subject, relation=relation).rstrip(' .')
        
        # Get wrong object
        wrong_obj = random.choice(all_objects)
        while wrong_obj == correct_obj:
            wrong_obj = random.choice(all_objects)
        
        # Format based on architecture
        if architecture == "encoder":
            input_text = f"{prompt} {tokenizer.mask_token}."
            correct_tokens = tokenizer.encode(correct_obj, add_special_tokens=False)
            incorrect_tokens = tokenizer.encode(wrong_obj, add_special_tokens=False)
        elif architecture == "decoder":
            input_text = prompt
            correct_tokens = tokenizer.encode(' ' + correct_obj, add_special_tokens=False)
            incorrect_tokens = tokenizer.encode(' ' + wrong_obj, add_special_tokens=False)
        else:  # encoder_decoder
            input_text = f"complete: {prompt}"
            correct_tokens = tokenizer.encode(correct_obj, add_special_tokens=False)
            incorrect_tokens = tokenizer.encode(wrong_obj, add_special_tokens=False)
        
        pairs.append(Tier3ExamplePair(
            positive_input=input_text,
            negative_input=input_text,  # Same input, different scoring
            correct_tokens=correct_tokens,
            incorrect_tokens=incorrect_tokens
        ))
    
    return pairs


def prepare_qa_tier3_pairs(
    raw_data: List[Dict],  # List of {'context': ..., 'question': ..., 'answer': ...}
    tokenizer,
    architecture: str,
    num_pairs: int,
    seed: int = None
) -> List[Tier3ExamplePair]:
    """
    Prepare QA pairs for Tier 3.
    
    CRITICAL: Input text does NOT include the answer.
    """
    if seed is not None:
        random.seed(seed)
    
    sampled = random.sample(raw_data, min(num_pairs, len(raw_data)))
    all_answers = list(set(item.get('answer', item.get('answers', [''])[0]) 
                          for item in raw_data if item.get('answer') or item.get('answers')))
    
    pairs = []
    for item in sampled:
        context = item.get('context', '')
        question = item.get('question', '')
        correct_ans = item.get('answer', item.get('answers', [''])[0])
        
        # Get wrong answer
        wrong_ans = random.choice(all_answers) if all_answers else "unknown"
        while wrong_ans == correct_ans and len(all_answers) > 1:
            wrong_ans = random.choice(all_answers)
        
        # Format WITHOUT the answer
        if architecture == "encoder":
            input_text = f"{question} {tokenizer.sep_token} {context}"
            correct_tokens = tokenizer.encode(correct_ans, add_special_tokens=False)
            incorrect_tokens = tokenizer.encode(wrong_ans, add_special_tokens=False)
        elif architecture == "decoder":
            input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
            correct_tokens = tokenizer.encode(' ' + correct_ans, add_special_tokens=False)
            incorrect_tokens = tokenizer.encode(' ' + wrong_ans, add_special_tokens=False)
        else:  # encoder_decoder
            input_text = f"question: {question} context: {context}"
            correct_tokens = tokenizer.encode(correct_ans, add_special_tokens=False)
            incorrect_tokens = tokenizer.encode(wrong_ans, add_special_tokens=False)
        
        pairs.append(Tier3ExamplePair(
            positive_input=input_text,
            negative_input=input_text,
            correct_tokens=correct_tokens,
            incorrect_tokens=incorrect_tokens
        ))
    
    return pairs


# =============================================================================
# VALIDATION HELPER
# =============================================================================

def validate_output_fn(
    output_fn: Callable,
    model,
    tokenizer,
    positive_text: str,
    negative_text: str,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Validate that an output function produces a meaningful gap.
    
    Returns dict with positive_score, negative_score, gap, valid.
    """
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        # Score positive
        pos_inputs = tokenizer(positive_text, return_tensors='pt', 
                               padding=True, truncation=True).to(device)
        pos_outputs = model(**pos_inputs)
        pos_score = output_fn(pos_outputs, pos_inputs, model, tokenizer).item()
        
        # Score negative
        neg_inputs = tokenizer(negative_text, return_tensors='pt',
                               padding=True, truncation=True).to(device)
        neg_outputs = model(**neg_inputs)
        neg_score = output_fn(neg_outputs, neg_inputs, model, tokenizer).item()
    
    gap = pos_score - neg_score
    
    return {
        'positive_score': pos_score,
        'negative_score': neg_score,
        'gap': gap,
        'valid': abs(gap) > 0.01
    }
