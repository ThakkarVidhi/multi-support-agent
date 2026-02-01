"""Tests for intent classification and entity extraction."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from src.agent.intent import (
    INTENT_BOTH,
    INTENT_CUSTOMER,
    INTENT_POLICY,
    classify_intent_and_entities,
)


def test_intent_policy_only():
    """Policy-only questions -> policy intent."""
    r = classify_intent_and_entities("What is the current refund policy?")
    assert r.intent == INTENT_POLICY
    assert r.confidence >= 0.9


def test_intent_customer_only():
    """Customer/ticket questions -> customer intent."""
    r = classify_intent_and_entities("Give me a quick overview of customer Denise Lee profile and past support ticket details.")
    assert r.intent == INTENT_CUSTOMER
    assert r.customer_name == "Denise Lee"
    assert r.confidence >= 0.8


def test_intent_both_qualify():
    """'Does X qualify under refund policy?' -> both intent and extracts name."""
    r = classify_intent_and_entities("Does Denise Lee qualify under the refund policy?")
    assert r.intent == INTENT_BOTH
    assert r.customer_name == "Denise Lee"
    assert r.confidence >= 0.8


def test_entity_extraction_customer_name():
    """Extract customer name from various phrasings."""
    r = classify_intent_and_entities("Overview of customer John Smith and tickets.")
    assert r.intent == INTENT_CUSTOMER
    assert r.customer_name == "John Smith"


def test_empty_message():
    """Empty message -> both with zero confidence."""
    r = classify_intent_and_entities("")
    assert r.intent == INTENT_BOTH
    assert r.confidence == 0.0
