"""Intent classification and entity extraction. Routes policy vs customer queries."""
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

INTENT_POLICY = "policy"
INTENT_CUSTOMER = "customer"
INTENT_BOTH = "both"


@dataclass
class IntentResult:
    intent: str  # "policy" | "customer" | "both"
    confidence: float
    customer_name: Optional[str] = None
    ticket_id: Optional[str] = None
    raw_json: Optional[str] = None
    entities: Optional[dict] = None  # email, product, ticket_status, etc.


def _has_person_name(text: str) -> bool:
    """True if message contains a person name (e.g. 'Denise Lee') even without the word 'customer'."""
    # Two capitalized words (First Last) or name after "customer"
    if re.search(r"\bcustomer\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", text):
        return True
    if re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:qualify|profile|details|tickets|bought|purchased)", text):
        return True
    if re.search(r"(?:Does|Did|Has)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+", text):
        return True
    return False


def _extract_customer_name_from_text(text: str) -> Optional[str]:
    """Extract customer name: e.g. 'customer Denise Lee' or 'Denise Lee qualify'."""
    # "customer Denise Lee" or "customer Denise Lee's profile" or "customer Denise Lee profile"
    m = re.search(r"\bcustomer\s+([A-Za-z][A-Za-z\s]+?)(?:\'s|\s+profile|\s+details|$)", text, re.I | re.DOTALL)
    if m:
        return m.group(1).strip()
    # "Does Denise Lee qualify" or "Did Denise Lee buy"
    m = re.search(r"(?:Does|Did|Has)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+", text)
    if m:
        return m.group(1).strip()
    # "overview of customer Denise Lee" - two-word name after "customer"
    m = re.search(r"\bcustomer\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", text)
    if m:
        return m.group(1).strip()
    # "Denise Lee profile" or "Denise Lee qualify" without "customer" before
    m = re.search(r"([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+profile|\s+details|\s+qualify|\s+buy)?", text)
    if m:
        return m.group(1).strip()
    return None


def _extract_entities(text: str) -> dict:
    """Extract customer_name, email, product, ticket_id, ticket_status, etc."""
    entities = {}
    name = _extract_customer_name_from_text(text)
    if name:
        entities["customer_name"] = name
    # Email
    m = re.search(r"[\w.+%-]+@[\w.-]+\.\w+", text)
    if m:
        entities["customer_email"] = m.group(0)
    # Product (e.g. "Philips Light", "Philips Hue Lights", "Did Denise Lee buy Philips Light")
    m = re.search(r"(?:buy|bought|purchase|product)\s+[\'\"]?([A-Za-z0-9\s]+?)[\'\"]?(?:\?|\.|,|$)", text, re.I)
    if m:
        entities["product_purchased"] = m.group(1).strip()
    # Alternative: "X buy Y" -> product Y
    m = re.search(r"\b(?:buy|bought)\s+([A-Za-z0-9][A-Za-z0-9\s]*?)(?:\?|\.|$)", text, re.I)
    if m and "product" not in entities:
        entities["product_purchased"] = m.group(1).strip()
    # Ticket ID (e.g. T001, ticket 123)
    m = re.search(r"ticket\s*(?:id)?\s*[#:]?\s*([A-Z]?\d+)", text, re.I)
    if m:
        entities["ticket_id"] = m.group(1)
    # Ticket status
    for status in ("open", "pending", "resolved", "closed", "in progress"):
        if re.search(rf"\b{status}\b", text, re.I):
            entities["ticket_status"] = status
            break
    return entities


def classify_intent_and_entities(user_message: str, llm=None) -> IntentResult:
    """
    Classify intent (policy / customer / both) and extract entities.
    Uses LLM when available; falls back to rules.
    """
    message = (user_message or "").strip()
    if not message:
        return IntentResult(intent=INTENT_BOTH, confidence=0.0)

    # Heuristics for policy-only
    policy_keywords = [
        "refund policy", "refund policy?", "current refund", "what is the refund",
        "policy", "terms", "cancellation", "qualify under", "qualify for refund",
        "policy document", "legal", "company policy"
    ]
    # Customer/ticket keywords
    customer_keywords = [
        "customer", "profile", "support ticket", "ticket details", "ticket history",
        "overview of customer", "customer's profile", "past support", "tickets for"
    ]

    msg_lower = message.lower()
    has_policy = any(k in msg_lower for k in policy_keywords) or "policy" in msg_lower
    has_customer = any(k in msg_lower for k in customer_keywords) or "customer" in msg_lower
    has_person_or_entity = _has_person_name(message) or bool(_extract_entities(message))

    # "Does Denise Lee qualify under refund policy?" -> both (person name + policy, even without word "customer")
    if has_policy and (has_customer or has_person_or_entity):
        entities = _extract_entities(message)
        name = entities.get("customer_name") or _extract_customer_name_from_text(message)
        return IntentResult(
            intent=INTENT_BOTH,
            confidence=0.9,
            customer_name=name,
            ticket_id=entities.get("ticket_id"),
            entities=entities or None,
            raw_json=json.dumps({"intent": INTENT_BOTH, "customer_name": name, "confidence": 0.9})
        )
    
    # Questions purely about policy (no customer/person mention)
    if has_policy and not has_customer and not has_person_or_entity:
        return IntentResult(
            intent=INTENT_POLICY,
            confidence=0.95,
            entities={},
            raw_json=json.dumps({"intent": INTENT_POLICY, "confidence": 0.95})
        )
    
    # Customer/ticket only (name, email, product, ticket, etc.)
    if has_customer or has_person_or_entity:
        entities = _extract_entities(message)
        name = entities.get("customer_name") or _extract_customer_name_from_text(message)
        return IntentResult(
            intent=INTENT_CUSTOMER,
            confidence=0.9,
            customer_name=name,
            ticket_id=entities.get("ticket_id"),
            entities=entities or None,
            raw_json=json.dumps({"intent": INTENT_CUSTOMER, "customer_name": name, "confidence": 0.9})
        )

    # LLM for ambiguous cases
    if llm is not None:
        try:
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an intent classifier. Reply with ONLY a JSON object, no other text.
Keys: "intent" (one of: policy, customer, both), "confidence" (0-1), "customer_name" (full name if mentioned, else null), "ticket_id" (if mentioned, else null).
- policy: question is only about company policy, refund, terms, documents.
- customer: question is only about a specific customer's profile, tickets, support history.
- both: question needs both (e.g. "Does customer X qualify under refund policy?")."""),
                ("human", "{message}"),
            ])
            resp = (prompt | llm).invoke({"message": message})
            content = resp.content if hasattr(resp, "content") else str(resp)
            content = content.strip()
            if "```" in content:
                for part in content.split("```"):
                    if "{" in part:
                        content = part.strip()
                        if content.startswith("json"):
                            content = content[4:].strip()
                        break
            obj = json.loads(content)
            intent = (obj.get("intent") or "both").lower()
            if intent not in (INTENT_POLICY, INTENT_CUSTOMER, INTENT_BOTH):
                intent = INTENT_BOTH
            entities = _extract_entities(message)
            return IntentResult(
                intent=intent,
                confidence=float(obj.get("confidence", 0.8)),
                customer_name=obj.get("customer_name") or _extract_customer_name_from_text(message),
                ticket_id=obj.get("ticket_id"),
                entities=entities or None,
                raw_json=content,
            )
        except Exception as e:
            logger.debug("LLM intent fallback failed: %s", e)

    # Default: both
    entities = _extract_entities(message)
    return IntentResult(
        intent=INTENT_BOTH,
        confidence=0.5,
        customer_name=_extract_customer_name_from_text(message),
        entities=entities or None,
        raw_json=json.dumps({"intent": "both", "confidence": 0.5}),
    )
