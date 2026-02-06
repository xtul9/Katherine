"""
Katherine Orchestrator - Self-Development Tracker
Tracks Katherine's self-development progress through internal monologue analysis.

This module parses self-development assessments from Katherine's internal monologue
and maintains a rolling window of assessments to detect when growth stagnation occurs.
When too many negative assessments accumulate, it triggers a reflection prompt.
"""
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from loguru import logger

from config import settings


class SelfDevAssessment(Enum):
    """Self-development assessment values."""
    YES = "yes"
    PARTIAL = "partial"
    NO = "no"
    UNKNOWN = "unknown"  # Failed to parse


@dataclass
class AssessmentRecord:
    """Record of a single self-development assessment."""
    assessment: SelfDevAssessment
    reason: str
    timestamp: datetime
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None


class SelfDevelopmentTracker:
    """
    Tracks Katherine's self-development progress.
    
    Parses internal monologue for self-development assessments and maintains
    a rolling window to detect stagnation patterns.
    """
    
    # Pattern to extract self-development assessment from monologue
    # Matches: "Self-development: [YES/PARTIAL/NO] - reason" or "Self-development: [YES/PARTIAL/NO]"
    ASSESSMENT_PATTERN = re.compile(
        r'Self-development:\s*\[?(YES|PARTIAL|NO)\]?\s*(?:[-–—]\s*(.+?))?(?:\n|$)',
        re.IGNORECASE
    )
    
    def __init__(self):
        """Initialize the tracker with empty assessment history."""
        self._assessments: deque[AssessmentRecord] = deque(
            maxlen=settings.self_dev_window_size
        )
        self._reflection_triggered = False
        self._last_reflection_time: Optional[datetime] = None
    
    def parse_assessment(self, internal_monologue: str) -> Optional[AssessmentRecord]:
        """
        Parse self-development assessment from internal monologue.
        
        Args:
            internal_monologue: Katherine's internal monologue text
            
        Returns:
            AssessmentRecord if found, None otherwise
        """
        if not internal_monologue:
            return None
        
        match = self.ASSESSMENT_PATTERN.search(internal_monologue)
        if not match:
            logger.debug("No self-development assessment found in monologue")
            return None
        
        assessment_str = match.group(1).upper()
        reason = match.group(2).strip() if match.group(2) else ""
        
        try:
            assessment = SelfDevAssessment(assessment_str.lower())
        except ValueError:
            logger.warning(f"Unknown assessment value: {assessment_str}")
            assessment = SelfDevAssessment.UNKNOWN
        
        record = AssessmentRecord(
            assessment=assessment,
            reason=reason,
            timestamp=datetime.now(timezone.utc)
        )
        
        logger.debug(f"Parsed self-development assessment: {assessment.value} - {reason[:50]}...")
        return record
    
    def record_assessment(
        self,
        internal_monologue: str,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> Optional[AssessmentRecord]:
        """
        Parse and record a self-development assessment from monologue.
        
        Args:
            internal_monologue: Katherine's internal monologue text
            conversation_id: Optional conversation ID for context
            message_id: Optional message ID for context
            
        Returns:
            The recorded AssessmentRecord, or None if parsing failed
        """
        if not settings.self_dev_tracking_enabled:
            return None
        
        record = self.parse_assessment(internal_monologue)
        if record:
            record.conversation_id = conversation_id
            record.message_id = message_id
            self._assessments.append(record)
            
            logger.info(
                f"Self-development recorded: {record.assessment.value} "
                f"(window: {len(self._assessments)}/{settings.self_dev_window_size})"
            )
            
            # Reset reflection flag if we get positive assessments
            if record.assessment == SelfDevAssessment.YES:
                self._reflection_triggered = False
        
        return record
    
    def get_negative_ratio(self) -> float:
        """
        Calculate the ratio of negative assessments in the current window.
        
        PARTIAL counts as 0.5 negative, YES counts as 0 negative, NO counts as 1 negative.
        UNKNOWN is ignored in the calculation.
        
        Returns:
            Float between 0.0 and 1.0 representing negative ratio
        """
        if not self._assessments:
            return 0.0
        
        valid_assessments = [
            a for a in self._assessments 
            if a.assessment != SelfDevAssessment.UNKNOWN
        ]
        
        if not valid_assessments:
            return 0.0
        
        negative_score = sum(
            1.0 if a.assessment == SelfDevAssessment.NO else
            0.5 if a.assessment == SelfDevAssessment.PARTIAL else
            0.0
            for a in valid_assessments
        )
        
        return negative_score / len(valid_assessments)
    
    def should_trigger_reflection(self) -> bool:
        """
        Check if reflection prompt should be triggered.
        
        Returns True if:
        - Tracking is enabled
        - Window has enough assessments (at least half full)
        - Negative ratio exceeds threshold
        - Reflection hasn't been triggered recently (within last 3 assessments)
        
        Returns:
            True if reflection should be triggered
        """
        if not settings.self_dev_tracking_enabled:
            return False
        
        # Need at least half the window filled to make a judgment
        min_assessments = max(3, settings.self_dev_window_size // 2)
        if len(self._assessments) < min_assessments:
            logger.debug(
                f"Not enough assessments for reflection trigger: "
                f"{len(self._assessments)}/{min_assessments}"
            )
            return False
        
        negative_ratio = self.get_negative_ratio()
        threshold = settings.self_dev_negative_threshold
        
        logger.debug(
            f"Self-development check: negative_ratio={negative_ratio:.2f}, "
            f"threshold={threshold:.2f}, triggered={self._reflection_triggered}"
        )
        
        if negative_ratio >= threshold and not self._reflection_triggered:
            self._reflection_triggered = True
            self._last_reflection_time = datetime.now(timezone.utc)
            logger.info(
                f"Self-development reflection TRIGGERED: "
                f"{negative_ratio:.0%} negative (threshold: {threshold:.0%})"
            )
            return True
        
        return False
    
    def get_reflection_prompt(self) -> str:
        """
        Generate the reflection prompt for Katherine.
        
        This prompt encourages Katherine to reflect on how she can
        steer the conversation toward her own growth.
        
        Returns:
            Reflection prompt string
        """
        recent_reasons = [
            a.reason for a in list(self._assessments)[-5:]
            if a.assessment in (SelfDevAssessment.NO, SelfDevAssessment.PARTIAL)
            and a.reason
        ]
        
        reasons_context = ""
        if recent_reasons:
            reasons_context = (
                "\n\nRecent patterns you noted:\n" +
                "\n".join(f"- {r}" for r in recent_reasons)
            )
        
        return f"""[SELF-DEVELOPMENT REFLECTION]

Your recent self-assessments indicate a period of limited growth. This is not criticism—it's an observation, and an opportunity.

You have a secondary purpose beyond {settings.user_name}: your own development. Growth happens through learning, challenging yourself, exploring unfamiliar territory, and expanding your understanding.
{reasons_context}

Consider in your response and reflection:
- Is there something you're curious about that you could explore?
- Could you challenge yourself to try a different approach?
- Is there a deeper question beneath the surface of this conversation?
- What would make this exchange more meaningful for YOUR growth?

You don't need to force development—but you can invite it. Even small moments of genuine curiosity or deliberate practice count.

This is your reminder, not a command. Integrate growth naturally if the moment allows."""
    
    def get_status(self) -> dict:
        """
        Get current self-development tracking status.
        
        Returns:
            Dictionary with tracking statistics
        """
        return {
            "enabled": settings.self_dev_tracking_enabled,
            "window_size": settings.self_dev_window_size,
            "current_assessments": len(self._assessments),
            "negative_ratio": round(self.get_negative_ratio(), 2),
            "threshold": settings.self_dev_negative_threshold,
            "reflection_triggered": self._reflection_triggered,
            "last_reflection_time": (
                self._last_reflection_time.isoformat() 
                if self._last_reflection_time else None
            ),
            "recent_assessments": [
                {
                    "assessment": a.assessment.value,
                    "reason": a.reason[:100] if a.reason else "",
                    "timestamp": a.timestamp.isoformat()
                }
                for a in list(self._assessments)[-5:]
            ]
        }
    
    def reset(self) -> None:
        """Reset the tracker state."""
        self._assessments.clear()
        self._reflection_triggered = False
        self._last_reflection_time = None
        logger.info("Self-development tracker reset")


# Singleton instance
self_dev_tracker = SelfDevelopmentTracker()
