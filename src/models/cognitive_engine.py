"""
This module implements the broader perspective engine for cognitive processing.
"""

from typing import Dict, Any, List, Optional, Callable

class BroaderPerspectiveEngine:
    """Processes information through multiple cognitive perspectives"""
    
    def __init__(self):
        self.available_perspectives = {
            "analytical": self._analytical_perspective,
            "creative": self._creative_perspective,
            "emotional": self._emotional_perspective,
            "critical": self._critical_perspective,
            "integrative": self._integrative_perspective,
            "systemic": self._systemic_perspective
        }
        self.active_perspectives = []

    def get_perspective_method(self, perspective: str) -> Optional[Callable]:
        """Get the method associated with a perspective"""
        return self.available_perspectives.get(perspective)

    def process_through_perspectives(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process input through all active perspectives"""
        results = []
        for perspective in self.active_perspectives:
            if perspective in self.available_perspectives:
                try:
                    result = self.available_perspectives[perspective](input_data)
                    results.append({
                        "perspective": perspective,
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "perspective": perspective,
                        "error": str(e)
                    })
        return results

    def _analytical_perspective(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process through analytical lens"""
        return {
            "type": "analytical",
            "focus": "logical structure and patterns",
            "processed_data": data
        }

    def _creative_perspective(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process through creative lens"""
        return {
            "type": "creative",
            "focus": "novel connections and possibilities",
            "processed_data": data
        }

    def _emotional_perspective(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process through emotional lens"""
        return {
            "type": "emotional",
            "focus": "emotional undertones and impact",
            "processed_data": data
        }

    def _critical_perspective(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process through critical lens"""
        return {
            "type": "critical",
            "focus": "evaluation and assessment",
            "processed_data": data
        }

    def _integrative_perspective(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process through integrative lens"""
        return {
            "type": "integrative",
            "focus": "synthesis and holistic view",
            "processed_data": data
        }

    def _systemic_perspective(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process through systemic lens"""
        return {
            "type": "systemic",
            "focus": "system-level interactions",
            "processed_data": data
        }

    def activate_perspective(self, perspective: str) -> bool:
        """Activate a specific perspective"""
        if perspective in self.available_perspectives and perspective not in self.active_perspectives:
            self.active_perspectives.append(perspective)
            return True
        return False

    def deactivate_perspective(self, perspective: str) -> bool:
        """Deactivate a specific perspective"""
        if perspective in self.active_perspectives:
            self.active_perspectives.remove(perspective)
            return True
        return False