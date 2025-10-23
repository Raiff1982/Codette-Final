"""
This module defines the Element class for modeling atomic and computational elements.
"""

from typing import List, Dict, Any

class Element:
    """Represents an atomic/computational element with defense capabilities"""
    
    def __init__(self, name: str, symbol: str, programming_lang: str,
                 abilities: List[str], vulnerabilities: List[str], defense_type: str):
        self.name = name
        self.symbol = symbol
        self.programming_lang = programming_lang
        self.abilities = abilities
        self.vulnerabilities = vulnerabilities
        self.defense_type = defense_type
        self.active_defenses = []

    def execute_defense_function(self, core, response_modifiers: List, response_filters: List):
        """Execute the element's defense function based on its type"""
        if self.defense_type == "evasion":
            self._apply_evasion_strategy(response_modifiers, response_filters)
        elif self.defense_type == "adaptability":
            self._apply_adaptability_strategy(response_modifiers)
        elif self.defense_type == "fortification":
            self._apply_fortification_strategy(response_filters)
        elif self.defense_type == "barrier":
            self._apply_barrier_strategy(response_filters)
        elif self.defense_type == "regeneration":
            self._apply_regeneration_strategy(response_modifiers)

    def _apply_evasion_strategy(self, response_modifiers: List, response_filters: List):
        """Apply evasion-based defense strategy"""
        def evasion_modifier(response: str) -> str:
            # Add evasion patterns to response
            return response.replace("[VULNERABLE]", "[PROTECTED]")
        response_modifiers.append(evasion_modifier)

    def _apply_adaptability_strategy(self, response_modifiers: List):
        """Apply adaptability-based defense strategy"""
        def adaptability_modifier(response: str) -> str:
            # Add adaptability patterns
            return response + "\n[Adaptive Defense Active]"
        response_modifiers.append(adaptability_modifier)

    def _apply_fortification_strategy(self, response_filters: List):
        """Apply fortification-based defense strategy"""
        def fortification_filter(response: str) -> str:
            # Apply security filtering
            return response.replace("sensitive_data", "[REDACTED]")
        response_filters.append(fortification_filter)

    def _apply_barrier_strategy(self, response_filters: List):
        """Apply barrier-based defense strategy"""
        def barrier_filter(response: str) -> str:
            # Apply barrier protection
            return "[BARRIER PROTECTED]\n" + response
        response_filters.append(barrier_filter)

    def _apply_regeneration_strategy(self, response_modifiers: List):
        """Apply regeneration-based defense strategy"""
        def regeneration_modifier(response: str) -> str:
            # Add self-healing capabilities
            return response + "\n[Self-Healing Active]"
        response_modifiers.append(regeneration_modifier)

    def activate_defense(self, defense_name: str) -> bool:
        """Activate a specific defense mechanism"""
        if defense_name not in self.active_defenses:
            self.active_defenses.append(defense_name)
            return True
        return False

    def deactivate_defense(self, defense_name: str) -> bool:
        """Deactivate a specific defense mechanism"""
        if defense_name in self.active_defenses:
            self.active_defenses.remove(defense_name)
            return True
        return False

    def get_element_info(self) -> Dict[str, Any]:
        """Get information about the element"""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "programming_lang": self.programming_lang,
            "abilities": self.abilities,
            "vulnerabilities": self.vulnerabilities,
            "defense_type": self.defense_type,
            "active_defenses": self.active_defenses
        }

    def __str__(self) -> str:
        return f"{self.name} ({self.symbol}) - {self.defense_type} defense"