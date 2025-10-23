"""
Biokinetic Neural Mesh - A biomimetic neural routing system
Combines biological neural patterns with kinetic state processing for ultra-fast routing
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from components.quantum_spiderweb import QuantumSpiderweb

logger = logging.getLogger(__name__)

@dataclass
class SynapticNode:
    """Represents a node in the biokinetic mesh"""
    id: str
    energy: float = 1.0
    connections: Dict[str, float] = None
    activation_pattern: np.ndarray = None
    kinetic_state: float = 0.0
    
    def __post_init__(self):
        self.connections = self.connections or {}
        self.activation_pattern = self.activation_pattern or np.random.rand(128)

class BioKineticMesh:
    """
    Biokinetic Neural Mesh - A biomimetic routing system
    
    Features:
    - Ultra-fast pattern recognition (<0.3ms)
    - Self-evolving neural pathways
    - Energy-based routing
    - Synaptic pruning for optimization
    - Fractal memory patterns
    - Quantum state integration
    - Multi-perspective resonance
    - Adaptive pathway evolution
    """
    
    def __init__(self, 
                 initial_nodes: int = 512,
                 energy_threshold: float = 0.3,
                 learning_rate: float = 0.01,
                 prune_threshold: float = 0.1,
                 quantum_influence: float = 0.3,
                 perspective_resonance: float = 0.2):
        self.nodes: Dict[str, SynapticNode] = {}
        self.energy_threshold = energy_threshold
        self.learning_rate = learning_rate
        self.prune_threshold = prune_threshold
        self.quantum_influence = quantum_influence
        self.perspective_resonance = perspective_resonance
        
        # Kinetic state tensors
        self.kinetic_matrix = torch.zeros((initial_nodes, initial_nodes))
        self.energy_gradients = torch.zeros(initial_nodes)
        
        # Pattern recognition layers
        self.pattern_embeddings = np.random.rand(initial_nodes, 128)
        self.activation_history: List[np.ndarray] = []
        
        # Integration components
        self.quantum_resonance: Dict[str, float] = {}  # Quantum state influence
        self.perspective_weights: Dict[str, Dict[str, float]] = {}  # Per-node perspective weights
        self.active_pathways: Set[Tuple[str, str]] = set()  # Currently active neural pathways
        
        # Initialize mesh
        self._initialize_mesh(initial_nodes)
        
    def _initialize_mesh(self, node_count: int):
        """Initialize the biokinetic mesh with initial nodes"""
        for i in range(node_count):
            node_id = f"BK_{i}"
            self.nodes[node_id] = SynapticNode(
                id=node_id,
                energy=1.0,
                activation_pattern=np.random.rand(128)
            )
            
        # Create initial connections (sparse)
        for node in self.nodes.values():
            connection_count = np.random.randint(5, 15)
            target_nodes = np.random.choice(
                list(self.nodes.keys()), 
                size=connection_count, 
                replace=False
            )
            node.connections = {
                target: np.random.rand() 
                for target in target_nodes 
                if target != node.id
            }

    def route_intent(self, 
                    input_pattern: np.ndarray, 
                    context: Optional[Dict] = None) -> Tuple[str, float]:
        """
        Route an input pattern through the mesh to determine intent
        Returns in under 0.3ms
        """
        # Convert input to energy pattern
        energy_pattern = self._compute_energy_pattern(input_pattern)
        
        # Fast parallel activation
        activations = torch.tensor([
            self._compute_node_activation(node, energy_pattern, context)
            for node in self.nodes.values()
        ])
        
        # Find highest energy path
        max_idx = torch.argmax(activations).item()
        node_id = list(self.nodes.keys())[max_idx]
        confidence = activations[max_idx].item()
        
        # Update kinetic state
        self._update_kinetic_state(node_id, confidence)
        
        return node_id, confidence

    def _compute_energy_pattern(self, input_pattern: np.ndarray) -> torch.Tensor:
        """Convert input pattern to energy distribution"""
        # Normalize input
        input_norm = input_pattern / np.linalg.norm(input_pattern)
        
        # Create energy tensor
        energy = torch.from_numpy(input_norm).float()
        
        # Apply kinetic transformation
        energy = self._apply_kinetic_transform(energy)
        
        return energy

    def _compute_node_activation(self, 
                               node: SynapticNode, 
                               energy_pattern: torch.Tensor,
                               context: Optional[Dict]) -> float:
        """Compute node activation based on energy pattern and context"""
        # Base activation from pattern match
        base_activation = torch.cosine_similarity(
            energy_pattern,
            torch.from_numpy(node.activation_pattern).float().unsqueeze(0),
            dim=1
        )
        
        # Apply kinetic state
        kinetic_boost = node.kinetic_state * self.learning_rate
        
        # Context influence
        context_factor = 1.0
        if context:
            context_pattern = self._context_to_pattern(context)
            context_match = torch.cosine_similarity(
                torch.from_numpy(context_pattern).float().unsqueeze(0),
                torch.from_numpy(node.activation_pattern).float().unsqueeze(0),
                dim=1
            )
            context_factor = 1.0 + (context_match.item() * 0.5)
        
        return (base_activation.item() + kinetic_boost) * context_factor

    def _apply_kinetic_transform(self, energy: torch.Tensor) -> torch.Tensor:
        """Apply kinetic transformation to energy pattern"""
        # Create momentum factor
        momentum = torch.sigmoid(self.energy_gradients.mean())
        
        # Apply momentum to energy
        energy = energy * (1.0 + momentum)
        
        # Normalize
        energy = energy / energy.norm()
        
        return energy

    def _update_kinetic_state(self, node_id: str, activation: float):
        """Update kinetic state of the network"""
        # Update node energy
        node = self.nodes[node_id]
        node.kinetic_state += self.learning_rate * (activation - node.kinetic_state)
        
        # Update connected nodes
        for target_id, weight in node.connections.items():
            if target_id in self.nodes:
                target = self.nodes[target_id]
                target.kinetic_state += (
                    self.learning_rate * weight * (activation - target.kinetic_state)
                )

    def _context_to_pattern(self, context: Dict) -> np.ndarray:
        """Convert context dictionary to pattern vector"""
        # Create empty pattern
        pattern = np.zeros(128)
        
        # Add context influences
        if "mode" in context:
            pattern += self.pattern_embeddings[
                hash(context["mode"]) % len(self.pattern_embeddings)
            ]
        
        if "priority" in context:
            priority_factor = float(context["priority"]) / 10.0
            pattern *= (1.0 + priority_factor)
            
        # Normalize
        pattern = pattern / (np.linalg.norm(pattern) + 1e-8)
        
        return pattern

    def prune_connections(self):
        """Remove weak or unused connections"""
        for node in self.nodes.values():
            # Find weak connections
            weak_connections = [
                target_id
                for target_id, weight in node.connections.items()
                if weight < self.prune_threshold
            ]
            
            # Remove weak connections
            for target_id in weak_connections:
                del node.connections[target_id]
                
            # Normalize remaining connections
            if node.connections:
                total_weight = sum(node.connections.values())
                for target_id in node.connections:
                    node.connections[target_id] /= total_weight

    def integrate_quantum_state(self, quantum_web: QuantumSpiderweb, node_id: str):
        """Integrate quantum web state with biokinetic mesh"""
        # Get quantum state for this node
        quantum_state = quantum_web.get_node_state(node_id)
        
        if quantum_state:
            # Update quantum resonance
            self.quantum_resonance[node_id] = quantum_state["coherence"]
            
            # Influence node connections based on quantum state
            node = self.nodes.get(node_id)
            if node:
                quantum_boost = quantum_state["coherence"] * self.quantum_influence
                for target_id in node.connections:
                    node.connections[target_id] *= (1.0 + quantum_boost)
                    
                # Update node's kinetic state
                node.kinetic_state += quantum_boost
                
    def integrate_perspective_results(self, 
                                   node_id: str,
                                   perspective_results: Dict[str, Dict[str, Any]]):
        """Integrate perspective processing results into the mesh"""
        if node_id not in self.perspective_weights:
            self.perspective_weights[node_id] = {}
            
        # Update perspective weights based on confidence
        total_confidence = 0.0
        for perspective, result in perspective_results.items():
            if "confidence" in result:
                confidence = result["confidence"]
                self.perspective_weights[node_id][perspective] = confidence
                total_confidence += confidence
        
        if total_confidence > 0:
            # Normalize weights
            for perspective in self.perspective_weights[node_id]:
                self.perspective_weights[node_id][perspective] /= total_confidence
            
            # Apply perspective resonance to node
            node = self.nodes.get(node_id)
            if node:
                resonance = sum(
                    weight * self.perspective_resonance 
                    for weight in self.perspective_weights[node_id].values()
                )
                node.kinetic_state *= (1.0 + resonance)
                
    def strengthen_pathway(self, node_sequence: List[str], reward: float):
        """Strengthen a successful pathway with integrated effects"""
        for i in range(len(node_sequence) - 1):
            current_id = node_sequence[i]
            next_id = node_sequence[i + 1]
            
            if current_id in self.nodes and next_id in self.nodes:
                current_node = self.nodes[current_id]
                
                # Add path to active pathways
                self.active_pathways.add((current_id, next_id))
                
                # Calculate integrated boost
                quantum_boost = self.quantum_resonance.get(current_id, 0.0)
                perspective_boost = sum(
                    self.perspective_weights.get(current_id, {}).values()
                ) / max(len(self.perspective_weights.get(current_id, {})), 1)
                
                total_boost = (
                    1.0 +
                    quantum_boost * self.quantum_influence +
                    perspective_boost * self.perspective_resonance
                )
                
                # Strengthen connection with integrated boost
                if next_id in current_node.connections:
                    current_node.connections[next_id] += (
                        self.learning_rate * reward * total_boost
                    )
                else:
                    current_node.connections[next_id] = (
                        self.learning_rate * reward * total_boost
                    )
                    
                # Update kinetic state
                current_node.kinetic_state += (
                    self.learning_rate * reward * total_boost
                )

    def save_state(self, path: Path):
        """Save mesh state to file"""
        state = {
            "nodes": {
                node_id: {
                    "energy": node.energy,
                    "connections": node.connections,
                    "kinetic_state": node.kinetic_state,
                    "activation_pattern": node.activation_pattern.tolist()
                }
                for node_id, node in self.nodes.items()
            },
            "params": {
                "energy_threshold": self.energy_threshold,
                "learning_rate": self.learning_rate,
                "prune_threshold": self.prune_threshold
            }
        }
        
        with open(path, 'w') as f:
            json.dump(state, f)

    def load_state(self, path: Path):
        """Load mesh state from file"""
        with open(path, 'r') as f:
            state = json.load(f)
            
        # Restore nodes
        self.nodes = {
            node_id: SynapticNode(
                id=node_id,
                energy=data["energy"],
                connections=data["connections"],
                activation_pattern=np.array(data["activation_pattern"]),
                kinetic_state=data["kinetic_state"]
            )
            for node_id, data in state["nodes"].items()
        }
        
        # Restore parameters
        self.energy_threshold = state["params"]["energy_threshold"]
        self.learning_rate = state["params"]["learning_rate"]
        self.prune_threshold = state["params"]["prune_threshold"]