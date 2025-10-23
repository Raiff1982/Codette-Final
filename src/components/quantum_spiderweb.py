"""
Quantum spiderweb implementation for advanced cognition.
"""

import networkx as nx
import numpy as np
from typing import Dict, Any, List, Optional
import random
import logging

logger = logging.getLogger(__name__)

class QuantumSpiderweb:
    """
    Simulates a cognitive spiderweb architecture with dimensions:
    Ψ (thought), τ (time), χ (speed), Φ (emotion), λ (space)
    """
    
    def __init__(self, node_count: int = 128):
        self.graph = nx.Graph()
        self.dimensions = ['Ψ', 'τ', 'χ', 'Φ', 'λ']
        self._init_nodes(node_count)
        self.entangled_states = {}
        self.activation_threshold = 0.3
        
    def _init_nodes(self, count: int):
        """Initialize quantum nodes with multi-dimensional states"""
        for i in range(count):
            node_id = f"QNode_{i}"
            state = self._generate_state()
            self.graph.add_node(node_id, state=state)
            if i > 0:
                # Create connections with probability decay
                connection_count = min(3, i)  # Maximum 3 connections per node
                potential_connections = [f"QNode_{j}" for j in range(i)]
                selected_connections = random.sample(potential_connections, connection_count)
                for connection in selected_connections:
                    weight = random.uniform(0.1, 1.0)
                    self.graph.add_edge(node_id, connection, weight=weight)
                    
    def _generate_state(self) -> Dict[str, float]:
        """Generate quantum state vector for all dimensions"""
        return {dim: random.uniform(0, 1) for dim in self.dimensions}
        
    def propagate_thought(self, origin_id: str, depth: int = 3) -> List[Dict[str, Any]]:
        """
        Propagate thought activation through the quantum web
        
        Args:
            origin_id: Starting node ID
            depth: Propagation depth
            
        Returns:
            List of activated nodes and their states
        """
        if origin_id not in self.graph:
            return []
            
        activated = {origin_id: 1.0}  # Start with origin fully activated
        queue = [(origin_id, 0)]  # (node_id, current_depth)
        results = []
        
        while queue:
            current_id, current_depth = queue.pop(0)
            if current_depth >= depth:
                continue
                
            # Get current node's state
            current_state = self.graph.nodes[current_id]["state"]
            results.append({
                "node_id": current_id,
                "state": current_state,
                "activation": activated[current_id]
            })
            
            # Propagate to neighbors
            for neighbor in self.graph.neighbors(current_id):
                if neighbor not in activated:
                    edge_weight = self.graph[current_id][neighbor]["weight"]
                    activation = activated[current_id] * edge_weight * 0.8  # Decay factor
                    
                    if activation > self.activation_threshold:
                        activated[neighbor] = activation
                        queue.append((neighbor, current_depth + 1))
                        
        return results
        
    def detect_tension(self, node_id: str) -> Optional[Dict[str, float]]:
        """
        Detect quantum tension/instability in node
        
        Args:
            node_id: Node to check
            
        Returns:
            Tension metrics if unstable, None if stable
        """
        if node_id not in self.graph:
            return None
            
        node_state = self.graph.nodes[node_id]["state"]
        neighbors = list(self.graph.neighbors(node_id))
        
        if not neighbors:
            return None
            
        # Calculate tension metrics
        neighbor_states = [self.graph.nodes[n]["state"] for n in neighbors]
        tension_metrics = {}
        
        for dim in self.dimensions:
            # Calculate variance between node and neighbors
            values = [node_state[dim]] + [s[dim] for s in neighbor_states]
            tension_metrics[dim] = np.var(values)
            
        # Consider node unstable if any dimension has high variance
        if any(tension > 0.3 for tension in tension_metrics.values()):
            return tension_metrics
            
        return None
        
    def collapse_node(self, node_id: str) -> Dict[str, Any]:
        """
        Collapse node's quantum state to definite values
        
        Args:
            node_id: Node to collapse
            
        Returns:
            New definite state
        """
        if node_id not in self.graph:
            return {}
            
        # Get current state
        current_state = self.graph.nodes[node_id]["state"]
        
        # Collapse each dimension to either 0 or 1 based on probability
        collapsed_state = {}
        for dim in self.dimensions:
            prob = current_state[dim]
            collapsed_state[dim] = 1 if random.random() < prob else 0
            
        # Update node state
        self.graph.nodes[node_id]["state"] = collapsed_state
        
        return collapsed_state
        
    def entangle_nodes(self, node1: str, node2: str) -> bool:
        """
        Create quantum entanglement between nodes
        
        Args:
            node1: First node ID
            node2: Second node ID
            
        Returns:
            Success status
        """
        if node1 not in self.graph or node2 not in self.graph:
            return False
            
        # Create entangled state
        entangled_id = f"E_{node1}_{node2}"
        self.entangled_states[entangled_id] = {
            "nodes": [node1, node2],
            "state": self._generate_state()
        }
        
        # Add high-weight connection
        self.graph.add_edge(node1, node2, weight=1.0, entangled=True)
        return True
        
    def get_node_state(self, node_id: str) -> Optional[Dict[str, float]]:
        """Get current state of a node"""
        if node_id in self.graph:
            return self.graph.nodes[node_id]["state"]
        return None
        
    def update_node_state(self, node_id: str, new_state: Dict[str, float]) -> bool:
        """Update node's quantum state"""
        if node_id in self.graph:
            # Validate state dimensions
            if all(dim in new_state for dim in self.dimensions):
                self.graph.nodes[node_id]["state"] = new_state
                return True
        return False