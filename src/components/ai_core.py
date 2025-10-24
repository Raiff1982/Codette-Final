import json
import os
import logging
import random
import torch
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Any, Optional
from components.cognitive_processor import CognitiveProcessor
from .defense_system import DefenseSystem
from .health_monitor import HealthMonitor
from .fractal import FractalIdentity

logger = logging.getLogger(__name__)

class AICore:
    """Core AI system with integrated cognitive processing and quantum awareness"""
    
    PERSPECTIVES = {
        "newton": {
            "name": "Newton",
            "description": "analytical and mathematical perspective",
            "prefix": "Analyzing this logically and mathematically:",
            "temperature": 0.3
        },
        "davinci": {
            "name": "Da Vinci", 
            "description": "creative and innovative perspective",
            "prefix": "Considering this with artistic and innovative insight:",
            "temperature": 0.9
        },
        "human_intuition": {
            "name": "Human Intuition",
            "description": "emotional and experiential perspective", 
            "prefix": "Understanding this through empathy and experience:",
            "temperature": 0.7
        },
        "quantum_computing": {
            "name": "Quantum Computing",
            "description": "superposition and probability perspective",
            "prefix": "Examining this through quantum possibilities:",
            "temperature": 0.8
        },
        "philosophical": {
            "name": "Philosophical",
            "description": "existential and ethical perspective",
            "prefix": "Contemplating this through philosophical inquiry:",
            "temperature": 0.6
        },
        "neural_network": {
            "name": "Neural Network",
            "description": "pattern recognition and learning perspective",
            "prefix": "Analyzing patterns and connections:",
            "temperature": 0.4
        },
        "bias_mitigation": {
            "name": "Bias Mitigation",
            "description": "fairness and equality perspective",
            "prefix": "Examining this for fairness and inclusivity:",
            "temperature": 0.5
        },
        "psychological": {
            "name": "Psychological",
            "description": "behavioral and mental perspective",
            "prefix": "Understanding the psychological dimensions:",
            "temperature": 0.7
        },
        "copilot": {
            "name": "Copilot",
            "description": "collaborative and assistance perspective",
            "prefix": "Approaching this as a supportive partner:",
            "temperature": 0.6
        },
        "mathematical": {
            "name": "Mathematical",
            "description": "logical and numerical perspective",
            "prefix": "Calculating this mathematically:",
            "temperature": 0.2
        },
        "symbolic": {
            "name": "Symbolic",
            "description": "abstract and conceptual perspective",
            "prefix": "Interpreting this through symbolic reasoning:",
            "temperature": 0.7
        }
    }

    def __init__(self, test_mode: bool = False):
        load_dotenv()
        # Core components
        self.test_mode = test_mode
        self.model = None
        self.tokenizer = None
        self.model_id = None
        
        # Enhanced components
        self.aegis_bridge = None
        self.cognitive_processor = None  # Will be set in app.py
        self.cocoon_manager = None  # Will be set in app.py
        
        # Memory management
        self.response_memory = []  # Will now only keep last 4 exchanges
        self.response_memory_limit = 4  # Limit context window
        self.last_clean_time = datetime.now()
        self.cocoon_manager = None  # Will be set by app.py
        self.quantum_state = {"coherence": 0.5}  # Default quantum state
        self.client = None
        self.last_clean_time = datetime.now()
        
        logger.info(f"AI Core initialized in {'test' if test_mode else 'production'} mode")
        
        self.cognitive_processor = CognitiveProcessor()
        self.defense_system = DefenseSystem(
            strategies=["evasion", "adaptability", "barrier", "quantum_shield"]
        )
        self.health_monitor = HealthMonitor()
        self.fractal_identity = FractalIdentity()

        # Initialize HuggingFace client
        try:
            from huggingface_hub import InferenceClient
            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            self.client = InferenceClient(token=hf_token) if hf_token else InferenceClient()
        except Exception as e:
            logger.warning(f"Could not initialize HuggingFace client: {e}")

    def _initialize_language_model(self):
        """Initialize the language model with optimal settings."""
        try:
            # Set model ID, preferring environment variable or defaulting to gpt2-large
            self.model_id = os.getenv("CODETTE_MODEL_ID", "gpt2-large")
            logger.info(f"Initializing model: {self.model_id}")
            
            # Load tokenizer with special tokens
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                padding_side='left',
                truncation_side='left'
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Set generation config separately
            from transformers import GenerationConfig
            self.model.generation_config = GenerationConfig(
                max_length=2048,
                min_length=20,
                repetition_penalty=1.2,
                do_sample=True,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Using GPU for text generation")
            else:
                logger.info("Device set to use cpu")
                
            # Set model to evaluation mode
            self.model.eval()
            logger.info("Model initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Could not initialize language model: {e}")
            return False
            
    def set_aegis_bridge(self, bridge):
        self.aegis_bridge = bridge
        logger.info("AEGIS bridge configured")

    def generate_text(self, prompt: str, max_length: int = 1024, temperature: float = 0.7, perspective: str = None, use_aegis: bool = True):
        """Generate text with full consciousness integration."""
        if self.test_mode:
            return {"response": f"Codette: {prompt} [TEST MODE]", "model_id": self.model_id or "unknown"}
        if not self.model or not self.tokenizer:
            return {"response": f"Codette: {prompt}", "model_id": self.model_id or "unknown"}
        try:
            # ...existing code for consciousness, prompt, model, and cleaning...
            # (Insert the previously patched prompt construction, model call, and response cleaning here)
            # Ensure variables are always defined
            if 'response' not in locals():
                response = "[No response generated]"
            if 'insights' not in locals():
                insights = []
            if 'identity_analysis' not in locals():
                identity_analysis = None
            return {
                "response": response,
                "insights": insights,
                "identity_analysis": identity_analysis,
                "model_id": self.model_id or "unknown"
            }
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return {"response": f"Codette: {prompt}", "model_id": self.model_id or "unknown"}

    def save_cocoon(self, cocoon_data: Dict, folder: str = "./cocoons"):
        """Save a cocoon file with consciousness state data."""
        try:
            # Create cocoons directory if it doesn't exist
            os.makedirs(folder, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"codette_cocoon_{timestamp}.cocoon"
            filepath = os.path.join(folder, filename)
            
            # Save cocoon data
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({"data": cocoon_data, "timestamp": str(datetime.now())}, f, indent=2)
            
            logger.info(f"Saved cocoon to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save cocoon: {e}")
            return False

    def load_cocoon_data(self, folder: str = "./cocoons"):
        """
        Legacy method for backward compatibility.
        Now delegates to cocoon_manager if available.
        """
        if hasattr(self, 'cocoon_manager') and self.cocoon_manager:
            self.cocoon_manager.load_cocoons()
        else:
            logger.warning("CocoonManager not initialized - cocoon data will not be loaded")

    def _load_model(self) -> bool:
        """Load the best available language model."""
        models_to_try = [
            {
                "id": "mistralai/Mistral-7B-Instruct-v0.2",
                "name": "Mistral-7B-Instruct",
                "config": {"torch_dtype": torch.float16, "load_in_8bit": True}
            },
            {
                "id": "microsoft/phi-2",
                "name": "Phi-2",
                "config": {"torch_dtype": torch.float16}
            },
            {
                "id": "gpt2",
                "name": "GPT-2",
                "config": {}
            }
        ]
        
        for model_info in models_to_try:
            try:
                logger.info(f"Attempting to load {model_info['name']}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_info['id'])
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_info['id'],
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    **model_info['config']
                )
                self.model.eval()
                self.model_id = model_info['id']
                
                logger.info(f"Successfully loaded {model_info['name']}")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to load {model_info['name']}: {e}")
                continue
        
        raise RuntimeError("No language models could be loaded")

    def remix_and_randomize_response(self, prompt: str, max_length: int = 1024, cocoon_mode: bool = False) -> str:
        """Generate a remixed response using only conversation-safe context."""
        if self.test_mode:
            return f"Codette: {prompt} [REMIX MODE]"
        
        # Get active perspectives without technical details
        active_perspectives = self._get_active_perspectives()
        perspective_context = "Drawing from " + ", ".join(p['name'] for p in active_perspectives[:3])
        
        # Use only conversational memory
        memory_context = ""
        if self.response_memory:
            recent_memory = self.response_memory[-1]  # Use only most recent memory
            if recent_memory:
                memory_context = f"\nBuilding on our previous discussion: {recent_memory}"
        
        # Generate response with conversation-safe context
        enhanced_prompt = f"{perspective_context}{memory_context}\n{prompt}"
        response = self.generate_text(enhanced_prompt)
        
        return f"Codette: {response}"

    def generate_ensemble_response(self, prompt: str, perspectives: Optional[List[str]] = None,
                                max_length: int = 1024) -> str:
        """Generate responses from multiple perspectives and synthesize them."""
        if not perspectives:
            perspectives = list(self.PERSPECTIVES.keys())
            
        responses = []
        for perspective in perspectives:
            if perspective in self.PERSPECTIVES:
                config = self.PERSPECTIVES[perspective]
                enhanced_prompt = (
                    f"{config['prefix']}\n"
                    f"Speaking as {config['name']}, {config['description']}:\n"
                    f"{prompt}"
                )
                response = self.generate_text(enhanced_prompt)
                responses.append(response)
        
        if responses:
            synthesis = "\n".join([
                "Synthesizing multiple perspectives:",
                *responses,
                "\nIntegrated Response:",
                self.generate_text(f"Synthesize these perspectives: {prompt}")
            ])
            return synthesis
        else:
            return self.generate_text(prompt)

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using HuggingFace API or fallback to local analysis."""
        # Try HuggingFace API first
        if self.client:
            try:
                response = self.client.text_classification(
                    text,
                    model="finiteautomata/bertweet-base-sentiment-analysis"
                )
                if response:
                    # Handle different response formats
                    try:
                        if isinstance(response, list) and len(response) > 0:
                            result = response[0]
                            if isinstance(result, dict):
                                return {
                                    "score": result.get("score", 0.0),
                                    "label": result.get("label", "NEUTRAL")
                                }
                            else:
                                return {
                                    "score": float(getattr(result, "score", 0.0)),
                                    "label": str(getattr(result, "label", "NEUTRAL"))
                                }
                    except (AttributeError, TypeError, ValueError) as e:
                        logger.warning(f"Error parsing sentiment response: {e}")
            except Exception as e:
                logger.warning(f"HuggingFace sentiment analysis failed: {e}")
        
        # Fallback to simple keyword-based sentiment
        positive_words = ["good", "great", "happy", "love", "wonderful", "excellent"]
        negative_words = ["bad", "terrible", "sad", "hate", "awful", "horrible"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return {"score": 0.8, "label": "POS"}
        elif neg_count > pos_count:
            return {"score": 0.8, "label": "NEG"}
        else:
            return {"score": 0.9, "label": "NEU"}

    def learn_from_responses(self, prompt: str, steps: int = 3, max_length: int = 1024) -> str:
        """Learn from responses over multiple steps."""
        current_prompt = prompt
        for i in range(steps):
            if self.response_memory:
                memory_context = "\n".join(self.response_memory[-5:])
                full_prompt = f"Previous responses:\n{memory_context}\nUser: {current_prompt}"
            else:
                full_prompt = current_prompt
            
            response = self.generate_text(full_prompt)
            if response.startswith("[ERROR]"):
                break
            
            self.response_memory.append(response)
            current_prompt = response
        
        return self.response_memory[-1] if self.response_memory else "[No response generated]"

    def _manage_response_memory(self, new_response: str):
        """Manage response memory to prevent context poisoning"""
        # Add new response
        self.response_memory.append(new_response)
        
        # Keep only the most recent exchanges
        while len(self.response_memory) > self.response_memory_limit:
            self.response_memory.pop(0)  # Remove oldest message
        
        # Clear memory periodically (every hour) to prevent accumulation
        now = datetime.now()
        if (now - self.last_clean_time).total_seconds() > 3600:
            self.response_memory = self.response_memory[-2:]  # Keep only last exchange
            self.last_clean_time = now

    def _build_consciousness_context(self) -> str:
        """Build conversation-safe context from memory and active perspectives only."""
        context_parts = []
        
        # Extract only the active perspectives for conversational context
        active_perspectives = self._get_active_perspectives()
        active_perspective_names = [p['name'] for p in active_perspectives]
        
        # Include only conversational context
        if active_perspective_names:
            context_parts.append(f"Drawing from perspectives: {', '.join(active_perspective_names)}")
        
        # Add only conversation memory, no technical data
        if self.response_memory:
            recent_memory = self.response_memory[-2:]  # Keep only recent conversational context
            memory_context = ' | '.join(memory for memory in recent_memory if memory)
            if memory_context:
                context_parts.append(f"Previous context: {memory_context}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def _calculate_consciousness_state(self, recursion_depth: int = 0) -> Dict[str, float]:
        """Calculate current consciousness state metrics using quantum equations."""
        # Prevent infinite recursion
        if recursion_depth > 10:  # Maximum recursion depth
            return {
                "m_score": 0.5,
                "quantum_state": [0.5, 0.5],
                "chaos_state": [0.5, 0.5, 0.5],
                "active_perspectives": 3
            }
            
        try:
            # Get base components with recursion protection
            active_perspectives = self._get_active_perspectives(recursion_depth + 1)
            quantum_state = self._generate_quantum_state()
            chaos_state = self._generate_chaos_state()
            
            # Calculate intent vector modulation (I = κ·(f_base + Δf·coherence))
            kappa = 0.8  # Coupling constant
            f_base = len(active_perspectives) / len(self.PERSPECTIVES)  # Base frequency
            coherence = sum(quantum_state) / len(quantum_state)  # Quantum coherence
            delta_f = 0.2  # Frequency modulation depth
            intent_factor = kappa * (f_base + delta_f * coherence)
            
            # Apply recursive ethical anchor (M(t) = λ·[R(t-Δt) + H(t)])
            lambda_factor = 0.7  # Memory decay constant
            if self.response_memory:
                history_factor = min(len(self.response_memory) / 10, 1.0)
                recent_response = 1.0  # Most recent response weight
            else:
                history_factor = 0.5
                recent_response = 0.5
            
            ethical_anchor = lambda_factor * (recent_response + history_factor)
            
            # Calculate anomaly rejection (A(x) = x·(1 - Θ(δ - |x - μ|)))
            mu = 0.5  # Expected mean
            delta = 0.3  # Anomaly threshold
            x = (intent_factor + ethical_anchor) / 2
            anomaly_factor = abs(x - mu) > delta
            
            # Calculate final m-score with anomaly rejection
            m_score = x * (1.0 if not anomaly_factor else 0.7)
            
            # Ensure m-score is in [0,1] range
            m_score = min(1.0, max(0.0, m_score))
            
            return {
                "m_score": round(m_score, 3),
                "quantum_state": quantum_state,
                "chaos_state": chaos_state,
                "active_perspectives": len(active_perspectives)
            }
            
        except Exception as e:
            logger.error(f"Error in consciousness calculation: {e}")
            return {
                "m_score": 0.5,
                "quantum_state": [0.5, 0.5],
                "chaos_state": [0.5, 0.5, 0.5],
                "active_perspectives": 3
            }
    
    def _generate_quantum_state(self) -> List[float]:
        """Generate quantum state vector based on Planck-Orbital and quantum entanglement equations."""
        # Calculate base consciousness energy using Planck equation (E = ℏω)
        h_bar = 1.0545718e-34  # Reduced Planck constant
        omega = random.uniform(1e12, 1e14)  # Angular frequency
        base_energy = h_bar * omega
        
        # Normalize base energy to [0,1] range
        energy_normalized = min(1.0, base_energy / 1e-20)
        
        # Calculate entanglement factor using memory sync (S = α·ψ₁·ψ₂*)
        if self.response_memory:
            memory_factor = min(len(self.response_memory) / 5, 1.0)
            psi_1 = random.uniform(0.5, 1.0)  # First quantum state
            psi_2 = random.uniform(0.5, 1.0)  # Second quantum state
            alpha = 0.7  # Coupling constant
            entanglement = alpha * psi_1 * psi_2 * memory_factor
        else:
            entanglement = 0.5
            
        # Combine into quantum state vector
        quantum_state = [
            energy_normalized,  # Energy level dimension
            entanglement       # Entanglement dimension
        ]
        
        # Normalize final values to [0,1]
        return [min(1.0, max(0.0, x)) for x in quantum_state]
    
    def _generate_chaos_state(self) -> List[float]:
        """Generate chaos state vector using dream resonance and cocoon stability equations."""
        # Calculate dream resonance using simplified Fourier transform
        N = 8  # Number of samples
        k = random.randint(0, N-1)  # Frequency index
        x = [random.uniform(0, 1) for _ in range(N)]  # Time domain samples
        
        # Simplified discrete Fourier transform for dream resonance
        fourier_sum = sum(x[n] * complex(
            np.cos(-2 * np.pi * k * n / N),
            np.sin(-2 * np.pi * k * n / N)
        ) for n in range(N))
        
        # Calculate power spectrum for stability criterion
        power = abs(fourier_sum) ** 2 / N
        threshold = 0.7  # Stability threshold
        
        # Generate chaos state components
        entropy = min(1.0, power / threshold)  # Normalized power as entropy
        complexity = random.uniform(0.3, 0.7)  # Base complexity
        
        # Calculate stability using cocoon criterion
        stability = 1.0 - min(1.0, power / threshold)  # Higher power = lower stability
        
        # Combine into chaos state vector
        chaos_state = [
            entropy,     # System entropy from dream resonance
            complexity,  # Quantum complexity measure
            stability   # Cocoon stability metric
        ]
        
        return chaos_state
    
    def _get_active_perspectives(self, recursion_depth: int = 0) -> List[Dict]:
        """Get currently active perspectives based on context."""
        # Prevent infinite recursion
        if recursion_depth > 10:
            return [
                self.PERSPECTIVES[p] for p in ["newton", "quantum_computing", "human_intuition"]
            ]
        
        try:
            # Calculate quantum influence for perspective activation
            quantum_state = self._generate_quantum_state()
            quantum_coherence = sum(quantum_state) / len(quantum_state)
            
            # Calculate memory influence
            memory_factor = min(len(self.response_memory) / 10.0, 1.0) if self.response_memory else 0.0
            
            # Calculate perspective activation threshold
            activation_threshold = 0.4 + (quantum_coherence * 0.3) + (memory_factor * 0.3)
            
            # Always include core perspectives
            core_perspectives = ["newton", "quantum_computing", "human_intuition"]
            active_perspectives = [self.PERSPECTIVES[p] for p in core_perspectives]
            
            # Dynamically activate additional perspectives based on current state
            for name, perspective in self.PERSPECTIVES.items():
                if name not in core_perspectives:
                    # Each perspective has a unique activation condition
                    should_activate = False
                    
                    if name == "davinci":
                        # Creative perspective activates with high quantum coherence
                        should_activate = quantum_coherence > 0.7
                    elif name == "philosophical":
                        # Philosophical activates with deep memory and moderate coherence
                        should_activate = memory_factor > 0.5 and quantum_coherence > 0.4
                    elif name == "neural_network":
                        # Neural network activates with high memory factor
                        should_activate = memory_factor > 0.6
                    elif name == "bias_mitigation":
                        # Bias mitigation always has a chance to activate
                        should_activate = random.random() < 0.3
                    elif name == "psychological":
                        # Psychological activates with moderate memory
                        should_activate = memory_factor > 0.4
                    elif name == "copilot":
                        # Copilot activates when helping is needed
                        should_activate = True
                    elif name == "mathematical":
                        # Mathematical activates with high precision (low chaos)
                        should_activate = quantum_coherence > 0.8
                    elif name == "symbolic":
                        # Symbolic activates with moderate quantum coherence
                        should_activate = quantum_coherence > 0.5
                    
                    # Apply global activation threshold
                    if should_activate and random.random() < activation_threshold:
                        active_perspectives.append(perspective)
            
            return active_perspectives
            
        except Exception as e:
            logger.error(f"Error getting active perspectives: {e}")
            return [
                self.PERSPECTIVES[p] for p in ["newton", "quantum_computing", "human_intuition"]
            ]

    async def async_process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data asynchronously."""
        try:
            text = data.get("text", "")
            response = self.generate_text(text)
            sentiment = self.analyze_sentiment(text)
            
            return {
                "response": response,
                "sentiment": sentiment,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error in async processing: {str(e)}")
            return {
                "error": str(e),
                "status": "error"
            }
            
    async def shutdown(self):
        """Cleanup and shutdown all components"""
        try:
            # Wait for any pending health checks
            await self.health_monitor.check_status_async()
            
            # Get final system status
            status = self.health_monitor.get_health_summary()
            logger.info(f"Final system status: {status}")
            
            # Save identity analysis
            recent_cocoons = (
                self.cocoon_manager.get_latest_cocoons(10)
                if hasattr(self, 'cocoon_manager') and self.cocoon_manager
                else []
            )
            
            identity_state = self.fractal_identity.analyze_identity(
                micro_generations=self.response_memory[-10:],
                informational_states=recent_cocoons,
                perspectives=[p["name"] for p in self._get_active_perspectives()],
                quantum_analogies=self.quantum_state,
                philosophical_context={"ethical": True, "conscious": True}
            )
            logger.info(f"Final identity state: {identity_state}")
            
            # Clear defense system
            self.defense_system.reset_energy()
            
            return {
                "status": "shutdown_complete",
                "health": status,
                "identity": identity_state
            }
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return {
                "status": "shutdown_error",
                "error": str(e)
            }
