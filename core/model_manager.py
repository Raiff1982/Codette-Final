import os
import json
import logging
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        """Initialize the model manager."""
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.load_model()
        
    def load_model(self, model_name: Optional[str] = None) -> bool:
        """
        Load the language model, trying different models in order of preference.
        
        Args:
            model_name: Optional specific model to load
            
        Returns:
            bool: True if any model was loaded successfully
        """
        models_to_try = [
            model_name
        ] if model_name else [
            "mistralai/Mistral-7B-Instruct-v0.2",  # Best balance of capability/size
            "microsoft/phi-2",                      # Fallback
            "gpt2"                                  # Last resort
        ]
        
        for model_id in models_to_try:
            try:
                logger.info(f"Loading {model_id}")
                self.current_tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.current_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,  # Use half precision
                    load_in_8bit=True
                )
                self.current_model_name = model_id
                self.current_model.eval()
                logger.info(f"Successfully loaded {model_id}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load {model_id}: {e}")
                continue
        
        return False
                torch_dtype=getattr(torch, self.config.get('torch_dtype', 'float32'))
            )
            
            self.current_model.eval()
            self.current_model_name = model_name
            
            logger.info(f"Successfully loaded model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
            
    def get_current_model(self) -> tuple:
        """Get currently loaded model and tokenizer."""
        return self.current_model, self.current_tokenizer
        
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.current_model is not None and self.current_tokenizer is not None