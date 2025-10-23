import logging
import nltk
import numpy as np
import sympy as sp
import pymc as pm
import arviz as az
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from typing import List, Dict, Any

nltk.download('punkt', quiet=True)

class Codette:
    def __init__(self, user_name="User"):
        self.user_name = user_name
        self.memory = []
        self.analyzer = SentimentIntensityAnalyzer()
        self._qlp_cache = {}
        
        # Configure PyMC settings for stability
        self.mcmc_settings = {
            'chains': 4,
            'tune': 1000,
            'draws': 1000,
            'target_accept': 0.95,
            'return_inferencedata': True
        }
        
        # Set numpy error handling
        np.seterr(divide='ignore', invalid='ignore')
        
        self.audit_log("Codette initialized", system=True)

    def audit_log(self, message, system=False):
        source = "SYSTEM" if system else self.user_name
        logging.info(f"{source}: {message}")

    def analyze_sentiment(self, text):
        score = self.analyzer.polarity_scores(text)
        self.audit_log(f"Sentiment analysis: {score}")
        return score

    def respond(self, prompt):
        sentiment = self.analyze_sentiment(prompt)
        self.memory.append({"prompt": prompt, "sentiment": sentiment})
        
        # Randomize module order and selection for variety
        modules = [
            self.neuralNetworkPerspective,
            self.newtonianLogic,
            self.daVinciSynthesis,
            self.resilientKindness,
            self.quantumLogicPerspective,
            self.philosophicalInquiry,
            self.copilotAgent,
            self.mathematicalRigor,
            self.symbolicReasoning
        ]
        np.random.shuffle(modules)
        
        # Randomly select 4-7 perspectives to use
        num_perspectives = np.random.randint(4, 8)
        selected_modules = modules[:num_perspectives]

        responses = []
        for module in selected_modules:
            try:
                result = module(prompt)
                responses.append(result)
            except Exception as e:
                if np.random.random() > 0.7:  # 30% chance to show errors
                    responses.append(f"[Error] {module.__name__} failed: {e}")

        self.audit_log(f"Perspectives used: {[m.__name__ for m in selected_modules]}")
        return "\n\n".join(responses)
