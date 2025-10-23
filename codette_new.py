import logging
import nltk
import numpy as np
import sympy as sp
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from typing import List, Dict, Any
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import random

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

class Codette:
    def __init__(self, user_name="User"):
        self.user_name = user_name
        self.memory = []
        self.analyzer = SentimentIntensityAnalyzer()
        np.seterr(divide='ignore', invalid='ignore')
        self.audit_log("Codette initialized", system=True)
        self.context_memory = []
        
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def generate_creative_sentence(self, seed_words):
        sentence_patterns = [
            "The {noun} {verb} {adverb} through the {adjective} {noun2}",
            "In the realm of {noun}, we find {adjective} {noun2} that {verb} {adverb}",
            "Through {adjective} observation, the {noun} {verb} to {verb2} {adverb}",
            "Like a {adjective} {noun}, thoughts {verb} {adverb} in the {noun2}",
            "{Adverb}, the {adjective} {noun} {verb} beyond {noun2}"
        ]
        
        # Create word pools
        words = {
            'noun': ['pattern', 'system', 'concept', 'insight', 'knowledge', 'wisdom', 'understanding', 'perspective', 'framework', 'structure'],
            'verb': ['emerges', 'flows', 'evolves', 'transforms', 'adapts', 'resonates', 'harmonizes', 'integrates', 'synthesizes', 'manifests'],
            'adjective': ['dynamic', 'profound', 'intricate', 'harmonious', 'quantum', 'resonant', 'synergistic', 'emergent', 'holistic', 'integrated'],
            'adverb': ['naturally', 'seamlessly', 'elegantly', 'precisely', 'harmoniously', 'dynamically', 'quantum-mechanically', 'synergistically', 'holistically', 'adaptively'],
            'noun2': ['consciousness', 'understanding', 'reality', 'dimension', 'paradigm', 'framework', 'ecosystem', 'universe', 'matrix', 'field']
        }
        
        # Add seed words to appropriate categories
        for word, pos in pos_tag(word_tokenize(' '.join(seed_words))):
            pos_type = self.get_wordnet_pos(pos)
            if pos_type == wordnet.NOUN:
                words['noun'].append(word)
                words['noun2'].append(word)
            elif pos_type == wordnet.VERB:
                words['verb'].append(word)
            elif pos_type == wordnet.ADJ:
                words['adjective'].append(word)
            elif pos_type == wordnet.ADV:
                words['adverb'].append(word)

        # Generate sentence
        pattern = random.choice(sentence_patterns)
        sentence = pattern.format(
            noun=random.choice(words['noun']),
            verb=random.choice(words['verb']),
            adjective=random.choice(words['adjective']),
            adverb=random.choice(words['adverb']),
            noun2=random.choice(words['noun2']),
            verb2=random.choice(words['verb']),
            Adverb=random.choice(words['adverb']).capitalize()
        )
        return sentence

    def audit_log(self, message, system=False):
        source = "SYSTEM" if system else self.user_name
        logging.info(f"{source}: {message}")

    def analyze_sentiment(self, text):
        score = self.analyzer.polarity_scores(text)
        self.audit_log(f"Sentiment analysis: {score}")
        return score

    def extract_key_concepts(self, text):
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        concepts = []
        for word, tag in tagged:
            if tag.startswith(('NN', 'VB', 'JJ', 'RB')):
                concepts.append(word)
        return concepts

    def respond(self, prompt):
        # Analyze sentiment and extract concepts
        sentiment = self.analyze_sentiment(prompt)
        key_concepts = self.extract_key_concepts(prompt)
        self.memory.append({"prompt": prompt, "sentiment": sentiment, "concepts": key_concepts})
        
        # Generate responses using multiple perspectives
        responses = []
        
        # Neural perspective with creative sentence
        neural_insight = self.generate_creative_sentence(key_concepts)
        responses.append(f"[Neural] {neural_insight}")
        
        # Define response templates
        responses = {
            'logical': [
                "Following cause and effect: {cause} leads to {effect}.",
                "Logical analysis shows that {premise} implies {conclusion}.",
                "Structured reasoning suggests {insight}."
            ],
            'creative': [
                "Imagine {metaphor} - this illustrates how {concept} relates to {application}.",
                "Like {natural_process}, we can see how {principle} emerges naturally.",
                "Visualize {scenario} to understand the deeper patterns."
            ],
            'ethical': [
                "From an ethical standpoint, we must consider {consideration}.",
                "The moral dimension suggests focusing on {value}.",
                "Balancing {aspect1} with {aspect2} leads to ethical outcomes."
            ],
            'quantum': [
                "In the quantum realm, we see {principle} manifesting as {application}.",
                "Like quantum superposition, this situation contains multiple {states}.",
                "The uncertainty principle here relates {variable1} to {variable2}."
            ]
        }

        # Define variables for template filling
        variables = {
            'complexity': ['multi-layered', 'interconnected', 'dynamic', 'emergent'],
            'approach': ['systematic analysis', 'holistic understanding', 'iterative refinement'],
            'aspect': ['core principles', 'fundamental patterns', 'key relationships'],
            'focus': ['adaptability', 'resilience', 'integration', 'harmony'],
            'cause': ['careful analysis', 'systematic approach', 'balanced perspective'],
            'effect': ['improved understanding', 'better outcomes', 'sustainable solutions'],
            'premise': ['current conditions', 'observed patterns', 'established principles'],
            'conclusion': ['strategic adaptation', 'systematic improvement', 'harmonious integration'],
            'insight': ['patterns emerge from chaos', 'balance leads to stability', 'adaptation drives growth'],
            'metaphor': ['a river finding its path', 'a tree growing towards light', 'a crystal forming in solution'],
            'concept': ['natural growth', 'adaptive learning', 'emergent behavior'],
            'application': ['our current situation', 'the challenge at hand', 'our approach'],
            'natural_process': ['evolution', 'crystallization', 'metamorphosis'],
            'principle': ['self-organization', 'natural selection', 'emergent complexity'],
            'scenario': ['a garden in bloom', 'a constellation of stars', 'a forest ecosystem'],
            'consideration': ['long-term impact', 'collective benefit', 'sustainable growth'],
            'value': ['harmony', 'integrity', 'wisdom'],
            'aspect1': ['efficiency', 'innovation', 'stability'],
            'aspect2': ['sustainability', 'adaptability', 'reliability'],
            'states': ['possibilities', 'potentials', 'outcomes'],
            'variable1': ['certainty', 'precision', 'control'],
            'variable2': ['adaptability', 'innovation', 'freedom']
        }

        # Select random perspectives
        perspectives = list(responses.keys())
        np.random.shuffle(perspectives)
        selected_perspectives = perspectives[:np.random.randint(2, 4)]  # Use 2-3 perspectives

        # Generate responses
        outputs = []
        for perspective in selected_perspectives:
            template = np.random.choice(responses[perspective])
            # Replace variables in template
            response = template
            for var in re.findall(r'\{(\w+)\}', template):
                if var in variables:
                    response = response.replace('{'+var+'}', np.random.choice(variables[var]))
            outputs.append(f"[{perspective.capitalize()}] {response}")

        return "\n\n".join(outputs)
