import asyncio
from knowledge_base.grounding_truth import GroundingTruth

def test_grounding_truth():
    # Initialize the grounding truth system
    truth = GroundingTruth()
    
    print("=== Testing Grounding Truth System ===\n")
    
    # Test 1: Identity verification
    print("Test 1: Identity Claims")
    statements = [
        "My name is Codette",
        "I am a human",
        "I can help with programming",
        "I can fly to the moon"
    ]
    
    for statement in statements:
        result = truth.verify_statement(statement)
        print(f"\nStatement: {statement}")
        print(f"Verified: {result['verified']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        
    # Test 2: Programming knowledge
    print("\nTest 2: Programming Knowledge")
    statements = [
        "Python is a high-level programming language",
        "JavaScript is a compiled language",
        "Python supports object-oriented programming",
        "Python runs on a quantum computer"
    ]
    
    for statement in statements:
        result = truth.verify_statement(statement)
        print(f"\nStatement: {statement}")
        print(f"Verified: {result['verified']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        
    # Test 3: Response verification
    print("\nTest 3: Response Verification")
    query = "What can you do?"
    response = "I can help with programming tasks and technical research"
    verified_response = truth.get_verified_response(query, response)
    print(f"Original Response: {response}")
    print(f"Verified Response: {verified_response}")
    
    # Test 4: Adding new fact
    print("\nTest 4: Adding New Fact")
    new_fact = {
        "rust": {
            "type": "Programming language",
            "paradigm": ["Systems programming", "Concurrent"],
            "level": "Low-level",
            "typing": "Static",
            "verified": True
        }
    }
    
    success = truth.add_verified_fact(new_fact, "core_knowledge.programming_languages")
    print(f"Added new fact successfully: {success}")
    
    # Verify the new fact
    statement = "Rust is a systems programming language"
    result = truth.verify_statement(statement)
    print(f"\nVerifying new knowledge:")
    print(f"Statement: {statement}")
    print(f"Verified: {result['verified']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")

if __name__ == "__main__":
    test_grounding_truth()