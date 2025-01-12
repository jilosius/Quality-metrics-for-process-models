
from difflib import SequenceMatcher

class TestSim:
    def __init__(self):
        self.metrics = ["F1Score", "NodeStructuralBehavioralMetric"]  # List of available metrics
        self.reference_model = None
        self.altered_model = None

    
    def calculate_syntactic_similarity(self,label1, label2):
        """Calculate syntactic similarity between two labels using SequenceMatcher."""
        return SequenceMatcher(None, label1, label2).ratio()

if __name__ == "__main__":
    test = TestSim()
    print(test.calculate_syntactic_similarity("Hungry", "Meal?"))