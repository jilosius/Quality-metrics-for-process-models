
from difflib import SequenceMatcher

from Levenshtein import ratio
import multiprocessing
from multiprocessing import Pool

class TestSim:
    def __init__(self):
        self.metrics = ["F1Score", "NodeStructuralBehavioralMetric"]  # List of available metrics
        self.reference_model = None
        self.altered_model = None

    
    def calculate_syntactic_similarity(self,label1, label2):
        """Calculate syntactic similarity between two labels."""
        # Normalize labels by converting to lowercase and stripping whitespace
        label1 = label1.strip().lower()
        label2 = label2.strip().lower()
        
        # Calculate and return similarity
        return ratio(label1, label2)

if __name__ == "__main__":
    test = TestSim()
    print(test.calculate_syntactic_similarity("Prepare chicken", "Prepare chicker"))
    print(test.calculate_syntactic_similarity("Malek", "Saif"))
    print(test.calculate_syntactic_similarity("Event_1u56gjw", "Gateway_0ve5rnc"))
    
    print(test.calculate_syntactic_similarity("Decide what's for dinner", "i4vUxmJR6V_1"))
    print(test.calculate_syntactic_similarity("Decide what's for dinner", "i4vUxmJR6V_1i4vUxmJR6V_1"))

    # print(multiprocessing.cpu_count())
