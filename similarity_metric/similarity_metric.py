from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from Levenshtein import ratio
import spacy

class SimilarityMetric(ABC):
    nlp = spacy.load("en_core_web_md")  # Shared spaCy model

    def __init__(self, reference_model, altered_model):
        self.reference_model = reference_model
        self.altered_model = altered_model

    @staticmethod
    def calculate_syntactic_similarity(label1, label2):
        """Calculate syntactic similarity between two labels."""
        # Normalize labels by converting to lowercase and stripping whitespace
        label1 = label1.strip().lower()
        label2 = label2.strip().lower()
        
        # Calculate and return similarity
        return ratio(label1, label2)


    @classmethod
    def calculate_semantic_similarity(cls, label1, label2):
        """Calculate semantic similarity between two labels using spaCy."""
        if not label1 or not label2:
            return 0.0
        doc1 = cls.nlp(label1)
        doc2 = cls.nlp(label2)
        return doc1.similarity(doc2)

    @staticmethod
    def calculate_type_similarity(type1, type2):
        """Calculate type similarity between two node types."""
        return 1.0 if type1 == type2 else 0.0

    @abstractmethod
    def calculate(self):
        pass

    @staticmethod
    def get_metric(metric_name, reference_model, altered_model, file_path=None, output_path=None):
        """
        Factory method to initialize metrics dynamically based on the metric name.

        Args:
            metric_name (str): Name of the metric to initialize.
            reference_model (Process): Reference model object.
            altered_model (Process): Altered model object.
            file_path (str, optional): Path to the reference BPMN file (used by ComplianceMetric).
            output_path (str, optional): Path to the altered BPMN file (used by ComplianceMetric).

        Returns:
            SimilarityMetric: An instance of the requested metric.
        """
        # Lazy import to avoid circular dependencies
        if metric_name == "NodeStructuralBehavioralMetric":
            from similarity_metric.node_structural_metric import NodeStructuralBehavioralMetric
            return NodeStructuralBehavioralMetric(reference_model, altered_model)
        elif metric_name == "F1Score":
            from similarity_metric.f1_score import F1Score
            return F1Score(reference_model, altered_model)
        elif metric_name == "ComplianceMetric":
            from similarity_metric.compliance_metric import ComplianceMetric
            return ComplianceMetric(
                reference_model=reference_model,
                altered_model=altered_model,
                file_path=file_path,
                output_path=output_path
            )
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
