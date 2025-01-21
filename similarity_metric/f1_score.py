from similarity_metric.similarity_metric import SimilarityMetric
from process.process import Process
from difflib import SequenceMatcher
import spacy
import numpy as np

from similarity_metric.similarity_metric import SimilarityMetric

class F1Score(SimilarityMetric):

    def __init__(self, reference_model, altered_model, label_similarity_threshold=0.8):
        super().__init__(reference_model, altered_model)
        self.label_similarity_threshold = label_similarity_threshold

    def calculate_similarity(self, label1, label2, type1, type2):
        """Calculate syntactic/semantic similarity between labels."""
        # print(f"Calculating similarity between labels: '{label1}' and '{label2}'")

        type_similarity = self.calculate_type_similarity(type1, type2)
        # print(f"  Type similarity: {type_similarity}")

        if type_similarity == 1:
            syntactic_score = self.calculate_syntactic_similarity(label1, label2)
            semantic_score = self.calculate_semantic_similarity(label1, label2)
            # print(f"  Syntactic similarity: {syntactic_score}\n  Semantic similarity: {semantic_score}")
            
            return max(syntactic_score, semantic_score)
        
        elif type_similarity == 0:
            # print("Flow Nodes not of same type. Moving on..")
            # print("-----------")
            return 0.0

        else:
            return 0.0

    def match_nodes(self, reference_flowNodes, altered_flowNodes):
        """Match nodes based on label similarity."""
        matches = []
        
        for ref_node in reference_flowNodes:
            best_match = None
            best_score = 0
            # print("")
            # print(f"\nReference node: {ref_node.flowNode_id} ({ref_node.type})")
            for alt_node in altered_flowNodes:
                # print(f"  Altered node: {alt_node.flowNode_id} ({alt_node.type})")
                similarity = self.calculate_similarity(ref_node.label, alt_node.label, ref_node.type, alt_node.type)
                # print(f"Label Similarity: {similarity}")
                # print("-----------")
                if similarity > self.label_similarity_threshold and similarity > best_score:
                    best_match = alt_node
                    best_score = similarity
            if best_match:
                # print(f"  Best match for node '{ref_node.label}': '{best_match.label}' with score {best_score}")
                matches.append((ref_node, best_match))
            # else:
                # print(f"  No match with same type & similarity > {self.label_similarity_threshold} found for node '{ref_node.label}'")

            # print(matches)
        return matches

    def match_flows(self, reference_flows, altered_flows):
        """Match flows based on source and target node similarity."""
        matches = []
        # print("\n========= Matching flows between reference and altered models =========")
        for ref_flow in reference_flows:
            for alt_flow in altered_flows:
                source_match = self.match_nodes([ref_flow.source], [alt_flow.source])
                target_match = self.match_nodes([ref_flow.target], [alt_flow.target])
                if source_match and target_match:
                    # print(f"Matched flow: {ref_flow.source.flowNode_id} -> {ref_flow.target.flowNode_id} with {alt_flow.source.flowNode_id} -> {alt_flow.target.flowNode_id}")
                    matches.append((ref_flow, alt_flow))
                    break
        return matches

    def calculate(self):
        # Get flow nodes and flows
        reference_flowNodes = self.reference_model.flowNodes
        altered_flowNodes = self.altered_model.flowNodes
        reference_flows = self.reference_model.flows
        altered_flows = self.altered_model.flows
        
        
        print("Calculating F1 Score...")

        print("\nTP (True Positives): The number of nodes in the reference model that were successfully matched with a node in the altered model based on the similarity criteria.")
        print("FP (False Positives): The number of nodes in the altered model that were incorrectly matched with a node in the reference model when they should not have been matched.")
        print("FN (False Negatives): The number of nodes in the reference model that did not have a match in the altered model, even though a match should exist.")

        print(f"\nReference model has {len(reference_flowNodes)} nodes and {len(reference_flows)} flows.")
        print(f"\nAltered model has {len(altered_flowNodes)} nodes and {len(altered_flows)} flows.")

        # Match flow nodes
        print("\n========= Matching nodes between reference and altered models =========")
        matched_nodes = self.match_nodes(reference_flowNodes, altered_flowNodes)
        tp_nodes = len(matched_nodes)
        fp_nodes = len(altered_flowNodes) - tp_nodes
        fn_nodes = len(reference_flowNodes) - tp_nodes
        print(f"\nMatched nodes: TP={tp_nodes}, FP={fp_nodes}, FN={fn_nodes}")
        

        # Match flows
        print("--------")
        matched_flows = self.match_flows(reference_flows, altered_flows)
        tp_flows = len(matched_flows)
        fp_flows = len(altered_flows) - tp_flows
        fn_flows = len(reference_flows) - tp_flows
        print(f"\nMatched flows: TP={tp_flows}, FP={fp_flows}, FN={fn_flows}")

        # Calculate Precision, Recall, F1 Score
        tp = tp_nodes + tp_flows
        fp = fp_nodes + fp_flows
        fn = fn_nodes + fn_flows

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nPrecision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
        print("\nPrecision: The proportion of correctly matched nodes and flows (TP) out of all predicted matches (TP + FP).")
        print("Recall: The proportion of correctly matched nodes and flows (TP) out of all actual matches (TP + FN).\n")

        return {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score
        }