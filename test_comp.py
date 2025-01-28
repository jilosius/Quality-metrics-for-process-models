import argparse
import time
from io_handler import IOHandler
from process.process import Process
from model_alteration.model_alteration import ModelAlteration
from similarity_metric.compliance_metric import ComplianceMetric

class TestCompliance:
    def __init__(self, file_path1: str, file_path2: str):
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.io_handler = IOHandler()
    
    def execute(self):
        # Load BPMN models
        print("Loading BPMN models...")
        model1_tree = self.io_handler.read_bpmn(self.file_path1)
        model2_tree = self.io_handler.read_bpmn(self.file_path2)
        
        


        # Convert models into process representations
        print("Parsing models...")
        process1 = Process()
        process1.from_bpmn(model1_tree.getroot())
        process1.print_process_state()

        process2 = Process()
        process2.from_bpmn(model2_tree.getroot())
        process2.print_process_state()
        
        
        self.compliance_metric = ComplianceMetric(process1, process2, self.file_path1, self.file_path2)
        

        # Compute compliance similarity
        print("Calculating compliance similarity...")
        start_time = time.time()
        compliance_score = self.compliance_metric.calculate()
        end_time = time.time()
        
        for key,value in compliance_score.items():
            print(f"{key}: {value}")
            
        print(f"Computation Time: {end_time - start_time:.4f} seconds")
        return compliance_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute compliance similarity between two BPMN models.")
    parser.add_argument("file_path1", type=str, help="Path to the first BPMN model file")
    parser.add_argument("file_path2", type=str, help="Path to the second BPMN model file")
    args = parser.parse_args()
    
    test_compliance = TestCompliance(args.file_path1, args.file_path2)
    test_compliance.execute()
