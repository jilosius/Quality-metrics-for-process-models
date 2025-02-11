import xml.etree.ElementTree as ET
from model_alteration.add_gateway import AddGateway
from model_alteration.remove_flow import RemoveFlow
from model_alteration.remove_activity import RemoveActivity
from model_alteration.remove_gateway import RemoveGateway
from process.process import Process
from model_alteration.model_alteration import ModelAlteration
from model_alteration.add_flowNode import AddFlowNode  
from model_alteration.add_flow import AddFlow
from model_alteration.change_label import ChangeLabel

from io_handler import IOHandler
import os


def print_process_state(model: Process, message: str):
    print(f"\n{message}")
    print("FlowNodes:")
    for flowNode in model.flowNodes:
        print(flowNode)
    print("\nFlows:")
    for flow in model.flows:
        print(flow)
    
    print("\n")

def test_process():
    # Define the file path
    filepath = "task_reference5.bpmn"
    
    # Parse the BPMN file
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Create and populate the Process object
    reference_model = Process()
    reference_model.from_bpmn(root)

    print_process_state(reference_model, "Original Process:")

    # Initialize ModelAlteration with the reference model
    alteration = ModelAlteration(reference_model)

    # Add nodes
    add_node = AddFlowNode()
    num_nodes_to_add = 1  
    for i in range(num_nodes_to_add):
        print(f"Adding FlowNode {i + 1}...")
        alteration.execute(add_node)
        print_process_state(alteration.altered_model, f"Process after adding FlowNode {i + 1}:")

    # Add flow
    add_flow = AddFlow()
    alteration.execute(add_flow)
    print_process_state(alteration.altered_model, "Process after adding Flow:")

    # Remove node
    remove_node = RemoveActivity()
    alteration.execute(remove_node)
    print_process_state(alteration.altered_model, "Process after removing FlowNode:")

    # Remove flow
    remove_flow = RemoveFlow()
    alteration.execute(remove_flow)
    print_process_state(alteration.altered_model, "Process after removing Flow:")

    # Add gateway
    add_gateway = AddGateway()
    alteration.execute(add_gateway)
    print_process_state(alteration.altered_model, "Process after adding Gateway:")

    # Remove gateway
    remove_gateway = RemoveGateway()
    alteration.execute(remove_gateway)
    print_process_state(alteration.altered_model, "Process after removing Gateway:")

    # Change label
    change_label = ChangeLabel()
    alteration.execute(change_label)
    print_process_state(alteration.altered_model, "Process after changing label:")

def test_io_handler():
    # Test read_bpmn
    input_path = "task_reference2.bpmn"
    output_path = "output_test.bpmn"

    # Read the BPMN file
    bpmn_tree = IOHandler.read_bpmn(input_path)
    print("Read BPMN file successfully.")

    # Write the BPMN file
    IOHandler.write_bpmn(bpmn_tree, output_path)
    print("Wrote BPMN file successfully.")

    # Check if the output file exists
    assert os.path.exists(output_path), "Output file was not created."
    



if __name__ == "__main__":
    test_io_handler()
