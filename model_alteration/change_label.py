from process.process import Process

import random
import time

class ChangeLabel:
    label_counter = 0  

    def __init__(self, node_id=None):
        """
        Optionally specify the node_id to update.
        If None, a random node will be selected.
        """
        self.node_id = node_id

    @staticmethod
    def generate_random_label():
        timestamp = int(time.time())  # Time in seconds
        ChangeLabel.label_counter += 1
        return f"Task_{timestamp}_{ChangeLabel.label_counter}"

    def apply(self, model: Process) -> Process:
        # Ensure there are FlowNodes available
        if not model.flowNodes:
            print("No FlowNodes available to update.")
            return model

        # Find the target node to update
        if self.node_id:
            target_node = None  # Ensure the variable is initialized
            for node in model.flowNodes:
                if node.flowNode_id == self.node_id:
                    target_node = node
                    break
            if not target_node:
                print(f"FlowNode with ID {self.node_id} not found. No changes made.")
                return model
        else:
            target_node = random.choice(model.flowNodes)

        # Change the label of the selected node
        original_label = target_node.label
        new_label = ChangeLabel.generate_random_label()
        target_node.label = new_label

        print(f"Changing label of node {target_node.flowNode_id} from '{original_label}' to '{new_label}'")

        return model
