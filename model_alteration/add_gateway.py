from .model_alteration import ModelAlteration
from process.process import Process
from process.flowNode import FlowNode
from process.flow import Flow


import random

GATEWAY_TYPE_MAPPING = {
    "XOR": "exclusiveGateway",
    "AND": "parallelGateway",
    "OR": "inclusiveGateway"
}

class AddGateway:
    def __init__(self):
        """
        Initialize the AddGateway class.
        No parameters needed as gateway type and connections are chosen randomly.
        """
        self.gateway_type = random.choice(list(GATEWAY_TYPE_MAPPING.values()))

    def apply(self, model: Process) -> Process:
        # Step 1: Randomly assign a lane ID
        if model.lanes:  # Ensure there are lanes available
            lane_id = random.choice(model.lanes).lane_id
        else:
            lane_id = None  # Assign None if no lanes are defined

        # Step 2: Create a new gateway node
        ModelAlteration.flowNode_count += 1
        gateway_id = f"gateway_{ModelAlteration.flowNode_count}"
        gateway_label = f"{self.gateway_type} Gateway {ModelAlteration.flowNode_count}"
        gateway_node = FlowNode(
            flowNode_id=gateway_id,
            label=gateway_label,
            flowNode_type=self.gateway_type,
            lane_id=lane_id  # Assign the randomly selected lane ID
        )
        model.flowNodes.append(gateway_node)

        # Step 3: Randomly select source and target nodes
        if len(model.flowNodes) < 2:
            print("Not enough nodes to add a gateway.")
            return model

        source_node = random.choice(model.flowNodes)
        target_node = random.choice([node for node in model.flowNodes if node != source_node])

        # Step 4: Create new flows
        ModelAlteration.flow_count += 1
        flow_to_gateway = Flow(
            flow_id=f"flow_{ModelAlteration.flow_count}",
            label=f"Flow to {gateway_label}",
            source=source_node,
            target=gateway_node
        )

        ModelAlteration.flow_count += 1
        flow_from_gateway = Flow(
            flow_id=f"flow_{ModelAlteration.flow_count}",
            label=f"Flow from {gateway_label}",
            source=gateway_node,
            target=target_node
        )

        # Add the new flows to the model
        model.flows.extend([flow_to_gateway, flow_from_gateway])

        # Debugging info
        if lane_id:
            print(f"Added {gateway_label} ({self.gateway_type}) in lane {lane_id} between {source_node.flowNode_id} and {target_node.flowNode_id}")
        else:
            print(f"Added {gateway_label} ({self.gateway_type}) with no lane between {source_node.flowNode_id} and {target_node.flowNode_id}")

        print(f"New flows: {flow_to_gateway.label}, {flow_from_gateway.label}")

        return model

