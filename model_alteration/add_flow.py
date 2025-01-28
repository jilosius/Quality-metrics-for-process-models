import random
from process.flow import Flow
from process.process import Process
from model_alteration.model_alteration import ModelAlteration


class AddFlow:

    def apply(self, model: Process) -> Process:
        # Ensure there are enough nodes to create a flow
        if len(model.flowNodes) < 2:
            print("Not enough nodes to add a flow.")
            return model

        # Filter potential source nodes to exclude EndEvent (no outgoing flows)
        potential_sources = [
            node for node in model.flowNodes if node.type.lower() != "endevent"
        ]
        if not potential_sources:
            print("No valid source nodes available to add a flow.")
            return model
        source_node = random.choice(potential_sources)

        # Filter potential target nodes to exclude StartEvent (no incoming flows)
        potential_targets = [
            node for node in model.flowNodes if node != source_node and node.type.lower() != "startevent"
        ]
        if not potential_targets:
            print("No valid target nodes available to add a flow.")
            return model
        target_node = random.choice(potential_targets)

        # Generate a unique ID for the new flow
        ModelAlteration.flow_count += 1
        new_flow_id = f"flow_{ModelAlteration.flow_count}"

        # Create the new flow
        new_flow = Flow(
            flow_id=new_flow_id,
            label=f"Flow from {source_node.flowNode_id} to {target_node.flowNode_id}",
            source=source_node,
            target=target_node
        )

        # Add the new flow to the model
        model.flows.append(new_flow)

        print(f"Added new flow: {new_flow.label} ({source_node.flowNode_id} -> {target_node.flowNode_id})")
        return model
