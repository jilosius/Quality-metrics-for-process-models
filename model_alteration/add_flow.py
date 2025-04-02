import random
from process.flow import Flow
from process.process import Process
from model_alteration.model_alteration import ModelAlteration


class AddFlow:

    def apply(self, model: Process) -> Process:
        if len(model.flowNodes) < 2:
            print("Not enough nodes to add a flow.")
            return model

        potential_sources = [
            node for node in model.flowNodes if node.type.lower() != "endevent"
        ]
        if not potential_sources:
            print("No valid source nodes available to add a flow.")
            return model
        source_node = random.choice(potential_sources)

        potential_targets = [
            node for node in model.flowNodes if node != source_node and node.type.lower() != "startevent"
        ]
        if not potential_targets:
            print("No valid target nodes available to add a flow.")
            return model
        target_node = random.choice(potential_targets)

        ModelAlteration.flow_count += 1
        new_flow_id = f"flow_{ModelAlteration.flow_count}"

        new_flow = Flow(
            flow_id=new_flow_id,
            label=f"Flow from {source_node.flowNode_id} to {target_node.flowNode_id}",
            source=source_node,
            target=target_node
        )

        model.flows.append(new_flow)

        print(f"Added new flow: {new_flow.label} ({source_node.flowNode_id} -> {target_node.flowNode_id})")
        return model
