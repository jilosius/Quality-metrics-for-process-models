from .flowNode import FlowNode

class Flow:
    def __init__(self, flow_id: str, label: str, source: FlowNode, target: FlowNode):
        self.id = flow_id  # Assign the flow ID
        self.label = label
        self.source = source
        self.target = target

    def __repr__(self):
        return f"Flow(id={self.id}, label={self.label}, source={self.source.flowNode_id}, target={self.target.flowNode_id})"
