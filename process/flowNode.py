class FlowNode:
    def __init__(self, flowNode_id: str, label: str, flowNode_type: str, lane_id: str):
        self.flowNode_id = flowNode_id
        self.label = label
        self.type = flowNode_type
        self.lane_id = lane_id

    def __repr__(self):
        return f"flowNode(id={self.flowNode_id}, label={self.label}, type={self.type}, lane_id={self.lane_id})"
