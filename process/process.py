from .flowNode import FlowNode
from .flow import Flow
from .lane import Lane
from .data_obj import DataObj
import xml.etree.ElementTree as ET
from .pool import Pool
import uuid
import copy

class Process:
    def __init__(self):
        self.flowNodes = []  
        self.flows = []       
        self.lanes = []       
        self.data_objects = []  
        self.pools = []  

    def from_bpmn(self, root: ET.Element):
        """
        Parses a BPMN XML tree and populates the Process object with Activities, Flows, Lanes, Data Objects, and Pools.
        :param root: The root of the BPMN ElementTree.
        """
        # Parse pools
        self.pools = []
        for pool in root.findall(".//{http://www.omg.org/spec/BPMN/20100524/MODEL}participant"):
            pool_id = pool.get("id")
            pool_name = pool.get("name", "")
            process_ref = pool.get("processRef")  # Reference to the process this pool is associated with
            pool_obj = Pool(pool_id=pool_id, name=pool_name)
            self.pools.append(pool_obj)

        # Parse lanes and associate them with pools
        node_to_lane = {}
        for lane_set in root.findall(".//{http://www.omg.org/spec/BPMN/20100524/MODEL}laneSet"):
            for lane in lane_set.findall("{http://www.omg.org/spec/BPMN/20100524/MODEL}lane"):
                lane_id = lane.get("id")
                lane_name = lane.get("name", "")
                lane_obj = Lane(lane_id=lane_id)
                self.lanes.append(lane_obj)

                # Assign lanes to pools based on processRef
                for pool in self.pools:
                    if process_ref and process_ref in [pool.pool_id]:
                        pool.add_lane(lane_obj)

                # Map flow nodes to lanes
                for flow_node_ref in lane.findall("{http://www.omg.org/spec/BPMN/20100524/MODEL}flowNodeRef"):
                    node_id = flow_node_ref.text
                    if node_id:
                        node_to_lane[node_id] = lane_id

        # Parse nodes (tasks, events, gateways, and data objects)
        for process in root.findall(".//{http://www.omg.org/spec/BPMN/20100524/MODEL}process"):
            for element in process:
                element_id = element.get("id")
                if not element_id:
                    continue  # Skip elements without an ID
                element_name = element.get("name", "")
                element_type = element.tag.split("}")[1]  # Extract tag name without namespace

                # Data objects
                if element_type in ["dataObject", "dataObjectReference"]:
                    if not element_name.strip():  # Skip blank data objects
                        continue
                    data_obj = DataObj(label=element_name, data_type=element_type)
                    self.data_objects.append(data_obj)
                    continue

                # Skip sequence flows and lane sets
                if element_type in ["sequenceFlow", "laneSet", "dataStoreReference"]:
                    continue

                lane_id = node_to_lane.get(element_id, "")

                # FlowNode objects
                flowNode = FlowNode(
                    flowNode_id=element_id,
                    label=element_name,
                    flowNode_type=element_type,
                    lane_id=lane_id
                )
                self.flowNodes.append(flowNode)

                # Assign activity to corresponding lane
                if lane_id:
                    lane = next((l for l in self.lanes if l.lane_id == lane_id), None)
                    if lane:
                        lane.add_flowNode(flowNode)

        # Parse sequence flows (edges)
        for process in root.findall(".//{http://www.omg.org/spec/BPMN/20100524/MODEL}process"):
            for sequence_flow in process.findall("{http://www.omg.org/spec/BPMN/20100524/MODEL}sequenceFlow"):
                flow_id = sequence_flow.get("id")
                source_ref = sequence_flow.get("sourceRef")
                target_ref = sequence_flow.get("targetRef")
                flow_label = sequence_flow.get("name", "")

                if source_ref and target_ref:  # source and target refs are IDs, we need to find corresponding activities
                    # Find source and target activities
                    source = next((a for a in self.flowNodes if a.flowNode_id == source_ref), None)
                    target = next((a for a in self.flowNodes if a.flowNode_id == target_ref), None)

                    if source and target:
                        flow = Flow(flow_id=flow_id, label=flow_label, source=source, target=target)
                        self.flows.append(flow)

        return self
    
    def to_bpmn(self):
        """
        Converts the Process object back into a BPMN XML tree with the correct namespace prefix.
        :return: An ElementTree representing the BPMN XML.
        """
        # Define the BPMN namespace and prefix
        bpmn_namespace = "http://www.omg.org/spec/BPMN/20100524/MODEL"
        ET.register_namespace("bpmn", bpmn_namespace)  # Register the namespace with the prefix 'bpmn'

        # Create the root BPMN element
        definitions = ET.Element(f"{{{bpmn_namespace}}}definitions")

        # Initialize the process element as a standalone process
        process = ET.SubElement(definitions, f"{{{bpmn_namespace}}}process")

        # Create a mapping of flow node IDs to their incoming and outgoing flows
        incoming_flows = {node.flowNode_id: [] for node in self.flowNodes}
        outgoing_flows = {node.flowNode_id: [] for node in self.flowNodes}

        for flow in self.flows:
            outgoing_flows[flow.source.flowNode_id].append(flow.id)
            incoming_flows[flow.target.flowNode_id].append(flow.id)

        # Add flow nodes
        for flow_node in self.flowNodes:
            node_element = ET.SubElement(process, f"{{{bpmn_namespace}}}{flow_node.type}")
            node_element.set("id", flow_node.flowNode_id)
            if flow_node.label:
                node_element.set("name", flow_node.label)

            # Add outgoing flows
            for outgoing in outgoing_flows[flow_node.flowNode_id]:
                outgoing_element = ET.SubElement(node_element, f"{{{bpmn_namespace}}}outgoing")
                outgoing_element.text = outgoing

            # Add incoming flows
            for incoming in incoming_flows[flow_node.flowNode_id]:
                incoming_element = ET.SubElement(node_element, f"{{{bpmn_namespace}}}incoming")
                incoming_element.text = incoming

        # Add sequence flows
        for flow in self.flows:
            flow_element = ET.SubElement(process, f"{{{bpmn_namespace}}}sequenceFlow")
            flow_element.set("id", flow.id)
            flow_element.set("sourceRef", flow.source.flowNode_id)
            flow_element.set("targetRef", flow.target.flowNode_id)
            if flow.label:
                flow_element.set("name", flow.label)

        return ET.ElementTree(definitions)

    def print_process_state(self):
        # print(f"\n{message}")
        print("FlowNodes:")
        for flowNode in self.flowNodes:
            print(flowNode)
        print("\nFlows:")
        for flow in self.flows:
            print(flow)
        print("\n")

    def clone(self):
        return copy.deepcopy(self)
    
    def get_node(self, flowNode_id):
        return next((node for node in self.flowNodes if node.flowNode_id == flowNode_id), None)


    def __repr__(self):
        return f"Process(flowNodes={len(self.flowNodes)}, flows={len(self.flows)}, lanes={len(self.lanes)})"
