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
        
        #Parse a BPMN XML tree and populates the Process object
        
        # parse pools
        self.pools = []
        for pool in root.findall(".//{http://www.omg.org/spec/BPMN/20100524/MODEL}participant"):
            pool_id = pool.get("id")
            pool_name = pool.get("name", "")
            process_ref = pool.get("processRef") 
            pool_obj = Pool(pool_id=pool_id, name=pool_name)
            self.pools.append(pool_obj)

        # parse lanes and associate them with pools
        node_to_lane = {}
        for lane_set in root.findall(".//{http://www.omg.org/spec/BPMN/20100524/MODEL}laneSet"):
            for lane in lane_set.findall("{http://www.omg.org/spec/BPMN/20100524/MODEL}lane"):
                lane_id = lane.get("id")
                lane_name = lane.get("name", "")
                lane_obj = Lane(lane_id=lane_id)
                self.lanes.append(lane_obj)

                # assign lanes to pools
                for pool in self.pools:
                    if process_ref and process_ref in [pool.pool_id]:
                        pool.add_lane(lane_obj)

                # flow nodes to lanes mapping
                for flow_node_ref in lane.findall("{http://www.omg.org/spec/BPMN/20100524/MODEL}flowNodeRef"):
                    node_id = flow_node_ref.text
                    if node_id:
                        node_to_lane[node_id] = lane_id

        # parse flow nodes 
        for process in root.findall(".//{http://www.omg.org/spec/BPMN/20100524/MODEL}process"):
            for element in process:
                element_id = element.get("id")
                if not element_id:
                    continue  
                element_name = element.get("name", "")
                element_type = element.tag.split("}")[1]  

                # Data objects
                if element_type in ["dataObject", "dataObjectReference"]:
                    if not element_name.strip():  
                        continue
                    data_obj = DataObj(label=element_name, data_type=element_type)
                    self.data_objects.append(data_obj)
                    continue

                
                if element_type in ["sequenceFlow", "laneSet", "dataStoreReference"]:
                    continue

                lane_id = node_to_lane.get(element_id, "")

                # flowNode objects
                flowNode = FlowNode(
                    flowNode_id=element_id,
                    label=element_name,
                    flowNode_type=element_type,
                    lane_id=lane_id
                )
                self.flowNodes.append(flowNode)

                # map flownodes to lanes
                if lane_id:
                    lane = next((l for l in self.lanes if l.lane_id == lane_id), None)
                    if lane:
                        lane.add_flowNode(flowNode)

        # parse sequence flows 
        for process in root.findall(".//{http://www.omg.org/spec/BPMN/20100524/MODEL}process"):
            for sequence_flow in process.findall("{http://www.omg.org/spec/BPMN/20100524/MODEL}sequenceFlow"):
                flow_id = sequence_flow.get("id")
                source_ref = sequence_flow.get("sourceRef")
                target_ref = sequence_flow.get("targetRef")
                flow_label = sequence_flow.get("name", "")

                if source_ref and target_ref:  
                    source = next((a for a in self.flowNodes if a.flowNode_id == source_ref), None)
                    target = next((a for a in self.flowNodes if a.flowNode_id == target_ref), None)

                    if source and target:
                        flow = Flow(flow_id=flow_id, label=flow_label, source=source, target=target)
                        self.flows.append(flow)

        return self
    
    def to_bpmn(self):


        bpmn_namespace = "http://www.omg.org/spec/BPMN/20100524/MODEL"
        ET.register_namespace("bpmn", bpmn_namespace)  

        definitions = ET.Element(f"{{{bpmn_namespace}}}definitions")

        process = ET.SubElement(definitions, f"{{{bpmn_namespace}}}process")

        incoming_flows = {node.flowNode_id: [] for node in self.flowNodes}
        outgoing_flows = {node.flowNode_id: [] for node in self.flowNodes}

        for flow in self.flows:
            outgoing_flows[flow.source.flowNode_id].append(flow.id)
            incoming_flows[flow.target.flowNode_id].append(flow.id)

        for flow_node in self.flowNodes:
            node_element = ET.SubElement(process, f"{{{bpmn_namespace}}}{flow_node.type}")
            node_element.set("id", flow_node.flowNode_id)
            if flow_node.label:
                node_element.set("name", flow_node.label)

            for outgoing in outgoing_flows[flow_node.flowNode_id]:
                outgoing_element = ET.SubElement(node_element, f"{{{bpmn_namespace}}}outgoing")
                outgoing_element.text = outgoing

            for incoming in incoming_flows[flow_node.flowNode_id]:
                incoming_element = ET.SubElement(node_element, f"{{{bpmn_namespace}}}incoming")
                incoming_element.text = incoming

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
