from process.process import Process

class ModelAlteration:
    flow_count = 0  # Shared counter for unique flow IDs across all alterations
    flowNode_count = 0  # Counter for unique FlowNode IDs

    def __init__(self, reference_model: Process):
        self.reference_model = reference_model
        self.altered_model = None

        # Mapping of alteration names to their module and class names
        self.alteration_mapping = {
            "add_activity": ("add_flowNode", "AddFlowNode"),
            "add_flow": ("add_flow", "AddFlow"),
            "add_gateway": ("add_gateway", "AddGateway"),
            "remove_activity": ("remove_flowNode", "RemoveFlowNode"),
            "remove_flow": ("remove_flow", "RemoveFlow"),
            "remove_gateway": ("remove_gateway", "RemoveGateway"),
            "change_label": ("change_label", "ChangeLabel"),
        }

    def apply_alteration(self, alteration_name: str, repetitions: int = 1):
        
        if self.altered_model is None:
            self.altered_model = self.reference_model.clone()

        # module and class names
        alteration_info = self.alteration_mapping.get(alteration_name)
        if not alteration_info:
            raise ValueError(f"Unknown alteration: {alteration_name}")

        module_name, class_name = alteration_info

        # dynamic import and instantiation
        alteration_module = __import__(f"model_alteration.{module_name}", fromlist=[class_name])
        alteration_class = getattr(alteration_module, class_name)
        alteration_instance = alteration_class()  # Instantiate the alteration class

        for i in range(repetitions):
            self.altered_model = alteration_instance.apply(self.altered_model)
            print(f"Applied alteration: {alteration_name}")
            print(f"Current number of Flow nodes: {len(self.altered_model.flowNodes)}")
            print(f"Current number of Flows: {len(self.altered_model.flows)}")

        self.altered_model.print_process_state()

        return self.altered_model
