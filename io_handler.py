import xml.etree.ElementTree as ET


class IOHandler:
    @staticmethod
    def read_bpmn(file_path: str) -> ET.ElementTree:
        """
        Reads a BPMN file and returns it as an ElementTree.
        
        :param file_path: Path to the BPMN file.
        :return: ElementTree representation of the BPMN file.
        """
        try:
            tree = ET.parse(file_path)
            print(f"Successfully loaded BPMN file: {file_path}")
            return tree
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            raise
        except ET.ParseError:
            print(f"Error: Failed to parse BPMN file at {file_path}")
            raise

    @staticmethod
    def write_bpmn(bpmn_model: ET.ElementTree, output_path: str):
        """
        Writes a BPMN model to a file.
        
        :param bpmn_model: ElementTree representation of the BPMN model.
        :param output_path: Path to save the BPMN file.
        """
        try:
            bpmn_model.write(output_path, encoding="utf-8", xml_declaration=True)
            print(f"Successfully saved BPMN file to: {output_path}")
        except Exception as e:
            print(f"Error: Failed to write BPMN file to {output_path}. {str(e)}")
            raise
