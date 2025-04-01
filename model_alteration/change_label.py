from process.process import Process
import random
import string
from nltk.corpus import wordnet
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ChangeLabel:
    def __init__(self, node_id=None, tokenizer=None, model=None):
        self.node_id = node_id
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.model = model or AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")


    def is_sentence(self, label: str) -> bool:
        """Decide whether to treat the label as a sentence."""
        return len(label.strip().split()) > 2  # more than 2 words = likely a sentence

    def paraphrase(self, sentence: str) -> str:
        try:
            text = f"paraphrase: {sentence} </s>"
            encoding = self.tokenizer.encode_plus(
                text,
                padding="max_length",
                return_tensors="pt",
                max_length=48,
                truncation=True
            )
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=48,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.9,
                num_return_sequences=1
            )
            print("[DEBUG] Output generated.", flush=True)

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Paraphrasing failed: {e}", flush=True)
            return sentence


    def get_synonym(self, label: str) -> str:
        """Replace each word with a random synonym from all WordNet synsets."""
        words = label.strip().split()
        new_words = []

        for word in words:
            cleaned_word = word.strip(string.punctuation)
            punctuation = word[len(cleaned_word):] if word.endswith(tuple(string.punctuation)) else ""

            synonyms = wordnet.synsets(cleaned_word)
            all_lemmas = [
                lemma.name().replace("_", " ")
                for syn in synonyms
                for lemma in syn.lemmas()
                if lemma.name().lower() != cleaned_word.lower()
            ]

            if all_lemmas:
                synonym = random.choice(all_lemmas)
                new_words.append(synonym + punctuation)
            else:
                new_words.append(word)

        return " ".join(new_words)

    def apply(self, model: Process) -> Process:
        if not model.flowNodes:
            print("No FlowNodes available to update.")
            return model

        # Choose node
        target_node = None
        if self.node_id:
            target_node = next((node for node in model.flowNodes if node.flowNode_id == self.node_id), None)
            if not target_node:
                print(f"FlowNode with ID {self.node_id} not found. No changes made.")
                return model
        else:
            target_node = random.choice(model.flowNodes)

        original_label = target_node.label

        # Empty label fallback
        if not original_label.strip():
            new_label = "new empty label"
        elif self.is_sentence(original_label):
            new_label = self.paraphrase(original_label)
        else:
            new_label = self.get_synonym(original_label)

        target_node.label = new_label

        print(f"Changing label of node {target_node.flowNode_id} from '{original_label}' to '{new_label}'")
        return model