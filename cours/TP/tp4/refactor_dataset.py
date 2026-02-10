import numpy as np
from json import loads, dumps, JSONEncoder
from dataclasses import is_dataclass, asdict, dataclass
import torch


@dataclass
class TeacherElement:
    question: str
    content: str
    tokens: list[str]
    logprobs: list[float]
    total_logprob: float
    mean_logprob: float
    num_tokens: int
    high: bool

@dataclass
class StudentElement:
    total_logprob: float
    mean_logprob: float
    num_tokens: int
    logprobs: list[float]


INPUT_FILE = "dd_rpg_questions_10000.jsonl"
OUTPUT_FILE = "formated.jsonl"

objs = []

with open(INPUT_FILE, "r", encoding="utf-8") as input:
    for obj in input.readlines():
        objs.append(loads(obj))
objs.sort(key=id)


class EnhancedJSONEncoder(JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


import numpy as np
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class DASPipelineQwen:
    def __init__(self, student_model_id="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"):
        """
        Initialise le pipeline DAS avec un Teacher (API) et un Student (Local 4-bit).
        """
        # 1. Configuration Student (4-bit quantization)
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, )

        self.student_model_id = student_model_id
        print(f"Chargement du modèle étudiant : {self.student_model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.student_model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
                self.student_model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
                )
        self.model.eval()

    def get_teacher_data(id) -> TeacherElement:
        """
        Génère la réponse du Teacher avec les logprobs.
        """
        element = objs[id]

        question = element["question"]
        content = element["response"]
        logprobs_data = element["logprob"]

        tokens = []
        logprobs = []
        # On vérifie si logprobs est disponible (certaines API compatibles ne le renvoient pas)
        if logprobs_data:
            for token_info in logprobs_data:
                tokens.append(token_info["token"])
                logprobs.append(token_info["logprob"])

        # Compute total log probability (sum of logprobs)
        total_logprob = sum(logprobs) if logprobs else 0.0

        # Compute geometric mean of probabilities
        # P_geom = exp(mean(logprobs))
        mean_logprob = np.exp(np.mean(logprobs)) if logprobs else 0.0
        return TeacherElement(
            question, content, tokens, logprobs, total_logprob, mean_logprob, len(tokens), id % 2
        )

    def get_student_logprobs(self, prompt: str, response: str) -> dict:
        """
        Calcule les log-probabilités de la réponse (Student) de manière robuste.
        Utilise la méthode de masquage standard (Labels = -100 pour le prompt).
        """
        # 1. Préparer le texte complet (Prompt + Réponse)
        # On utilise le chat template qui gère proprement les balises <|im_start|>, etc.
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
            ]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # 2. Tokenizer le tout
        # return_tensors='pt' nous donne directement les tenseurs PyTorch
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids

        # 3. Identifier la longueur du Prompt pour le masquage
        # On regénère le prompt SEUL avec l'amorce de réponse (add_generation_prompt=True)
        # Cela inclut "<|im_start|>assistant\n" à la fin, pour s'aligner parfaitement.
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
                )

        # On tokenise le prompt seul pour avoir sa longueur exacte en tokens
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        response_start_idx = prompt_tokens.shape[1]

        # 4. Créer les Labels (Masking du Prompt)
        # -100 est l'index ignoré par défaut par CrossEntropyLoss de PyTorch
        labels = input_ids.clone()
        # On masque tout ce qui est avant le début de la réponse
        labels[:, :response_start_idx] = -100

        # 5. Calcul "Clean" avec CrossEntropyLoss
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

            # Shift des logits et labels pour la prédiction "next token"
            # logits[t] prédit labels[t+1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # reduction='none' nous donne la perte pour chaque token individuel
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)

            # La Loss est par définition -log(p), donc log_prob = -loss
            token_logprobs = -token_losses

            # On ne garde que les tokens de la réponse (ceux qui n'étaient pas masqués à -100)
            # Note: shift_labels a été décalé, donc on utilise son masque
            valid_mask = shift_labels != -100
            valid_logprobs = token_logprobs[valid_mask].cpu().numpy()

        # Calcul des statistiques DAS
        total_logprob = np.sum(valid_logprobs)
        mean_logprob = np.exp(np.mean(valid_logprobs)) if len(valid_logprobs) > 0 else 0.0

        return {
            "total_logprob": total_logprob,
            "mean_logprob":  mean_logprob,
            "num_tokens":    len(valid_logprobs),
            "logprobs":      valid_logprobs.tolist()
            }

    def decide_keep_prompt(teacher_answer, student_answer):
        teacher_logprob = teacher_answer.get("mean_logprob", 0.0)
        student_logprob = student_answer.get("mean_logprob", 0.0)

        print(teacher_logprob, student_logprob)

        divergence = teacher_logprob - student_logprob

        print(divergence)


if __name__ == "__main__":
    formated = []
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for obj in formated:
            out.write(dumps(obj, cls=EnhancedJSONEncoder) + "\n")