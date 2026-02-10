import numpy as np
from json import loads, dumps, JSONEncoder
from dataclasses import is_dataclass, asdict, dataclass
import torch
from random import sample
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from concurrent.futures import ThreadPoolExecutor


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
OUTPUT_TEACHER_FILE = "formated_teacher.jsonl"
OUTPUT_STUDENT_FILE = "formated_student.jsonl"
OUTPUT_KEEP_B_FILE = "keep_b.jsonl"
OUTPUT_KEEP_H_FILE = "keep_h.jsonl"

objs = []

with open(INPUT_FILE, "r", encoding="utf-8") as input:
    for obj in input.readlines():
        objs.append(loads(obj))
objs.sort(key=lambda x: x["id"])


class EnhancedJSONEncoder(JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


class DASPipelineQwen:
    def __init__(
        self, student_model_id="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
    ):
        """
        Initialise le pipeline DAS avec un Teacher (API) et un Student (Local 4-bit).
        """
        # 1. Configuration Student (4-bit quantization)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.student_model_id = student_model_id
        print(f"Chargement du modèle étudiant : {self.student_model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.student_model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.student_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def get_teacher_data(_, id) -> TeacherElement:
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
            question,
            content,
            tokens,
            logprobs,
            total_logprob,
            mean_logprob,
            len(tokens),
            id % 2,
        )

    def get_student_logprobs(
        self, l_prompt: list[str], l_response: list[str]
    ) -> list[StudentElement]:
        """
        Calcule les log-probabilités de la réponse (Student) de manière robuste.
        Utilise la méthode de masquage standard (Labels = -100 pour le prompt).
        """
        # 1. Préparer le texte complet (Prompt + Réponse)
        # On utilise le chat template qui gère proprement les balises <|im_start|>, etc.
        l_messages = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            for prompt, response in zip(l_prompt, l_response)
        ]
        l_full_text = [
            self.tokenizer.apply_chat_template(messages, tokenize=False)
            for messages in l_messages
        ]

        # 2. Tokenizer le tout
        # return_tensors='pt' nous donne directement les tenseurs PyTorch
        inputs = self.tokenizer(
            l_full_text, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        input_ids = inputs.input_ids

        # 3. Identifier la longueur du Prompt pour le masquage
        # On regénère le prompt SEUL avec l'amorce de réponse (add_generation_prompt=True)
        # Cela inclut "<|im_start|>assistant\n" à la fin, pour s'aligner parfaitement.
        l_prompt_messages = [
            [{"role": "user", "content": prompt}] for prompt in l_prompt
        ]
        l_prompt_text = [
            self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            for prompt_messages in l_prompt_messages
        ]

        # On tokenise le prompt seul pour avoir sa longueur exacte en tokens
        l_prompt_tokens = self.tokenizer(
            l_prompt_text, return_tensors="pt", add_special_tokens=False, padding=True
        ).input_ids

        pad_id = self.tokenizer.pad_token_id
        response_start_idx = (l_prompt_tokens != pad_id).sum(dim=1).tolist()

        # 4. Créer les Labels (Masking du Prompt)
        # -100 est l'index ignoré par défaut par CrossEntropyLoss de PyTorch
        labels = input_ids.clone()
        # On masque tout ce qui est avant le début de la réponse
        for i, start in enumerate(response_start_idx):
            labels[i, :start] = -100
        print("torch time")

        # 5. Calcul "Clean" avec CrossEntropyLoss
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        print("torch finished")
        # Shift des logits et labels pour la prédiction "next token"
        # logits[t] prédit labels[t+1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # reduction='none' nous donne la perte pour chaque token individuel
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)

        # La Loss est par définition -log(p), donc log_prob = -loss
        token_logprobs = -token_losses

        l_results: list[StudentElement] = []

        for i in range(token_logprobs.size(0)):
            valid_mask = shift_labels[i] != -100
            valid_logprobs = token_logprobs[i][valid_mask].detach().cpu().numpy()

            total_logprob = float(np.sum(valid_logprobs))
            mean_logprob = (
                float(np.exp(np.mean(valid_logprobs)))
                if len(valid_logprobs) > 0
                else 0.0
            )

            l_results.append(
                StudentElement(
                    total_logprob,
                    mean_logprob,
                    len(valid_logprobs),
                    valid_logprobs.tolist(),
                )
            )

        return l_results

    def decide_keep_prompt(_, teacher_answer, student_answer) -> bool:
        teacher_logprob = teacher_answer.get("mean_logprob", 0.0)
        student_logprob = student_answer.get("mean_logprob", 0.0)

        divergence = student_logprob - teacher_logprob

        return divergence < 0.01


# if __name__ == "__main__":
#     pipeline = DASPipelineQwen()
#     formated_student: list[StudentElement] = []
#     with ThreadPoolExecutor() as pool:
#         formated_teacher: list[TeacherElement] = list(
#             pool.map(pipeline.get_teacher_data, range(len(objs)))
#         )
#     print("init finished")
#     batch_size = 10
#     for i in range(0, len(objs), batch_size):
#         last_iteration = formated_teacher[i : i + batch_size]
#         questions = [l.question for l in last_iteration]
#         responses = [l.content for l in last_iteration]
#         student_batch = pipeline.get_student_logprobs(questions, responses)
#         formated_student.extend(student_batch)
#         print(f"{i} - {i+batch_size-1} done")

#     with open(OUTPUT_TEACHER_FILE, "w", encoding="utf-8") as out:
#         for obj in formated_teacher:
#             out.write(dumps(obj, cls=EnhancedJSONEncoder) + "\n")
#     with open(OUTPUT_STUDENT_FILE, "w", encoding="utf-8") as out:
#         for obj in formated_student:
#             out.write(dumps(obj, cls=EnhancedJSONEncoder) + "\n")

#     kept = [
#         t
#         for t, s in zip(formated_teacher, formated_student)
#         if pipeline.decide_keep_prompt(t, s)
#     ]

#     with open(OUTPUT_KEEP_FILE, "w", encoding="utf-8") as out:
#         for obj in kept:
#             out.write(dumps(obj, cls=EnhancedJSONEncoder) + "\n")

if __name__ == "__main__":
    pipeline = DASPipelineQwen()
    with ThreadPoolExecutor() as pool:
        formated_teacher: list[TeacherElement] = list(
            pool.map(pipeline.get_teacher_data, range(len(objs)))
        )
    print("init finished")
    basse = []
    haute = []
    for i in range(len(objs)):
        iteration = formated_teacher[i]
        student = {
            "instruction": iteration.question,
            "output": iteration.content,
        }
        if iteration.high % 2:
            haute.append(student)
        else:
            basse.append(student)

    with open("haut_" + OUTPUT_TEACHER_FILE, "w", encoding="utf-8") as out:
        for obj in [f for f in formated_teacher if f.high % 2]:
            out.write(dumps(obj, cls=EnhancedJSONEncoder) + "\n")
    with open("basse_" + OUTPUT_TEACHER_FILE, "w", encoding="utf-8") as out:
        for obj in [f for f in formated_teacher if not f.high % 2]:
            out.write(dumps(obj, cls=EnhancedJSONEncoder) + "\n")

    # kept_basse = [t for t in sample(basse, 500)]
    # with open(OUTPUT_KEEP_B_FILE, "w", encoding="utf-8") as out:
    #     for obj in kept_basse:
    #         out.write(dumps(obj, cls=EnhancedJSONEncoder) + "\n")
    
    # kept_haute = [t for t in sample(haute, 500)]
    # with open(OUTPUT_KEEP_H_FILE, "w", encoding="utf-8") as out:
    #     for obj in kept_basse:
    #         out.write(dumps(obj, cls=EnhancedJSONEncoder) + "\n")