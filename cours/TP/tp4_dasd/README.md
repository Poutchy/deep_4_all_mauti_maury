# TP4 — Fine-tuning d’un modèle avec DASD

## Description

Ce projet correspond au TP4 et contient les scripts et résultats permettant de :

1. Générer un dataset de questions
2. Générer les réponses associées via un modèle
3. Fine-tuner un modèle de langage avec la méthode LoRA
4. Produire un modèle final prêt à être utilisé pour l’inférence

Le projet est organisé en deux dossiers principaux : `tp4` et `tp4_dasd`.

---

## Dossier `tp4`

Ce dossier contient le code permettant de générer le dataset.

### Fonctionnalités

* Génération d’un dataset de questions
* Formatage des données
* Création d’un dataset final au format `.jsonl`
* Génération des paires :

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

Ce dataset est ensuite utilisé pour entraîner le modèle.

---

## Dossier `tp4_dasd`

Ce dossier contient le code et les résultats du fine-tuning réalisés sur Google Colab.

Les fichiers volumineux inutiles (`optimizer.pt`) ont été supprimés.

---

## Modèle final

Le modèle LoRA entraîné est stocké dans le fichier :

```
adapter_model.safetensors
```

Ce fichier contient les poids entraînés.

Les fichiers associés :

* `adapter_config.json` : configuration LoRA
* `tokenizer.json` : tokenizer
* `tokenizer_config.json` : configuration tokenizer
* `chat_template.jinja` : template de prompt

---

## Dataset

Le dataset est au format JSONL.

Chaque ligne contient une instruction, un input et une réponse :

```json
{
  "instruction": "Question posée",
  "input": "",
  "output": "Réponse attendue"
}
```

Ce dataset est utilisé pour entraîner le modèle avec un apprentissage supervisé.

---