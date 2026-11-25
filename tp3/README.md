# Compte rendu du TP 3 — RAG multimodal et agentique

**Hiba Hamdaoui**

## Introduction

Dans ce TP, j’ai construit un assistant capable de rechercher des passages et des figures dans l’article *Attention Is All You Need*. L’assistant accepte une question en texte, éventuellement accompagnée d’une image, et répond en citant les pages du PDF. Le [notebook du TP](TP_RAG_Multimodal.ipynb) contient le code et les résultats enregistrés.

## Installation

Le projet utilise Python 3.13 et `uv`. Depuis le dossier du TP, exécuter les commandes suivantes.

```powershell
uv sync --locked
uv run --locked --with jupyterlab jupyter lab
```

Ouvrir `TP_RAG_Multimodal.ipynb` et exécuter les cellules dans l’ordre. Le notebook lit le [PDF fourni](pdfs/attention_is_all_you_need.pdf). Le premier lancement télécharge les poids de CLIP.

Pour exécuter les tests de l’agent avec Groq, créer un fichier `.env` à côté du notebook et remplacer la valeur de `GROQ_API_KEY` par sa propre clé.

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=votre_cle
GROQ_MODEL=qwen/qwen3.8-27b
```

Le notebook peut aussi utiliser un serveur Ollama local avec `LLM_PROVIDER=ollama` et un modèle installé, désigné par `OLLAMA_MODEL`. Pour exécuter le test avec une figure, ce modèle doit accepter les images et les appels d’outils.

## 1. Extraction du PDF

J’ai utilisé PyMuPDF pour extraire le texte page par page et enregistrer les images intégrées. Les noms des fichiers conservent le numéro de page, ce qui permet de retrouver la source d’un passage ou d’une figure. L’extraction a produit **15 pages de texte et 3 images**.

![Architecture du Transformer extraite de la page 3 du PDF](resultats_pdf/bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697/images/page_003_image_128.png)

## 2. Vectorisation et indexation

J’ai découpé le texte en fragments de 60 jetons avec un chevauchement de 10 jetons. CLIP ViT-B/32 encode les fragments et les images en vecteurs normalisés de dimension 512 dans un espace commun. J’ai stocké les vecteurs dans ChromaDB avec le type de contenu, le numéro de page et le chemin de la source. L’empreinte SHA-256 du PDF permet de réindexer le même document sans créer de doublons.

| Contenu | Nombre indexé |
|---|---:|
| Fragments de texte | 208 |
| Images | 3 |
| Total | 211 |

## 3. Recherche multimodale

ChromaDB classe les résultats par similarité cosinus. Le filtre permet de rechercher uniquement des passages ou des images. Lorsqu’une question est accompagnée d’une image, l’outil effectue les deux recherches et combine leurs classements par fusion des rangs réciproques (RRF).

La recherche « multi-head attention » a retrouvé notamment le passage de la page 5 qui explique le rôle des têtes parallèles. En soumettant une figure déjà indexée, j’ai retrouvé cette même figure en première position avec une similarité de **1,000**.

## 4. Agent ReAct avec LangGraph

J’ai construit un graphe qui alterne entre le modèle de langage et l’outil `retriever_tool`. Le modèle formule une recherche, l’outil renvoie les passages ou figures avec leurs références, puis le modèle peut lancer une autre recherche ou répondre en français. Le notebook affiche les appels d’outil et les sources retrouvées.

## 5. Tests de l’agent

Les cinq cas ci-dessous ont produit une réponse et une trace d’appel à l’outil dans le notebook.

| Test | Résultat observé |
|---|---|
| Rôle de l’attention multi-têtes | L’agent explique que les têtes peuvent traiter différentes relations en parallèle et cite notamment la page 5. |
| Recherche de la figure 2 | L’agent retrouve les mécanismes d’attention par produit scalaire mis à l’échelle et d’attention multi-têtes. |
| Attention multi-têtes et encodage positionnel | L’agent compare le rôle des deux mécanismes à partir des passages récupérés. |
| Score de GPT-4 sur MMLU | L’agent indique que le score demandé ne figure pas dans l’article. |
| Identification d’une image fournie | L’agent reconnaît la figure 1, qui représente l’encodeur et le décodeur du Transformer. |

## Conclusion

Ce TP m’a permis d’associer la recherche de passages et de figures à un agent capable de citer ses sources. Les essais couvrent la recherche textuelle, l’identification d’une figure et le traitement d’une question sans réponse dans le PDF. La pertinence des passages, l’interprétation des figures et l’exactitude des citations restent à vérifier dans les réponses générées.
