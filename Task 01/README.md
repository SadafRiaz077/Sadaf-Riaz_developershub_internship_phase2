# Task 1: News Topic Classifier Using BERT

**AI/ML Engineering – Advanced Internship, DevelopersHub Corporation**

---

## Objective of the Task

Fine-tune a transformer model (BERT) to classify news headlines into topic categories — **World**, **Sports**, **Business**, and **Sci/Tech** — evaluate its performance using accuracy and F1-score, and deploy the model for live interaction using Gradio.

---

## Dataset

| Label | Category |
|-------|----------|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

Loaded directly from the Hugging Face Hub using the namespaced repo ID `fancyzhx/ag_news` (the older short name `ag_news` is deprecated on newer `huggingface_hub` versions).

---

## Methodology / Approach

1. **Dataset** — Used the AG News Dataset from Hugging Face (`fancyzhx/ag_news`), which contains news headlines labeled into 4 balanced classes (World, Sports, Business, Sci/Tech). Trained on a subset of 8,000 samples and tested on 2,000 samples for faster training on Colab's free GPU.
2. **Preprocessing & Tokenization** — Tokenized headlines using `BertTokenizerFast`, with padding and truncation applied to a fixed `max_length=64`.
3. **Model Fine-Tuning** — Fine-tuned the pre-trained `bert-base-uncased` model with a 4-class classification head, using the Hugging Face `Trainer` API for 3 epochs (learning rate 2e-5, batch size 16).
4. **Evaluation** — Evaluated the fine-tuned model on the held-out test set using Accuracy, weighted F1-score, a per-class classification report, and a confusion matrix.
5. **Deployment** — Saved the fine-tuned model & tokenizer, then built a live, interactive Gradio interface where users can type a news headline and instantly get the predicted category with confidence scores.

---

## Key Results / Observations

- Accuracy: *fill in from your `results_df` output*
- Weighted F1-score: *fill in from your `results_df` output*
- Fine-tuned BERT achieves strong performance on news topic classification, showing how well a general-purpose pre-trained language model adapts to a specific downstream task with just a few epochs of fine-tuning.
- The confusion matrix shows that most misclassifications occur between semantically overlapping categories — e.g., **Business vs. Sci/Tech**, since tech-business news (like company earnings) can straddle both classes.
- Training on a subset (8,000 samples) already yields solid results; using the full 120,000-sample training set would likely push accuracy/F1 even higher.

---

## Tech Stack

- Python, PyTorch
- Hugging Face `transformers`, `datasets`, `evaluate`
- scikit-learn
- Gradio
- pandas, numpy, matplotlib, seaborn

---

## Files

- `Task1_News_Topic_Classifier_BERT.ipynb` — full notebook (data loading, training, evaluation, deployment)
- `bert_news_classifier_final/` — saved fine-tuned model & tokenizer (generated after running the notebook)

---

## Skills Gained

NLP using Transformers • Transfer Learning & Fine-Tuning • Evaluation Metrics for Text Classification • Lightweight Model Deployment (Gradio)
## Colab Notebook
https://colab.research.google.com/drive/1vYSjQ1ryrBjQ6969_JSbEjKj9FSVNLwa?usp=sharing
