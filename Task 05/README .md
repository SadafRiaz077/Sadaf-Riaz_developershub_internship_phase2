# Auto-Tagging Support Tickets Using an LLM

**Task 5 — DevelopersHub Corporation | AI/ML Engineering Internship (Advanced Task Set)**

An automated system that tags free-text customer support tickets with the most
relevant category labels using a Large Language Model (LLM), comparing **zero-shot
classification** against **few-shot prompting**, and returning the **top 3 most
probable tags per ticket**.

---

## Objective

Support teams receive large volumes of free-text tickets that need to be
categorized and routed to the right team. Manual tagging is slow and
inconsistent. This project uses pretrained LLMs to automate that tagging step
with **no dedicated training set required**, and evaluates whether giving the
model a few labeled examples in the prompt (few-shot / in-context learning)
improves on a pure zero-shot baseline.

## Tag Taxonomy

| Tag | Description |
|---|---|
| Technical Issue | App/website performance, crashes, connectivity problems |
| Billing & Payment | Charges, invoices, refund amounts, payment methods |
| Account Access | Login, password reset, 2FA, account lockouts |
| Feature Request | Suggestions for new functionality |
| Bug Report | Unexpected/incorrect product behavior |
| Cancellation & Refund | Subscription cancellation, downgrades, refunds |
| Product Complaint | Dissatisfaction with product quality, UX, or service |
| General Inquiry | Informational questions (pricing, hours, policies) |

## Methodology / Approach

1. **Dataset** — 96 free-text support tickets, hand-authored and generated
   directly inside the notebook (balanced across the 8 categories above — see
   [Dataset](#-dataset) below).
2. **Preprocessing** — light text normalization (whitespace cleanup); an 8-tag
   candidate label set is derived directly from the data.
3. **Zero-shot classification** — `facebook/bart-large-mnli` via Hugging Face's
   `zero-shot-classification` pipeline, using Natural Language Inference to
   score every candidate tag against each ticket with no labeled examples.
4. **Few-shot prompting** — `google/flan-t5-base`, prompted with 2 labeled
   example tickets per category (in-context learning) plus instructions to
   output the top 3 most relevant tags, ranked.
5. **Evaluation** — both approaches are scored on the *same* held-out
   evaluation set (80 tickets; the 16 few-shot example tickets are excluded
   from evaluation to keep the comparison fair) using:
   - Top-1 Accuracy
   - Macro Precision / Recall / F1
   - Top-3 Accuracy (was the correct tag anywhere in the top 3?)
6. **Output** — for every ticket, the system returns its **top 3 ranked tags**,
   matching the intended production use case of giving a support agent a
   ranked shortlist rather than a single forced label.

## Key Results & Observations

Results from the evaluation run (80 held-out tickets, 8 balanced categories):

| Metric | Zero-Shot (BART-MNLI) | Few-Shot (FLAN-T5-base) |
|---|---|---|
| Top-1 Accuracy | **0.59** | 0.44 |
| Macro Precision | 0.66 | 0.48 |
| Macro Recall | 0.59 | 0.44 |
| Macro F1-score | **0.54** | 0.38 |
| Top-3 Accuracy | *see printed output, Section 6* | *see printed output, Section 6* |

> The exact Top-3 Accuracy values are printed by the `evaluate()` function in
> Section 6, just above the classification reports (look for the lines
> `Top-3 Accuracy : 0.xxx` under `--- Zero-Shot (BART-MNLI) ---` and
> `--- Few-Shot (FLAN-T5, in-context examples) ---`). They are also saved to
> `outputs_metrics_summary.csv` — paste both numbers in here before submitting.

**Qualitative observations (based on the actual run above):**

- **Contrary to the common expectation, zero-shot outperformed few-shot in
  this run** (Top-1 accuracy 0.59 vs. 0.44; macro F1 0.54 vs. 0.38). This is a
  genuinely useful finding rather than a bug: `flan-t5-base` is a relatively
  small (250M parameter) generative model, and it struggled to reliably follow
  the "return exactly 3 ranked, comma-separated tags from this list" instruction
  — several raw generations returned only a single tag or an unclear phrase,
  which the fallback-padding logic in `parse_tags_from_generation()` then had
  to fill in with the next unused tags in a fixed order. This explains why
  *Account Access* and *Billing & Payment* (the two alphabetically-first
  candidate tags) appear disproportionately often in the few-shot top-3 output
  even when they aren't the true category — they were frequently used as
  padding rather than being genuine model predictions.
- **BART-MNLI (zero-shot)**, by contrast, is purpose-built for exactly this
  kind of "does this text entail this label" scoring task, so it produced a
  clean probability over all 8 candidate tags for every ticket with no parsing
  ambiguity — which is likely why it was the more reliable of the two here.
- Both models did well on **Billing & Payment** and **Cancellation & Refund**
  (distinctive vocabulary — "charged", "refund", "invoice") and struggled most
  on **Bug Report** vs. **Technical Issue**, which overlap semantically.
- **Takeaway:** few-shot prompting only pays off if the underlying generative
  model is strong enough to reliably follow the output-format instruction. A
  larger instruction-tuned model (`flan-t5-large`/`flan-t5-xl`, or a commercial
  LLM such as GPT-4 or Claude via API) would likely reverse this result — the
  prompt-engineering scaffold in Section 5 is written so that swap is a
  one-function change (see `few_shot_predict()`).

## Suggested Improvement (optional, for extra credit)

If you want to demonstrate few-shot actually *winning* (the more commonly
expected result), the single highest-leverage change is swapping
`google/flan-t5-base` for a larger model — e.g. `google/flan-t5-large` — in the
`FEWSHOT_MODEL_NAME` variable in Section 5, then re-running from that cell
onward. Larger instruction-tuned models follow the ranked-tag-list format far
more reliably, which should reduce the fallback-padding effect described above
and raise both accuracy and macro F1.

## Repository Structure

```
.
├── auto_tagging_support_tickets.ipynb   # Main deliverable notebook (self-contained — just run it)
├── build_notebook.py                    # Script that programmatically generates the notebook (reference only)
├── data/
│   ├── support_tickets.csv              # Standalone copy of the 96-ticket dataset, for reference
│   └── generate_dataset.py              # Script that generated support_tickets.csv
├── requirements.txt
└── README.md
```

> Note: the notebook itself does **not** read from `data/support_tickets.csv`
> — it generates the same 96-ticket dataset inline in Section 3 so the whole
> notebook runs standalone in Colab with zero file uploads. The `data/` folder
> is kept in the repo purely as a reference copy / for use outside the notebook.

After running the notebook, it additionally produces:
- `outputs_tag_distribution.png` — dataset category distribution
- `outputs_metric_comparison.png` — zero-shot vs. few-shot bar chart
- `outputs_confusion_matrix.png` — confusion matrix (few-shot, top-1)
- `outputs_zero_shot_predictions.csv`, `outputs_few_shot_predictions.csv`
- `outputs_metrics_summary.csv`

## How to Run (Google Colab — recommended)

`auto_tagging_support_tickets.ipynb` is **fully self-contained**: the dataset is
generated in-notebook (no file uploads needed) and the first code cell installs
every dependency. Just:

1. Go to [colab.research.google.com](https://colab.research.google.com) →
   **File → Upload notebook** → select `auto_tagging_support_tickets.ipynb`
2. **Runtime → Run all** (CPU runtime is fine; no GPU or API key required)
3. Wait a few minutes on the first run while the two models
   (`bart-large-mnli` ~1.6GB, `flan-t5-base` ~1GB) download from the Hugging
   Face Hub — every cell after that runs in seconds
4. The last section has an optional cell to download all generated CSVs/charts
   to your own computer via `google.colab.files.download(...)`

## How to Run (locally)

```bash
git clone <your-repo-url>
cd auto-tagging-support-tickets
pip install -r requirements.txt
jupyter notebook auto_tagging_support_tickets.ipynb
# Then: Kernel → Restart & Run All
```

> **Using a commercial LLM instead (OpenAI, Anthropic, etc.):** Section 5 of
> the notebook (`few_shot_predict()`) is written so you can swap in an API call
> in place of the local `flan-t5-base` pipeline, keeping the same
> prompt-construction and evaluation logic.

## Dataset

The notebook generates a hand-authored, realistic dataset of 96 support
tickets (12 per category) inline in Section 3, since the brief does not point
to one fixed, license-clear public dataset. A standalone copy also lives at
`data/support_tickets.csv` (produced by `data/generate_dataset.py`) for
reference or for use outside the notebook. To use real ticket data instead,
replace the generation cell in Section 3 with
`df = pd.read_csv("your_file.csv")`, where your CSV has `ticket_text` and
`true_tag` columns — every cell after that works unchanged.

## Tech Stack

`transformers` (Hugging Face) · `facebook/bart-large-mnli` · `google/flan-t5-base` ·
`scikit-learn` · `pandas` · `matplotlib` / `seaborn` · Jupyter

## Skills Demonstrated

Prompt engineering · LLM-based text classification · Zero-shot & few-shot
learning · Multi-class prediction and ranking · Evaluation methodology
## colab Notebook link:
https://colab.research.google.com/drive/1Crqfa4_Tkg4aIDj41dwbLZJ4EsY0p5Qv?usp=sharing
