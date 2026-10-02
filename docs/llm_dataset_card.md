# Data Card: RetainAI Support Fine-Tuning Dataset

**Source:** bitext/Bitext-customer-support-llm-chatbot-training-dataset (Hugging Face)
**Format:** JSONL, chat format (system / user / assistant) with intent metadata
**Purpose:** fine-tuning a customer-support assistant that RetainAI's agent can use for retention messaging

## Pipeline
| Step | Rows |
|---|---|
| Raw | 26,872 |
| Near-duplicate removal (case/punctuation/whitespace normalized) | 23,984 |
| Length filter (prompt 3–80 words, response 20–400 words) | 23,818 |
| Balanced (max 600 per intent) | 16,002 |
| Train / validation split (95/5, stratified by intent) | 15,201 / 801 |

## Composition
- 27 intents across 11 categories
- 46% of responses contain template placeholders such as `{{Order Number}}`

## Known limitations
- Much of the source text appears template-generated, so it is cleaner and more uniform than real customer messages;
  models trained on it will likely score lower on real traffic.
- English only. Domain is general e-commerce support, not telecom-specific.
- Placeholders must be filled from real account data at inference time.
