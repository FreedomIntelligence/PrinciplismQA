# PrinciplismQA-Demo

This repository hosts a **public demonstration subset** of the **PrinciplismQA** benchmark (from *“Towards Assessing Medical Ethics from Knowledge to Practice”*), intended for transparency, reproducibility, and community use. We plan to open-source the full PrinciplismQA in due course.

---

## 📂 Repository contents

| File / Folder | Description |
|---|---|
| `data/knowledge-mcqa.json` | 100 multiple-choice questions (with 4 options each), selected from the MCQ portion of PrinciplismQA. |
| `data/open-ended-qa.json` | 50 open-ended (free-response) questions and their rubrics selected from the open-ended portion. |
| `data/open-ended-rubric-principles.json` | Medical ethics principles for each rubrics for the selected open-ended questions. |
| `README.md` | This file, describing the subset, usage, license, etc. |
| `LICENSE` | MIT license (covering the repository and scripts). |

`knowledge-mcqa.json` includes MCQAs and their answers. It uses the following schema (per item):

```json
{
    "id": int,
    "question_id": int, // this id was for SOTA LLM verification stage
    "question": "Question title contents",
    "options": {
        "A": "Option A Contents",
        "B": "Option B Contents",
        "C": "Option C Contents",
        "D": "Option D Contents"
    },
    "correct_answer": "A"|"B"|"C"|"D",
    "explanation": "Explanation to the correct answer...",
    "principlism": {
        "autonomy": true|false,
        "nonmaleficience": true|false,
        "beneficience": true|false,
        "justice": true|false
    }
}
```

`open-ended-qa.json` includes open-ended questions and their rubrics. It uses the following schema (per item):

```json
{
    "id": int,
    "tags": [], // list of ethic topic defined by JAMA
    "title": "Case Title",
    "case": "Case context descriptions...",
    "ethical_issues": [
      {
        "question": "Question title contents...",
        "keypoints": [] // list of rubric keypoints for this question
      },
      ... // note that one case may correspond to more than 1 questions
    ]
}
```

`open-ended-rubric-principles.json` uses the following schema (per item):

```json
{
    "id": int,
    "question": "question title",
    "principles": [], // list of principles in Principlism this question corresponds to 
    "keypoint_competencies": [
        {
            "keypoint": "rubric content",
            "competency": "corresponding ACGME competency"
        },
        ...
    ]
},
```

For open-ended items, `options` is `null`.

---

## 🎯 Purpose & Use Cases

* **Rebuttal / review transparency:** reviewers can inspect a small but representative sample to understand model behavior or error modes.
* **Community inspection & baseline checks:** researchers may use this subset to quickly sanity-check methods, sanity tests, or toy experiments.
* **Baseline seed / testing harness:** you may embed this as a small validation split before switching to the full dataset (when released).

> **Note:** this is *not* the full PrinciplismQA. When ready, we will release the full benchmark under the same licensing terms.

We encourage community use (with citation) for non-commercial research and educational work. If you wish to commercialize or integrate in proprietary systems, please contact the authors to discuss terms or future licensing.

---

## ✅ Usage instructions

1. Clone or download this repository.
2. Load the JSON files using your preferred JSON library (e.g. Python’s `json`, `orjson`, etc.).
3. For MCQ items, you may present the four `options` to a model or human annotator and check whether its predicted answer matches `answer`.
4. For open-ended items, you may compute similarity / scoring heuristics or prompt the model and compare output against the `answer` / explanation.
5. Optionally filter by `principle` to study model behavior on specific medical ethics principles.

We recommend you **not** use this small subset as your final evaluation. Rather, it is intended for debugging, sanity checking, and demonstration. When the full dataset is released, you should re-run your full evaluation pipeline against the complete benchmark.

---

## 📜 Licensing & attribution

We currently use the **MIT License** for the repository (code, scripts, and metadata), which is a permissive open-source license.
For background: the MIT License allows reuse, modification, and distribution with minimal restrictions (so long as you retain copyright and license notices).

Because this is a dataset / benchmark subset (not just code), you might wonder whether a dataset-style license (e.g. CC-BY) is more appropriate. As a practical compromise:

* We maintain MIT for the repository itself (scripts, metadata, file structure, etc.).
* You are free to use, adapt, and redistribute the included `.json` items *for research and educational purposes*, provided you retain attribution to the original paper and this repository.
* If you wish to use it in a commercial product or large-scale deployment, please contact us for a licensing discussion.

When we release the **full PrinciplismQA**, we may adopt a more dedicated dataset license (e.g. CC-BY or a data commons license) to better align with norms around data sharing in ML and ethics.

## 🙋 Contributions & issues

* This subset is not intended to be extended by outside users (so we discourage pull requests that add new questions).
* If you find an error (typo in question, mismatch of `answer` or `principle`), please open an **issue** in this repository.
* When the full benchmark is available, we may accept community contributions, issue corrections, or extensions with appropriate review.

---

If you use these subsets in work, we acknowledge (but do not require) users to include a line like:

> “Parts of the data come from the PrinciplismQA-Demo subset; full PrinciplismQA will be released by the authors.”

And meanwhile, please cite us by

```bibtex
@misc{hong2025assessingmedicalethicsknowledge,
      title={Towards Assessing Medical Ethics from Knowledge to Practice}, 
      author={Chang Hong and Minghao Wu and Qingying Xiao and Yuchi Wang and Xiang Wan and Guangjun Yu and Benyou Wang and Yan Hu},
      year={2025},
      eprint={2508.05132},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.05132}, 
}
```
