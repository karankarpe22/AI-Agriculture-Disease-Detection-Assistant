# RULES.md — Engineering Rules & Development Invariants

> **Project**: AI Agriculture Assistant (कृषी सहाय्यक)  
> **Status**: Mandatory & Non-Negotiable  
> **Scope**: All AI coding agents, contributors, and maintainers  

---

### Rule 1: Never Delete or Replace Working Functionality
- Existing working code, trained model checkpoints, and API routes must be preserved.
- Never delete a feature to solve a minor bug; fix the root cause with targeted modifications.

### Rule 2: Never Restart the Project From Scratch
- The project is fully implemented, verified, and in production.
- Do not propose migrations, full rewrites, or phase-one resets. Continue from the active implementation state.

### Rule 3: Inspect Before Any Modification
- Inspect all caller modules, configuration files, and test files before making any change.
- Never make assumptions about how a component works; verify directly in the source code.

### Rule 4: Zero Fabrication of ML Metrics
- All reported accuracy, precision, recall, and F1 scores must be traceable to empirical test evaluations in `outputs/evaluation.json`.
- Never invent hypothetical or inflated metrics (e.g., claiming 99% accuracy when measured test accuracy is 81.89%).

### Rule 5: Zero Fabrication of Agricultural Sources
- All treatment, fungicide, and biological control recommendations must originate from verified Indian Council of Agricultural Research (ICAR) or State Agricultural University (SAU) compendiums in `knowledge_base/`.
- Never generate unverified chemical recommendations.

### Rule 6: Never Hardcode API Keys or Credentials
- API keys (Gemini, OpenWeatherMap) must be loaded dynamically from environment variables using `python-dotenv`.
- Never insert raw API tokens into source files, test fixtures, or commits.

### Rule 7: Strictly Isolate `.env`
- The `.env` file must always be listed in `.gitignore`.
- Provide placeholders and configuration instructions strictly through `.env.example`.

### Rule 8: Strictly Preserve the Perception vs. Generation Boundary
- Google Gemini must **never** be used to classify crop leaf diseases directly from images.
- MobileNetV3-Large is the sole diagnostic authority. Gemini acts exclusively as a contextual guidance synthesizer.

### Rule 9: Never Silently Alter the 13 Disease Classes
- The 13 class labels, ordering, and mapping in `models/class_names.json` are cryptographically tied to the output layer weights of `models/mobilenetv3_best.pth`.
- Altering class names or ordering corrupts model predictions across the entire system.

### Rule 10: Never Modify the Dataset Without Documenting Splits
- The 70/15/15 train/val/test split is sealed in `outputs/split_manifest.json` with random seed 42.
- Data modifications must be documented to prevent train-test contamination or leakage.

### Rule 11: Preserve Reproducibility
- All stochastic operations (data shuffling, splitting, transform sampling) must adhere to `random_seed: 42` as defined in `config.yaml`.

### Rule 12: Test Before and After Changes
- Run relevant unit tests prior to modifying code, and execute the full test suite (`python -m pytest tests/ -v`) after every change.
- Never commit code with failing tests.

### Rule 13: Prefer Small, Incremental Changes
- Refactor and patch with surgical precision. Avoid massive, wide-ranging refactors that introduce multi-surface regression bugs.

### Rule 14: Maintain Undergraduate Engineering Accessibility
- Keep code clean, readable, well-commented, and understandable for academic evaluation.
- Avoid over-engineered abstractions or unnecessary design patterns that obscure core logic.

### Rule 15: Avoid Unnecessary Technologies & Dependencies
- Do not introduce complex frameworks (e.g., Docker, Kubernetes, Celery, Redis, React, Webpack) unless explicitly required by user requirements.
- Maintain the lightweight, dependency-minimal Python + Vanilla Web architecture.

### Rule 16: Keep RAG Lightweight and Memory-Safe
- Maintain zero-RAM or low-RAM retrieval mechanisms during web inference.
- Do not load heavy multi-hundred-megabyte transformer models in memory during request handling to preserve 512MB hosting stability.

### Rule 17: Keep the Core System Stable Before Optional Features
- Prioritize the stability of leaf diagnosis, weather risk, RAG retrieval, and Gemini guidance before introducing new peripheral features.
