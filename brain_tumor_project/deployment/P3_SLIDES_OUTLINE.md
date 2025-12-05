# P3 Presentation Slides Outline (Dec 2)

Use this as a guide to build your PPT. Keep text minimal, visuals strong.

1. Title & Team
- Project title, course, date
- Team member names, roles

2. Problem & Business Context
- RadiologyFirst Medical Center: backlog + slow diagnosis
- Goal: faster triage via brain tumor classification

3. Data Overview
- Dataset: 4 classes (glioma, meningioma, notumor, pituitary)
- Train/Test split; sample images grid

4. Technical Approach (Milestone 1)
- Baseline CNN architecture (diagram)
- Key choices: IMG_SIZE=150, batch=32, early stopping
- Result summary: accuracy, confusion matrix (baseline)

5. Technical Approach (Milestone 2)
- Transfer learning with VGG16
- Two-stage training: feature extraction + fine-tuning
- Augmentation + class weights + LR scheduling
- Result summary: accuracy, confusion matrix (enhanced)

6. Model Comparison
- Side-by-side metrics table
- Per-class F1 improvements bar chart
- Why enhanced wins (features learned)

7. Data Pipeline (Required Slide)
- Boxes/flow: Ingestion → Preprocess → Augment → Train → Evaluate → Predict → Reports
- Mention scripts: `preprocess.py`, `train_model*.py`, `evaluate*.py`, `predict.py`

8. Demo Plan
- Run order (from EXECUTION_ORDER.md): train → evaluate → compare → predict
- Quick prediction screenshots or live demo

9. Business Value
- Time saved per case; potential revenue impact
- Risk mitigation + operational benefits

10. Lessons & Feedback Implemented
- Encoding fixes for Windows console
- Clear docs, execution order, reproducibility

11. Next Steps
- Deploy as web app/API
- Add explainability (Grad-CAM)

12. Thank You + Q&A
- Contact info; links to repo/docs
