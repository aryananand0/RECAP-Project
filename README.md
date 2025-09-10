# Disaster Damage Project (DDP)

Turn paired **pre** and **post** satellite images into **building-level damage labels** with **calibrated confidence**, then present the results on a clear, clickable map for triage.

---

## Problem statement and motivation
After a disaster, teams need a fast view of which buildings are likely safe, damaged, or destroyed. Field surveys are slow and risky. Raw satellite images are not directly actionable. This project builds a compact pipeline that converts pre and post image pairs into building-level damage categories with usable confidence, and exposes the results in a simple map that non-technical users can navigate.

---

## Data and task
- **Dataset**: xView2/xBD style pairs with building footprints and four labels: **no, minor, major, destroyed**  
- **Chips**: small image crops (for example 256×256 px) centered on each building, taken from both **pre** and **post** images  
- **Split**: by **event** to avoid leakage between train and test

---

## Approach (MVP)
- **Model**: Siamese ResNet-18 on pre and post chips with shared weights  
  - Combine features as `[f_pre, f_post, |f_pre − f_post|]`  
  - Loss: class-weighted cross-entropy  
  - Augmentations: flips, small rotations, light color jitter  
- **Calibration**: temperature scaling on validation logits, report ECE and a calibration curve  
- **Explainability**: Grad-CAM on the post branch for qualitative checks  
- **Demo app**: Streamlit with Leafmap or Folium  
  - Color-coded polygons, event selector, confidence threshold slider  
  - On click: show pre and post chips, predicted label, confidence, optional Grad-CAM toggle  
  - Exports: CSV and GeoJSON  
  - Load **precomputed** predictions for 1 to 2 events so the app runs smoothly on a laptop

---

## Initial hypotheses and goals
- **H1**: Image-only Siamese baseline reaches **macro-F1 ≥ 0.60** on a held-out event split. Most confusions are minor vs major.  
- **H2**: Temperature scaling reduces Expected Calibration Error by **≥ 30%** relative to the uncalibrated model and yields steadier threshold behavior.  
- **H3 (stretch)**: A light re-ranking with tweet embeddings improves Top-K recall for major and destroyed at K in {20, 50} on one event.

**Success criteria**
- Macro-F1 near or above 0.60 on held-out events  
- Calibrated probabilities with predictable threshold behavior  
- Clean demo app that runs locally with precomputed predictions

---

## Architecture overview

    User
      ↓
    Streamlit Frontend (map + UI)
      - event selector, threshold slider
      - popup: pre/post chips, label, confidence, Grad-CAM toggle
      ↓
    Option A: direct Python call (simple local demo)
      streamlit_app.py → infer.predict_batch()
      → outputs/predictions/<event>/{predictions.csv, buildings.geojson}
    
    Option B: API boundary (optional)
      streamlit_app.py → FastAPI /predict or /batch
                        → Inference module (PyTorch)
                        → returns GeoJSON + metrics JSON

**Artifacts**
- Model weights (.pt)  
- Chips (PNG 224–256 px) in `data/chips/{pre,post}/`  
- Index (parquet or csv) with paths, labels, coords, event_id  
- GeoJSON polygons and predictions.csv for the app

---


## Quickstart (local demo with precomputed predictions)
1. Create environment and install requirements  
   `pip install -r requirements.txt`
2. Generate chips and an index for one event  
   `python ddp/etl/make_chips.py --event <EVENT_ID>`
3. Train baseline and save checkpoint  
   `python ddp/models/train.py --config experiments/exp001.yaml`
4. Run batch inference to produce CSV and GeoJSON  
   `python ddp/models/infer.py --event <EVENT_ID> --ckpt outputs/checkpoints/best.pt`
5. Launch the app  
   `streamlit run ddp/app/streamlit_app.py`

---

## Stretch (time permitting)
- Inspection queue page ranked by severity × confidence  
- Late-fusion re-ranker using tweet embeddings on a single event  
- Small ablations: focal loss vs class-weighted CE, chip size sensitivity

---

## Out of scope
No live satellite tasking, no production service, no operational claims. This is a reproducible classroom prototype.
