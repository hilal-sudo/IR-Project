# Multimodal Video Retrieval System

A semantic video retrieval system that combines visual and textual representations to retrieve the most relevant video for a natural-language text query. Built using OpenAI's CLIP model, Late Fusion of visual/textual embeddings, and FAISS for efficient similarity search.

## Overview

Given a text query (e.g. *"woman talking"*), the system retrieves the most semantically relevant videos from a video collection — without relying on metadata or keyword matching. It does this by representing both videos and text captions in a shared embedding space and performing nearest-neighbor search.

## How It Works

### 1. Data Acquisition
- Source: a 1,000-sample subset of the **Panda-70M** dataset (`panda_test_1000.csv`), containing video URLs and corresponding captions.
- Videos are downloaded using `yt-dlp`, capped at 100MB per file, in MP4 format.

### 2. Frame Extraction
- For each downloaded video, **12 frames** are uniformly sampled across the full video duration using `numpy.linspace` over the total frame count.
- Frames are extracted with OpenCV (`cv2.VideoCapture`) and saved as individual JPEG images per video.

### 3. Visual Embedding (CLIP)
- Each extracted frame is encoded into a 512-dimensional vector using OpenAI's **CLIP (ViT-B/32)** model.
- Frame-level embeddings are L2-normalized, then aggregated into a single video-level embedding via **max pooling** across the 12 frame embeddings.

### 4. Text Embedding (CLIP)
- Each video's caption is encoded using the same CLIP model's text encoder, producing a 512-dimensional text embedding aligned with the visual embedding space.

### 5. Late Fusion
- Visual and textual embeddings for each video are combined using a weighted **Late Fusion** strategy:

  ```
  fused_embedding = α × visual_embedding + (1 − α) × text_embedding
  ```

  with `α = 0.9`, weighting the visual signal more heavily. The fused vector is re-normalized (L2) after combination.

### 6. Indexing & Retrieval (FAISS)
- All fused video embeddings are stacked into a matrix and indexed using **FAISS** (`IndexFlatIP`, inner-product similarity on L2-normalized vectors — equivalent to cosine similarity).
- At query time, the input text is encoded with CLIP's text encoder, normalized, and used to search the FAISS index for the top-k most similar videos.

### 7. Evaluation
- The labeled dataset (each video paired with its ground-truth caption) is split 80/20 into train/test sets.
- Retrieval quality is measured using:
  - **Recall@K** — whether the correct video appears in the top-K retrieved results
  - **Precision@K**
- Evaluated at K = 1, 5, and 10.

## Results

| Metric | Score |
|---|---|
| Recall@5 | > 0.80 |

*(Evaluated on the held-out test split using exact-match video retrieval against ground-truth captions.)*

## Tech Stack

- **Python**
- **PyTorch** + **OpenAI CLIP** (ViT-B/32) — visual & text embeddings
- **OpenCV** — video decoding and frame extraction
- **FAISS** — vector indexing and nearest-neighbor search
- **NumPy / Pandas** — data handling
- **scikit-learn** — train/test splitting
- **yt-dlp** — video acquisition

## Project Structure

```
.
├── notebook.ipynb              # Full pipeline: download → frames → embeddings → fusion → index → eval
├── panda_test_1000.csv         # Dataset subset (video IDs, URLs, captions)
├── data/videos/                # Downloaded video files (.mp4)
├── frames/<video_id>/          # Extracted frames per video
├── train_set.csv               # 80% split, used for reference
└── test_set.csv                # 20% split, used for evaluation
```

## Running the Project

1. Install dependencies:
   ```bash
   pip install pandas numpy opencv-python faiss-cpu torch torchvision ftfy regex tqdm yt-dlp
   pip install git+https://github.com/openai/CLIP.git
   ```
2. Place `panda_test_1000.csv` in the project root.
3. Run the notebook top-to-bottom:
   - Downloads videos → extracts frames → computes CLIP embeddings → builds the FAISS index → runs retrieval and evaluation.
4. To query the system directly:
   ```python
   results = retrieve_videos("woman talking", top_k=5)
   ```

## Notes & Limitations

- Some videos in the source dataset may fail to download (removed, geo-restricted, or exceeding the 100MB cap) and are automatically skipped.
- Evaluation is exact-match (each query has exactly one ground-truth video), so Recall@K is a conservative metric — real-world queries are often relevant to multiple videos.
- The fusion weight (`α = 0.9`) was set manually; it was not tuned via grid search in this version.

## Future Improvements

- Tune the fusion weight (`α`) via validation search rather than a fixed value.
- Replace max pooling over frames with attention-based temporal aggregation.
- Scale evaluation to the full Panda-70M dataset rather than a 1,000-sample subset.
