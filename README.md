# Avsar-AI: Intelligent Internship Recommendation System

An AI-powered internship recommendation system that matches students with internships using semantic embeddings, FAISS-based fast similarity search, and multi-criteria scoring.

## Overview

Avsar-AI is a two-stage recommendation engine:

1. **Offline Stage** (`offline_build_index.py`): Pre-processes internship data and builds a FAISS index for fast similarity search
2. **Online Stage** (`online_ranker.py`): Ranks internships for a given student profile using semantic matching and heuristic scoring

The system leverages:
- **Sentence Transformers** (paraphrase-multilingual-MiniLM-L12-v2) for semantic embeddings
- **FAISS** (Facebook AI Similarity Search) for fast nearest-neighbor lookup
- **Multi-criteria scoring** combining semantic similarity, location preferences, stipend expectations, and experience requirements

## Features

✅ **Fast Retrieval**: FAISS-based indexing for sub-millisecond retrieval of top-K candidates  
✅ **Semantic Understanding**: Embeddings capture meaning beyond keyword matching  
✅ **Multi-Criteria Matching**: Scores based on skills, location, stipend, and experience  
✅ **Explainability**: Provides reasoning for each recommendation  
✅ **Batch Processing**: Efficient batch encoding during index building  

## System Architecture

### Offline Index Building
```
CSV Data → Extract Sections → Encode with Sentence Transformers → Normalize → Build FAISS Index
```

### Online Ranking
```
Student Profile → Extract Sections → Query Vector → FAISS Search (K=300) → Heuristic Scoring → Sort → Top-5 Results
```

## Project Structure

```
trying_to_be_fast.py/
├── offline_build_index.py          # Build FAISS index offline (batch mode)
├── online_ranker.py                # Rank internships for a student (real-time)
├── internship_data_modified_v2.csv # Internship dataset
├── internships.faiss               # Pre-built FAISS index
├── internship_vectors.npy          # Embedding vectors (backup)
└── README.md
```

## How It Works

### 1. Offline Index Building (`offline_build_index.py`)

**Purpose**: Pre-process all internship data once and create a searchable index.

**Steps**:
1. Load internship data from CSV
2. Extract key sections: skills, tasks, experience
3. Batch-encode sections using sentence transformers (BATCH_SIZE=64)
4. Normalize vectors
5. Build FAISS IndexFlatIP (Inner Product) index
6. Save index and vectors for reuse

**Output**: `internships.faiss` (FAISS index) + `internship_vectors.npy` (embeddings)

### 2. Online Ranking (`online_ranker.py`)

**Purpose**: Rank internships for a given student in real-time.

**Scoring Pipeline**:

1. **Parse Student Profile**
   - Extract skills, projects, experience, location preference, expected stipend
   
2. **Build Query Vector**
   - Weighted embedding: 60% skills + 30% projects + 10% experience
   - Normalize to unit vector

3. **FAISS Search**
   - Find top-300 semantically similar internships (fast)
   - Return semantic similarity scores

4. **Heuristic Scoring** (for each result)
   - **Location Match** (20% weight)
     - 1.0 if location matches
     - 0.7 if student can relocate/work remote
     - 0.2 otherwise
   
   - **Stipend Match** (15% weight)
     - 1.0 if offered ≥ expected
     - 0.6 if offered ≥ 70% of expected
     - 0.2 otherwise
   
   - **Experience Match** (10% weight)
     - 1.0 if student meets requirement
     - 0.6 if close (within 0.5 years)
     - 0.2 otherwise

5. **Final Score**
   ```
   final = 0.55 × semantic_similarity + 
           0.20 × location_score + 
           0.15 × stipend_score + 
           0.10 × experience_score
   ```

6. **Explainability**
   - Generate human-readable reasons for each score component

### Example Output

```
Rank 1
[Internship description...]
Score: 87.45 %
- High skill match
- Preferred location
- Meets Expectation
- Experience matches requirement
```

## Usage

### Prerequisites

```bash
pip install numpy pandas faiss-cpu sentence-transformers
```

For GPU acceleration (optional):
```bash
pip install faiss-gpu
```

### Step 1: Build Index (One-time Setup)

```bash
python offline_build_index.py
```

**Output**: 
- `internships.faiss` - FAISS index
- `internship_vectors.npy` - Vector embeddings

### Step 2: Rank Internships

Edit the `student_text` variable in `online_ranker.py`:

```python
student_text = """Skills: Python, Machine Learning, NumPy, Pandas
Projects: CNN image classifier
Experience: 3 months internship
Location Preference: NewYork"""
```

Then run:
```bash
python online_ranker.py
```

## Scoring Weights Breakdown

| Component | Weight | Purpose |
|-----------|--------|---------|
| Semantic Similarity | 55% | Core skill/task alignment |
| Location Match | 20% | Geographic preferences |
| Stipend Match | 15% | Compensation alignment |
| Experience Match | 10% | Experience requirements |

## Data Format

### Input CSV (`internship_data_modified_v2.csv`)

Column: `raw_data` (string with structured text)

Example:
```
"We are [Company]. Job title: [Title]. Job description: [Description]. 
Skills: [Skills]. Location: [Location]. Stipend: [Stipend]. Experience: [Experience]"
```

### Section Extraction

Both student and internship data are parsed for:
- `skills` / `required skills`
- `projects` / `responsibilities` / `tasks`
- `experience`
- `location` / `location preference`
- `stipend` / `expected_stipend`

## Performance Notes

- **Indexing**: ~10-50ms per batch (depends on BATCH_SIZE)
- **Search**: <1ms for K=300 FAISS search (CPU)
- **Ranking**: ~100-500ms for final scoring (100-300 candidates)
- **Total latency**: ~1-2 seconds for full ranking pipeline

## Future Improvements

- [ ] Add **learning-to-rank** (LTR) models for dynamic weight optimization
- [ ] Implement **active learning** to improve weights based on student feedback
- [ ] Add **filter before rank**: pre-filter by experience/location before scoring
- [ ] Support **user feedback loop**: track accepted/rejected recommendations
- [ ] Build **web API** (FastAPI) for production deployment
- [ ] Add **caching** for frequently searched profiles
- [ ] Support **batch recommendations** for multiple students
- [ ] Integrate **real-time updates** to internship index without rebuilding

## Diversity & De-duplication

The recommender includes a simple diversity mechanism to avoid returning many results from the same employer in the top-K recommendations. This improves user experience by exposing students to a variety of employers while still prioritizing high-quality matches.

- **Approach**: a configurable penalty is applied to the score of internships from companies that have already appeared in the ranked list. The penalty decays multiplicatively for each repeated appearance (default decay = 0.75).
- **Effect**: First item from a company keeps its original score, the second is multiplied by 0.75, the third by 0.75^2, etc. This produces a balance between relevance and variety.

### How to tune

- To make results more diverse, lower the `penalty_decay` (e.g. 0.5 for stronger penalty).  
- To make results more tolerant of repeats, raise the `penalty_decay` (e.g. 0.9 for weaker penalty).  
- Optionally set `max_per_company` to hard-limit the number of results per employer.

The function implementing this behavior is `apply_diversity_penalty()` in `online_ranker.py` and is enabled by default in the example workflow.

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Vector operations |
| `pandas` | Data loading/manipulation |
| `faiss` | Fast similarity search |
| `sentence-transformers` | Semantic embeddings |
| `re` | Text parsing |

## Author

Aaryan Rathod  
GitHub: [aaryanrathod](https://github.com/aaryanrathod)

## License

MIT