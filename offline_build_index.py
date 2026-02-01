import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import re
BATCH_SIZE = 64

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\Internship Recommendation\trying_to_be_fast.py\internship_data_modified_v2.csv")
internship_data = df["raw_data"].dropna().astype(str).tolist()

header_map_internship = {
    "required skills:": "skills",
    "skills:": "skills",
    "responsibilities:": "tasks",
    "job title:": "title",
    "location:": "location",
    "stipend:": "stipend",
    "experience:": "experience"
}

def extract_sections(text, header_map):
    text = text.lower()
    lines = text.split("\n")
    sections = {v: "" for v in header_map.values()}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        matched = False

        for header, section_name in header_map.items():
            if header in line:
                collected = line.split(header, 1)[1].strip()
                j = i + 1
                while j < len(lines):
                    nxt = lines[j].strip()
                    if nxt == "" or any(h in nxt for h in header_map):
                        break
                    collected += " " + nxt
                    j += 1
                sections[section_name] = collected
                i = j
                matched = True
                break

        if not matched:
            i += 1

    if not sections.get("skills"):
        match = re.search(r"(skill|skills)\s*:\s*(.+)", text)
        if match:
            sections["skills"] = match.group(2)

    return sections

def normalize(v):
    norm = np.linalg.norm(v)   # finding the length of the vector -> square-root(v1^2 + v2^2 + ....vn^2)
    if norm == 0:
        return v
    return v / norm

#wieghts
W_SKILLS = 0.6
W_TASKS = 0.3
W_EXP = 0.1

final_vectors = []

for start in range(0, len(internship_data), BATCH_SIZE):
    batch_texts = internship_data[start : start + BATCH_SIZE]

    skills_texts = []
    tasks_texts = []
    exp_texts = []

    for text in batch_texts:
        sec = extract_sections(text, header_map_internship)
        skills_texts.append(sec["skills"])
        tasks_texts.append(sec["tasks"])
        exp_texts.append(sec["experience"])

    skill_embs = model.encode(skills_texts, convert_to_numpy=True, show_progress_bar=False)
    task_embs  = model.encode(tasks_texts,  convert_to_numpy=True, show_progress_bar=False)
    exp_embs   = model.encode(exp_texts,    convert_to_numpy=True, show_progress_bar=False)

    for s in skill_embs:
        final_vectors.append(normalize(s))


matrix = np.vstack(final_vectors).astype("float32")

index = faiss.IndexFlatIP(matrix.shape[1])
index.add(matrix)

faiss.write_index(index, "internships.faiss")
np.save("internship_vectors.npy", matrix)

print("OFFLINE BUILD COMPLETE")
print("Total internships indexed:", len(matrix))