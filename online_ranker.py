import numpy as np
import pandas as pd
import faiss
import re
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Point to the correct location of FAISS index and vectors
index_path = r"C:\Users\ASUS\OneDrive\Documents\Internship Recommendation\internships.faiss"
vectors_path = r"C:\Users\ASUS\OneDrive\Documents\Internship Recommendation\internship_vectors.npy"
data_path = r"C:\Users\ASUS\OneDrive\Documents\Internship Recommendation\trying_to_be_fast.py\internship_data_modified_v2.csv"

index = faiss.read_index(index_path)

df = pd.read_csv(data_path)

student_text = """Skills: Python, Machine Learning Python Developer, NumPy, Pandas
Projects: CNN image classifier
Experience: 3 months internship
Location Preference: NewYork"""

internship_data = df["raw_data"].dropna().astype(str).tolist()

internship_active = [True] * len(internship_data)

header_map_student = {
    "skills:": "skills",
    "projects:": "projects",
    "experience:": "experience",
    "location preference:": "location_preference",
    "location:": "location_preference",
    "stipend:": "expected_stipend"
}

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

    if not sections["skills"]:
        match = re.search(r"(skill|skills)\s*:\s*(.+)", text)
        if match:
            sections["skills"] = match.group(2)

    return sections


def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

student_weights = {
    "skills": 0.6,
    "projects": 0.3,
    "experience": 0.1
}

def build_student_query_vector(student_text):
    sec = extract_sections(student_text, header_map_student)

    skills_text = sec.get("skills", "")
    tasks_text  = sec.get("projects", "")   # project ≈ task
    exp_text    = sec.get("experience", "")

    s = model.encode(skills_text)
    t = model.encode(tasks_text)
    e = model.encode(exp_text)

    combined = 0.6 * s + 0.3 * t + 0.1 * e
    return normalize(combined).astype("float32")


def location_match(student_location, internship_location):
    student_location = student_location.split()
    if(internship_location in student_location):
        return 1.0
    elif("relocate" in student_location or "anywhere" in student_location):
        return 0.7
    else:
        return 0.2
    
def stipend_match(student_stipend_text, internship_stipend_text):
    
    student_numbers = re.findall(r'\d+', student_stipend_text.replace(',', ''))
    internship_numbers = re.findall(r'\d+', internship_stipend_text.replace(',', ''))

    if not internship_numbers:
        return 0.2
    
    if not student_numbers:
        return 0.6
    
    std_acceptable_min = int(student_numbers[0])
    offered_min = int(internship_numbers[0])   #taking lower bound

    if offered_min >= std_acceptable_min:
        return 1.0
    elif offered_min >= 0.7 * std_acceptable_min:
        return 0.6
    else:
        return 0.2
    
def parse_experience(exp_text):
    exp_text = exp_text.lower()

    if "fresher" in exp_text or "no experience" in exp_text:
        return 0.0
    
    numbers = re.findall(r'\d+', exp_text)

    if not numbers:
        return 0.0
    
    value = int(numbers[0])

    if "month" in exp_text:
        return value / 12       
    
    return float(value)

def exp_match(student_exp_text, internship_exp_text):
    student_exp = parse_experience(student_exp_text)
    internship_exp = parse_experience(internship_exp_text)

    if student_exp >= internship_exp:
        return 1.0
    elif student_exp + 0.5 >= internship_exp:
        return 0.6
    else:
        return 0.2
    
def extract_company(internship_text):
    """Extract company name from internship text."""
    company_match = re.search(r'we are\s+([^.]+)\.', internship_text, re.IGNORECASE)
    return company_match.group(1).strip() if company_match else "Unknown Company"


def apply_diversity_penalty(results, internship_data, penalty_decay=0.75, max_per_company=3):
    """
    Apply diversity penalty to reduce duplicate companies in recommendations.
    
    Args:
        results: List of ranked results
        internship_data: Raw internship data list
        penalty_decay: Factor to multiply scores by for repeated companies (0.0-1.0)
                      Lower = stronger penalty. Default 0.75 = 25% penalty
        max_per_company: Max internships from same company to show (None = no limit)
    
    Returns:
        Re-ranked results with diversity penalty applied
    """
    company_count = {}
    diversified_results = []
    
    for r in results:
        idx = r["internship_id"]
        internship_text = internship_data[idx]
        company = extract_company(internship_text)
        
        # Count how many times this company has appeared
        times_seen = company_count.get(company, 0)
        company_count[company] = times_seen + 1
        
        # If max_per_company is set, skip if exceeded
        if max_per_company and times_seen >= max_per_company:
            continue
        
        # Apply penalty based on how many times company appeared
        penalty_multiplier = penalty_decay ** times_seen
        
        # Create penalized copy
        r_penalized = r.copy()
        r_penalized["final_score"] = r["final_score"] * penalty_multiplier
        r_penalized["company_name"] = company
        r_penalized["company_count"] = times_seen + 1
        r_penalized["original_score"] = r["final_score"]
        
        diversified_results.append(r_penalized)
    
    # Re-sort by penalized scores
    diversified_results = sorted(
        diversified_results,
        key=lambda x: x["final_score"],
        reverse=True
    )
    
    return diversified_results


def explainability(scores):

    reasons = []

    if(scores["semantic"] >= 0.80):
        reasons.append("High skill match")
    elif(scores["semantic"] >= 0.65 and scores["semantic"] < 0.80):
        reasons.append("Good Skill Match")
    elif(scores["semantic"] < 0.65):
        reasons.append("Partial Skill Match")

    if(scores["location"] == 1.0):
        reasons.append("Preferred location")
    elif(scores["location"] == 0.7):
        reasons.append("Relocation / Remote Possible")
    elif(scores["location"] == 0.2):
        reasons.append("Location Mismatch")
    
    if(scores["stipend"] == 1.0):
        reasons.append("Meets Expectation")
    elif(scores["stipend"] == 0.6):
        reasons.append("Slightly below expectation")
    elif(scores["stipend"] == 0.2):
        reasons.append("Low / undisclosed stipend")
    
    if(scores["experience"] == 1.0):
        reasons.append("Experience matches requirement")
    elif(scores["experience"] == 0.6):
        reasons.append("Slightly Below Requirement")
    elif(scores["experience"] == 0.2):
        reasons.append("Experience gap")

    return reasons

internship_weights = {
    "skills": 0.6,
    "tasks": 0.3,
    "experience": 0.1
}

def build_internship_master_vector(sections):
    skill_emb = model.encode(sections.get("skills", ""))
    task_emb  = model.encode(sections.get("tasks", ""))
    exp_emb   = model.encode(sections.get("experience", ""))

    combined = (
        internship_weights["skills"] * skill_emb +
        internship_weights["tasks"] * task_emb +
        internship_weights["experience"] * exp_emb
    )

    return normalize(combined).astype("float32")


def final_score(student_text, internship_text):
    
    student_sections = extract_sections(student_text, header_map_student)
    internship_sections = extract_sections(internship_text, header_map_internship)

    student_vec = build_student_query_vector(student_text)

    internship_vec = build_internship_master_vector(internship_sections)

    semantic_similarity = np.dot(student_vec, internship_vec)  

    location_score = location_match(student_sections["location_preference"], internship_sections["location"])
    stipend_score = stipend_match(student_sections["expected_stipend"],internship_sections["stipend"])
    experience_score = exp_match(student_sections["experience"], internship_sections["experience"])

    final_score_value = (0.55 * semantic_similarity) + (0.20 * location_score) + (0.15 * stipend_score) + (0.10 * experience_score)
    
    scores = {
        "semantic": semantic_similarity,
        "location": location_score,
        "stipend": stipend_score,
        "experience": experience_score,
        "final": final_score_value
    }

    return scores


student_sections = extract_sections(student_text, header_map_student)
student_vec = build_student_query_vector(student_text).reshape(1, -1)

K = 300

results = []


scores, indices = index.search(student_vec, K)

flat_indices = indices.reshape(-1).tolist()

results = []

# for idx in flat_indices:
#     if idx == -1:
#         continue
#     if not internship_active[idx]:
#         continue

#     internship_text = internship_data[idx]
#     score_dict = final_score(student_text, internship_text)
#     explanation = explainability(score_dict)

#     results.append({
#         "internship_id": idx,
#         "final_score": score_dict["final"],
#         "explanation": explanation
#     })

for sim, idx in zip(scores[0], indices[0]):
    if idx == -1 or not internship_active[idx]:
        continue

    internship_text = internship_data[idx]

    # only heuristic scores here
    sections = extract_sections(internship_text, header_map_internship)

    location = location_match(student_sections['location_preference'], sections['location'])
    stipend  = stipend_match(student_sections['expected_stipend'], sections['stipend'])
    exp      = exp_match(student_sections['experience'], sections['experience'])

    score_dict = final_score(student_text, internship_text)
    explanation = explainability(score_dict)

    final = (
        0.55 * sim +
        0.20 * location +
        0.15 * stipend +
        0.10 * exp
    )

    results.append({
    "internship_id": int(idx),
    "final_score": float(final),
    "semantic": float(sim),
    "location": location,
    "stipend": stipend,
    "experience": exp,
    "scores": score_dict,
    "explanation": explanation
    })


results = sorted(
    results,
    key=lambda x: x["final_score"],
    reverse=True
)

# Apply diversity penalty to reduce duplicate companies
results = apply_diversity_penalty(results, internship_data, penalty_decay=0.75, max_per_company=None)

# ... (after the sort) ...

def print_recommendations(results, internship_data, top_k=5):
    """Pretty-print top-K recommendations with improved readability."""
    
    print("\n" + "=" * 90)
    print(f"{'TOP INTERNSHIP RECOMMENDATIONS':^90}")
    print("=" * 90)
    
    for rank, r in enumerate(results[:top_k], 1):
        idx = r["internship_id"]
        internship_text = internship_data[idx]
        
        # Extract company and job title if available
        company_match = re.search(r'we are\s+([^.]+)\.', internship_text, re.IGNORECASE)
        title_match = re.search(r'job title:\s+([^.]+)', internship_text, re.IGNORECASE)
        
        company = company_match.group(1).strip() if company_match else "Unknown Company"
        job_title = title_match.group(1).strip() if title_match else "Unknown Position"
        
        # Format score with visual indicator
        score_pct = round(r["final_score"] * 100, 2)
        filled = int(score_pct / 5)
        score_bar = "#" * filled + "-" * (20 - filled)
        
        # Diversity info
        company_count = r.get("company_count", 1)
        diversity_note = f" (Company appearance #{company_count})" if company_count > 1 else ""
        
        # Print recommendation header
        print(f"\n{'-' * 90}")
        print(f"[*] RANK #{rank} | Score: {score_pct:>6.2f}% [{score_bar}]{diversity_note}")
        print(f"{'-' * 90}")
        print(f"Company:     {company}")
        print(f"Position:    {job_title}")
        
        # Print score breakdown
        print(f"\nScore Breakdown:")
        print(f"  - Semantic Match:    {r['scores']['semantic']:.3f}")
        print(f"  - Location Match:    {r['location']:.1f}")
        print(f"  - Stipend Match:     {r['stipend']:.1f}")
        print(f"  - Experience Match:  {r['experience']:.1f}")
        
        # Show penalty info if applied
        if "original_score" in r and r["original_score"] != r["final_score"]:
            original_pct = round(r["original_score"] * 100, 2)
            print(f"\nDiversity Adjustment:")
            print(f"  - Original Score:    {original_pct:.2f}%")
            print(f"  - Penalty Applied:   {(1 - r['final_score']/r['original_score'])*100:.1f}%")
        
        # Print explainability reasons
        print(f"\nWhy this match:")
        for reason in r["explanation"]:
            print(f"  [+] {reason}")
        
        # Print truncated description
        description = internship_text[:800]
        if len(internship_text) > 800:
            description += "..."
        print(f"\nDescription Preview:\n{description}")
    
    print(f"\n{'=' * 90}\n")


# Run the recommendation display
print_recommendations(results, internship_data, top_k=5)