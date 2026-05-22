import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Improved NEP pillars
nep_pillars = {
    "Research & Innovation": "research innovation idea creativity invention design thinking critical thinking research paper publication patent problem solving",
    
    "Skill Development & Experiential Learning": 
"hands-on workshop practical training implementation project coding programming lab software tools experiment skill development real application labview matlab ansys simulation development",
    
    "Emerging Technologies & Future Readiness": "artificial intelligence machine learning deep learning iot internet of things robotics automation data science digital transformation industry 4.0",
    
    "Industry-Academia Integration": "industry internship corporate real world experience startup entrepreneurship business career development professional exposure industrial visit",
    
    "Multidisciplinary Learning": "interdisciplinary cross disciplinary integration mathematics physics engineering management combined subjects applied learning",
    
    "Holistic Development": "leadership communication teamwork personality development ethics cultural extracurricular co curricular soft skills life skills",
    
    "General": "seminar lecture awareness introduction fundamentals basic session orientation"
}

# Prepare embeddings
pillar_names = list(nep_pillars.keys())
pillar_texts = list(nep_pillars.values())
pillar_embeddings = model.encode(pillar_texts, convert_to_tensor=True)

# Load your Excel
df = pd.read_excel("C:/Users/charusat/Desktop/IQAC/CHARUSAT NEP Webpage/WEBSITE/NEP Event.xlsx")

# Combine text
df["Combined"] = df["Event Title"].fillna("") + " " + df["Event Objective"].fillna("")

# Classification function (Top 3)
def classify_event(text):
    if not text.strip():
        return "General", 0.0, ""

    # 🔹 Extract keywords automatically
    keywords = kw_extractor.extract_keywords(text)
    extracted_words = " ".join([kw[0] for kw in keywords]).lower()

    # 🔹 Combine original text + keywords
    enhanced_text = text + " " + extracted_words

    # 🤖 BERT on enhanced text
    emb = model.encode(enhanced_text, convert_to_tensor=True)
    scores = util.cos_sim(emb, pillar_embeddings)[0]

    top_idx = scores.argsort(descending=True)[:3]
    top_results = [(pillar_names[i], float(scores[i])) for i in top_idx]

    best_pillar = top_results[0][0]
    best_score = round(top_results[0][1], 3)

    # Show extracted keywords (VERY useful)
    keywords_str = ", ".join([kw[0] for kw in keywords])

    return best_pillar, best_score, keywords_str
    
# Apply model
results = df["Combined"].apply(classify_event)

df["NEP Pillar"] = results.apply(lambda x: x[0])
df["Confidence"] = results.apply(lambda x: x[1])
df["Top 3 Predictions"] = results.apply(lambda x: x[2])

# Confidence Level
df["Confidence Level"] = df["Confidence"].apply(
    lambda x: "High" if x > 0.75 else "Medium" if x > 0.55 else "Low"
)

# Save output
output_path = "C:/Users/charusat/Desktop/NEP_BERT_Output.xlsx"
df.to_excel(output_path, index=False)

print("Done! File saved at:", output_path)