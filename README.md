mart Resume Intelligence: AI-Powered Career Analytics
The Smart Resume Intelligence system is a high-performance, automated screening tool designed to provide objective alignment scores between professional profiles and industry-standard roles. By integrating Deep Learning for contextual intent and Statistical NLP for keyword auditing, it offers a "human-like" review process in seconds.

🎯 Project Overview
In a competitive job market, standard keyword-matching ATS (Applicant Tracking Systems) often fail. This project addresses this by:

Contextual Understanding: Moving beyond "exact match" to understand professional synonyms and intent.

Transparent Scoring: Providing a breakdown of match probability versus raw keyword similarity.

Interactive UI: Allowing candidates to tune evaluation sensitivity to see how they rank for different hiring strictness levels.

🚀 Key Features & Functionality
1. Zero-Shot Role Classification
Utilizes the facebook/bart-large-mnli transformer model. This allows the system to classify a resume into roles like "ML Engineer" or "Cybersecurity" even if the specific title is missing, by analyzing the semantic weight of the experience.

2. Hybrid Similarity Engine
Combines the best of two worlds:

Vectorization: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to weigh the significance of technical skills.

Mathematical Alignment: Calculates Cosine Similarity scores to verify keyword density against industry benchmarks.

3. Dynamic Results Dashboard
Role Match Probabilities: Interactive horizontal bar charts powered by Plotly.

Sensitivity Thresholding: A real-time slider to adjust the "strictness" of the AI's success/warning criteria.

Skillset Deep Dive: A tabbed interface providing granular feedback on required vs. missing keywords.
