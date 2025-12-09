from test import scorer
scores=scorer("en_core_web_sm")

print("=== OVERALL NER METRICS (en_core_web_sm)===")
print(f"Precision: {scores['ents_p']:.3f}")
print(f"Recall:    {scores['ents_r']:.3f}")
print(f"F1 Score:  {scores['ents_f']:.3f}")
print(f"Number of categories:  {len(scores["ents_per_type"].items()):.3f}")



scores=scorer("model-best")

print("=== OVERALL NER METRICS (fine_tuned)===")
print(f"Precision: {scores['ents_p']:.3f}")
print(f"Recall:    {scores['ents_r']:.3f}")
print(f"F1 Score:  {scores['ents_f']:.3f}")
print(f"Number of categories:  {len(scores["ents_per_type"].items()):.3f}")
