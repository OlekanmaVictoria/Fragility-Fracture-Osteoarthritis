import joblib
model = joblib.load("hybrid_model.pkl")
print("✅ Loaded without error.")

import os
print(os.path.getsize('hybrid_model.pkl'))