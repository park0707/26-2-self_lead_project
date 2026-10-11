import os

print("Running RTMP extraction...")
os.system("python d:/extract_rtmp_features.py")

print("Running eval...")
os.system("python d:/eval_rtmp_models.py")

print("Generating report...")
os.system("python d:/make_report.py")

print("Done.")
