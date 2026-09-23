---
title: Phishing URL Detector
emoji: 🎣
colorFrom: indigo
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Judges a link from the string alone. Research demo, not a security product.
---

# Phishing URL detector

Judges a link from **the string alone**: no network call, no page fetch.

- **Model**: [valezion7/url-phishing-detector](https://huggingface.co/valezion/url-phishing-detector)
- **Code, data sources and the diary of eight ways the model fooled me**:
  [github.com/valezion7/url-phishing-detector](https://github.com/valezion7/url-phishing-detector)

On an independently collected holdout (today's verified phishing against links people posted
today) it reaches ROC-AUC 0.948: blocking 0.9% of good links it catches 53% of fresh phishing,
and 93% of URLhaus malware URLs — a threat it never saw in training.

**This is a research demo, not a security product.** It is blind to scams hosted inside
legitimate compromised sites, and "no signal" is never a guarantee that a link is safe.
