---
tags:
  - Marinus
  - exam
  - Notes
  - definition
Created: 2025-01-31
---
This is something we see often, so i thought it would be good to have it in here as well. cause now we can start linking things. 

Have a look at the backlinks down below.

# Common Techniques
- **Min-Max Scaling:** Rescales data to a $[0,1]$ range.
- **Standardization (Z-score):** Rescales data to have a mean of 0 and a standard deviation of 1.
- **Whitening (PCA-based):** De-correlates features and sets covariance to the identity matrix.

# Benefits of normalizing regression labels:
- Makes it easier for the model to <mark style="background: #D2B3FFA6;">learn relationships</mark> between inputs and outputs.
- Helps models with certain activation functions (e.g., sigmoid, tanh) stay within predictable output ranges.
- Can **prevent extreme values** from dominating the learning process.
- Standardizing labels ensures the model outputs **values within a reasonable range**, preventing prediction drift.
