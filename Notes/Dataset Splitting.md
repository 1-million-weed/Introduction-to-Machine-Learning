---
tags:
  - Notes
  - Marinus
  - LLM
  - _FirstPass
Date: 2024-12-11
---

### Notes on Test, Train, and Validation Splits

#### **Definitions:**

>[!def] **Training Set** 
>Used to train the model; the optimization process runs here. Metrics computed are called training metrics.

>[!def] **Validation Set** 
>Used for making decisions during training, like hyper-parameter tuning. It is independent of the training and testing sets.

>[!def] **Test Set** 
>An independent dataset for evaluating the model's final performance. Metrics computed are called testing metrics.

#### **Key Practices:**

- The training, validation, and testing sets should be mutually exclusive with no overlapping samples.
- Validation is only needed during the model development process. After tuning, the model can be retrained on the combined training and validation sets (train-val dataset) before final evaluation on the test set.

#### **Typical Split Ratios:**

- Training: 80%, Testing: 20%
- Training: 80%, Validation: 10%, Testing: 10%
- Training: 70%, Validation: 15%, Testing: 15%
- Training: 60%, Validation: 20%, Testing: 20%

#### **Splitting Strategies:**

1. Random Splits: Assuming no duplicated samples in the dataset, random splits are common.
2. Specific Scenarios:
   - **User Data:** Ensure different individuals are assigned to different splits.
   - **Time Series:** Assign entire time sequences to either train or test, never split a sequence between them.
   - **Objects:** If the method is object-independent, do not split objects across train and test sets.
   - **Data Augmentation or Oversampling:** Apply only on the training set after the splits.

#### **Preventing Leakage:**

>[!def] **Leakage Definition** 
>Occurs when training set data or related information is present in validation or testing datasets, leading to overly optimistic evaluation metrics.

- **Prevention:** Ensure data independence, exclude features that reveal the target label, and carefully handle data augmentation.

#### **Cross-Validation:**

- **K-Fold Cross-Validation:**

  - Splits data into `k` equally sized folds.
  - Each fold serves as the test set once, while the remaining `k-1` folds form the training set.
  - Requires training `k` models and provides a mean and standard deviation of results.
  - *Insert relevant slide: K-Fold Cross Validation Workflow*

- **Leave-One-Out Cross-Validation (LOO-CV):**

  - A special case of K-Fold where `k = N` (number of samples).
  - Each model tests on a single sample, evaluating generalization for every data point.
  - *Insert relevant slide: Leave-One-Out CV Example*

- **Nested Cross-Validation:** Used for hyper-parameter tuning while assessing the generalization of the final model.

![[Pasted image 20241211174917.png]]

#### **Typical Workflow:**

1. Split the dataset into training, validation, and test sets.
2. Train the model on the training set.
3. Tune hyper-parameters using the validation set.
4. Evaluate the final model on the test set.
5. Optionally retrain using the train-val set for the final evaluation.

- *Insert relevant slide: Train/Validation/Test Workflow*

#### **Summary:**

- Proper splitting of datasets ensures robust model evaluation.

- Cross-validation provides a reliable way to assess model performance while minimizing overfitting.

- Prevent leakage at all costs to maintain valid evaluation metrics.

- Use appropriate splitting strategies for the type of data being handled (e.g., time series, objects).

- *Insert relevant slide: Types of Data Splits*

- *Insert relevant slide: Leakage Examples and Prevention*

---

Let me know if you want further refinement or additional sections!

