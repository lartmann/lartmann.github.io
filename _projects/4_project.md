---
layout: page
title: ECG classification
description: Training and evaluation of machine learning models for ECG classification using nested cross-validation.
img: assets/img/ecg.png
importance: 3
category: individual
---

# Overview

## Context

ECG signals are used to assess the health of the heart. However, especially for long term monitoring it is very time consuming to analyze all the signals. With an average heart rate of 80, a single day creates an amount of almost 7 million heart beats which is unfeasible to manually analyze in practice. Therefore, machine learning can help to find abnormal heart beats which might indicate diseases of the heart or cardio vascular system.

## Evaluation Criteria

1. **Balanced Accuracy** is used because the classes are unequally distributed. For example, the class of the normal heart beats is a lot larger than the other classes. However, correctly classifying an abnormal heart beat might be more important.
2. **F1 weighted** is a robust metric for imbalanced datasets and balances precision and recall across all classes.

## The Data

The dataset is the MIT-BIH Arrhythmia Database which is downloaded as zip file from the [official website](https://physionet.org/content/mitdb/1.0.0/)<sup>[1]</sup><sup>[2]</sup>. After unpacking, the data is processed into a pandas Dataframe.

The dataset consists of 48 ~30 min long two - channel ambulatory ECG recordings which was collected from 47 subjects. A two-channel ambulatroy ECG is a portable device used to monitor and record a patients heart activity over an extended period. The recordings have a resolution of 360 samples per second. Each record as annotated by at least two independent cardiologists.
The goal is to predict these annotation and to classify the individual heart beat. Therefore, the values at different time points are the data while the annotation is the target to be predicted.

This dataset is suitable for the task described above for several reasons:

1. The data is from a two-channel ambulatroy ECG which is typically used for long term monitoring.
2. Each heart beat is annotated.
3. The dataset quite big with more than 100,000 heart beats.

This means, a model resulting from this project could ideally classify a heart beat from a two-channel ambulatroy ECG.

**Note:** Detecting the beat itself is not part of this project. In practice, it would be a necessary step to first identify the peaks of heart beats in the ECG signal.

---

[1]: Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

[2]: Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)

---

# The Project

> Note: Below is only part of the implementation. The whole code with all classes and functions can be found in [this repository](https://github.com/lartmann/ECG-Classification).

{::nomarkdown}

<div style="background-color: white; color: black; padding: 1em; border-radius: 8px;">
{% assign jupyter_path = "assets/jupyter/classification.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/classification.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}

<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
</div>
{:/nomarkdown}
