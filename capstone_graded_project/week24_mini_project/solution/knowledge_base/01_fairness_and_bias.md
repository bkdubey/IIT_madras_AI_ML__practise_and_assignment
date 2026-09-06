# Fairness and Bias

Fairness in AI means evaluating whether a model treats different user groups equitably and whether its predictions or recommendations are consistent with accepted societal norms. A fair system should not produce systematically worse outcomes for a protected group unless there is a legitimate and documented reason.

Bias often enters through data collection, labels, feature design, or the historical context that shaped the training examples. For example, if a hiring model learns from past hiring decisions that underrepresent women or marginalized communities, it may reproduce those patterns even when the training data is stored without explicit protected attributes.

To mitigate fairness issues, teams should measure disparate impact, calibrate model outputs, and test performance across demographic segments. It is also important to document which group-level metrics are monitored and what thresholds trigger intervention or retraining.

Source: Responsible AI Handbook, internal policy summary.
