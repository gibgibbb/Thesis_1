# Crash Course: Logistic Regression for Fire Ignition Prediction

## Part 1: What is Logistic Regression?

### The Basic Idea
Imagine you have a cell on the grid and you want to answer: **"Will this cell catch fire? Yes or no?"**

You have information about the cell (features):
- Is there a building? (building_presence)
- How steep is the ground? (slope_risk)
- How many neighbors are burning? (neighbor_burning_count)
- etc.

Logistic Regression takes all these features, multiplies each by a "weight" (coefficient), adds them up, and squishes the result into a probability between 0 and 1.

### The Math (Simple Version)

**Step 1: Add up the weighted features**
```
score = (w1 × slope_risk) + (w2 × proximity_risk) + (w3 × building_presence) + ... + bias
```

For our trained model (v2), this looks like:
```
score = (0.23 × slope) + (0.32 × proximity) + (0.81 × building) + (0.0 × wind_speed)
      + (0.0 × wind_sin) + (0.0 × wind_cos) + (0.64 × neighbors) + (-0.48 × composite)
      + bias
```

**Step 2: Squish into probability using the sigmoid function**
```
probability = 1 / (1 + e^(-score))
```

The sigmoid function converts any number into 0-1:
- score = -∞  →  probability ≈ 0  (definitely won't ignite)
- score = 0   →  probability = 0.5 (coin flip)
- score = +∞  →  probability ≈ 1  (definitely will ignite)

```
Probability
1.0 |                        ___________
    |                   ____/
    |                  /
0.5 |- - - - - - - - -/- - - - - - - - -
    |                /
    |           ____/
0.0 |__________/
    +------------------------------------→ Score
         negative        0        positive
```

**Step 3: Decision**
- If probability > 0.5 → predict "Ignited" (1)
- If probability ≤ 0.5 → predict "Not Ignited" (0)

### What the Coefficients Mean

Each coefficient tells you: **"How much does this feature push toward ignition?"**

| Coefficient | Meaning |
|:-----------:|---------|
| Positive (+0.81) | Higher values of this feature → MORE likely to ignite |
| Negative (-0.48) | Higher values of this feature → LESS likely to ignite |
| Zero (0.00) | This feature has NO effect on prediction |
| Larger absolute value | Stronger influence |

Our model learned:
- `building_presence = +0.81` → buildings are more likely to catch fire (makes sense!)
- `neighbor_burning_count = +0.64` → more burning neighbors = more likely (makes sense!)
- `slope_risk = +0.23` → steeper terrain = more likely (makes sense — fire goes uphill)
- `wind_speed = 0.00` → wind has no effect (because it's the same value for every row)

### How is it Different from Random Forest?

| | Logistic Regression | Random Forest |
|---|---|---|
| **How it works** | One equation with weights | Hundreds of decision trees vote |
| **Output** | Smooth probability curve | Average of many yes/no votes |
| **Interpretability** | Easy — you can read the coefficients | Harder — it's a "black box" |
| **Handles nonlinear patterns** | No (linear only) | Yes (trees can capture complex patterns) |
| **Speed** | Very fast | Slower |
| **Overfitting risk** | Low | Higher (but manageable) |

This is exactly why the thesis compares both: LR is the simple, interpretable baseline; RF is the more powerful but less transparent alternative.

---

## Part 2: Why Does `class_weight='balanced'` Matter?

Our dataset has way more "Not Ignited" (0) than "Ignited" (1). Without balancing:

```
Model thinks: "If I just say NO to everything, I'm right 83% of the time!"
Result: Never predicts fire. Useless.
```

`class_weight='balanced'` tells the model: "Treat each positive example as if it's worth 5x more than a negative." This forces the model to actually learn to detect fire instead of being lazy.

---

## Part 3: StandardScaler — Why Scale Features?

Our features have different ranges:
- `building_presence`: 0 or 1
- `wind_speed`: always 10.0
- `neighbor_burning_count`: 0 to 7

LR treats all features equally in its equation. If one feature has big numbers, it dominates just because of scale, not because it's more important.

StandardScaler normalizes everything to mean=0, std=1, so all features compete fairly.

We wrapped it in a **Pipeline** (StandardScaler → LogisticRegression) so the scaling happens automatically when `predict_proba()` is called. This is important because `automata_engine.py` just calls `model.predict_proba(raw_features)` — it doesn't know about scaling.

---

## Part 4: The Multicollinearity Problem (composite_flammability)

### What is Multicollinearity?
When two features contain overlapping information, the model can't tell which one deserves credit. It splits the weight unpredictably between them.

### Our Case
Look at how `composite_flammability` is computed (from `feature_pipeline.py`):
```python
composite_flammability = building_presence × (0.5 + 0.3 × slope_risk + 0.2 × proximity_risk)
```

It literally IS a combination of `building_presence`, `slope_risk`, and `proximity_risk`. So we're feeding the model:
1. `building_presence` (raw)
2. `slope_risk` (raw)
3. `proximity_risk` (raw)
4. `composite_flammability` (a formula of 1, 2, and 3 above)

That's like telling someone the same information twice in different words.

### What Happens
The model sees:
- "building_presence is useful!" → gives it +0.81
- "composite_flammability also correlates with ignition!" → but wait, the signal is already captured by building_presence...
- Model: "I'll give composite_flammability a NEGATIVE weight (-0.48) to compensate for the overlap"

This is called **coefficient instability**. The individual coefficients become unreliable, even though the overall prediction still works.

### Visual Example
Imagine two friends (Feature A and Feature B) always arrive at a party together:

```
Reality:  "Fire happens when buildings are present and flammable"
           building_presence = important
           composite_flammability = important (because it includes building_presence)

Model sees: Both features go up and down together
Model does: Gives A a big positive weight, gives B a negative weight
            The positives and negatives cancel out to roughly the right answer
            But individually, the coefficients look wrong
```

### Is This a Problem for Our Model?
- **For predictions:** Not really. The model still predicts correctly because the positive and negative weights balance out.
- **For interpretation:** Yes. You can't point at the -0.48 coefficient and say "composite_flammability reduces fire risk." That would be a wrong conclusion.
- **For the thesis:** You should mention this as a known limitation. The feature was designed for the CA engine, not for ML interpretation.

### How to Verify It's Multicollinearity
If you remove `composite_flammability` and retrain, you'd expect:
- `building_presence`, `slope_risk`, `proximity_risk` coefficients all increase
- Overall model performance stays about the same

---

## Part 5: Understanding Our Evaluation Metrics

### Confusion Matrix
```
                    Predicted
                 NO        YES
Actual NO    [ TN=353    FP=149 ]    ← 149 false alarms
Actual YES   [ FN=36     TP=64  ]    ← 36 missed fires
```

- **TP (True Positive):** Model said fire, actually fire → 64
- **FP (False Positive):** Model said fire, actually no fire → 149 (false alarms)
- **TN (True Negative):** Model said no fire, actually no fire → 353
- **FN (False Negative):** Model said no fire, actually fire → 36 (missed fires!)

### Metrics from the Matrix
- **Precision (0.30):** Of all cells the model flagged as "will ignite," only 30% actually did. Lots of false alarms.
- **Recall (0.64):** Of all cells that actually ignited, the model caught 64%. Missed 36%.
- **F1 (0.41):** Harmonic mean of precision and recall. Balances both.
- **ROC AUC (0.73):** How well the model ranks igniting cells higher than non-igniting. 1.0 = perfect ranking, 0.5 = random guessing.

### What the Thesis Wants
- **Recall ≥ 0.80:** Missing fires is dangerous. For fire safety, we'd rather have false alarms than miss real fires.
- **F1 ≥ 0.80:** But we also don't want TOO many false alarms.

### Where We Stand
```
Target:  Recall ≥ 0.80,  F1 ≥ 0.80
Current: Recall = 0.64,  F1 = 0.41
Gap:     Need improvement
```

---

## Summary: What We Know So Far

1. LR is a simple equation: weighted sum → sigmoid → probability
2. Our model correctly identifies building_presence and neighbor_burning_count as the main fire drivers
3. The perfect v1 scores were fake — caused by including millions of irrelevant cells
4. The realistic v2 scores (F1=0.41) are below thesis targets but honest
5. composite_flammability has a weird negative coefficient due to multicollinearity — not a bug, but a known statistical artifact
6. Wind features contribute nothing because they're constant across the dataset
7. More diverse training data (varied wind, more ignition scenarios) would help the most
