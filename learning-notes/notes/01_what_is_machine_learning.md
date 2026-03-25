# 📚 Notes 01 — What is Machine Learning?

---

## The Core Idea

Machine learning is teaching a computer to find patterns **from examples** rather than writing explicit rules.

> **Analogy:** Teaching a child to recognise apples. You don't write a rulebook — you just show them hundreds of apples and say *"apple... apple... apple..."* and their brain figures out the pattern.

Instead of: `if red AND round AND has stalk → apple`
Machine learning does: show 10,000 apples → let the computer figure out the pattern itself.

---

## The Three Ingredients

Every machine learning project needs exactly three things:

| Ingredient | What It Is | In Your Project |
|---|---|---|
| **Data** | The examples | 3,900 stars |
| **Features** | What you know about each example | 37 measurements per star |
| **Labels** | The answer you're trying to learn | CONFIRMED / FALSE POSITIVE / CANDIDATE |

---

## The Doctor Analogy

A doctor diagnosing patients uses:
- **Features:** blood pressure, temperature, age, symptoms
- **Label:** healthy / sick / needs surgery

A doctor who has seen 10,000 patients gets good at spotting patterns.
Machine learning does the same — but with maths instead of experience.

---

## Types of Learning

### Supervised Learning
- You give the model **both features AND the correct label** during training
- It learns the relationship between them
- Both pipelines in this project use supervised learning

### Classification
- A specific type of supervised learning
- The answer is a **category**, not a number
- *"Is this a planet, false positive, or candidate?"* = classification
- *"How many days until this planet transits again?"* = regression (different type)

---

## The Training / Validation / Test Split

This is one of the most important concepts in machine learning.

```
All Data (3,900 stars)
        │
        ├── Training Set (70% = ~2,730 stars)
        │     └── Model learns patterns here
        │
        ├── Validation Set (15% = ~585 stars)
        │     └── Used to tune the model during training
        │         Model never officially "trained" on these
        │
        └── Test Set (15% = ~585 stars)
              └── The real exam — model sees these for the FIRST TIME
                  This is where we measure TRUE performance
```

### Why Not Use All Data for Training?

Because the model would **memorise** the answers instead of learning the pattern.

> **Analogy:** A student who memorises every practice question word-for-word. They'd fail on any new question they haven't seen before.

This is called **overfitting** — too good on training data, terrible on new data.

### Stratified Split
In your project, the split is **stratified** — meaning each split has the same proportion of each class.

So if 59% of all data is CONFIRMED, then training, validation, AND test sets each have ~59% CONFIRMED. This ensures the model sees a fair representation of all classes.

---

## What Happens During Training?

```
Model makes a guess
        ↓
Checks if it's right (compares to label)
        ↓
Calculates how wrong it was (the "loss")
        ↓
Adjusts itself to be slightly less wrong
        ↓
Repeat millions of times across thousands of examples
```

> **Analogy:** Learning to ride a bike. You wobble, fall, correct, try again. Eventually it becomes automatic.

---

## How Do We Know If The Model Is Good?

After training, we test on the test set (data it has never seen) and measure:

| Metric | What It Means | Simple Analogy |
|---|---|---|
| **Accuracy** | % of predictions that were correct | Exam score |
| **F1 Score** | Balances precision and recall — better for imbalanced data | How well you find the right answers without guessing randomly |
| **ROC-AUC** | How well the model ranks examples (1.0 = perfect, 0.5 = random) | Can the model correctly rank "more likely planet" vs "less likely planet"? |
| **Cohen's κ** | Accuracy adjusted for chance — more honest than raw accuracy | Exam score minus the marks you'd get from random guessing |

---

## Where Your Project Fits

```
3,900 stars with known dispositions (labels)
              ↓
    37 features measured per star
              ↓
  Machine learning models learn the pattern
              ↓
  Given a NEW unknown star → predict:
  Confirmed Planet / False Positive / Candidate
```

---

## Key Terms Glossary

| Term | Meaning |
|---|---|
| **Machine Learning** | Teaching computers to find patterns from examples |
| **Supervised Learning** | Training with both features AND labels |
| **Classification** | Predicting which category something belongs to |
| **Feature** | A measurable property used as input to the model |
| **Label** | The correct answer the model is trying to predict |
| **Training** | The process of the model learning from data |
| **Overfitting** | Model memorises training data but fails on new data |
| **Test Set** | Held-out data used to measure true model performance |
| **Accuracy** | % of correct predictions |
| **F1 Score** | Metric that balances finding correct cases vs missing them |
| **ROC-AUC** | How well the model ranks examples by likelihood |

---

*Notes from MSc Data Science Dissertation — Exoplanet Candidate Vetting*
