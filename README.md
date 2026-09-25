# Employee Attrition Analysis

A People Analytics project exploring whether routinely collected employee data can be used to model attrition risk.

## Approach

The project has two stages:

### 1. IBM HR Analytics dataset

I trained a **logistic regression** model on 1,470 employee records.

```python
LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)
```

The model assigns a coefficient (weight) to each feature. The coefficients are combined with the employee's feature values to calculate a **log-odds of attrition**, which is then converted into a probability using the sigmoid function:

$$
P(\text{attrition}) =
\frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + ... + \beta_nx_n)}}
$$

So the weights were **learned by logistic regression**, rather than manually assigned.

For example, in this model:

* Overtime: **+0.808**
* Marital Status: **+0.531**
* Department: **+0.322**
* Job Satisfaction: **−0.267**
* Environment Satisfaction: **−0.359**
* Years With Current Manager: **−0.131**

These coefficients represent model associations, **not causal effects**.

### 2. Synthetic longitudinal dataset

I then generated a synthetic monthly dataset containing **1,365 employee-month observations and 68 simulated departures**.

Here, the simulated attrition probability was deliberately generated from a predefined weighted formula:

$$
risk =
-3.0
+0.9(O)
+0.5(C_t)
+0.7(C_j)
+0.6(B)
-0.5(V)
-0.35(J_s-3)
-0.30(E_s-3)
-0.12(Y_m)
+0.15(P_r-3)
$$

where the variables represent overtime, complaints, behavioural change, volunteering, satisfaction, years with manager and performance.

The probability was then:

$$
P(\text{leave}) = sigmoid(risk)
$$

The subsequent logistic regression **did not receive this true simulated risk variable**. It had to infer the underlying relationships from the observable features.

## Results

| Model                  | Accuracy | Attrition Recall | Attrition Precision |
| ---------------------- | -------: | ---------------: | ------------------: |
| IBM dataset            |    71.1% |              74% |                 32% |
| Synthetic longitudinal |    66.7% |              57% |                  9% |

The project is therefore a demonstration of **how attrition scores can be constructed and investigated**, rather than a production-ready employee prediction system.

## Tools

Python · Pandas · NumPy · scikit-learn · Matplotlib · Streamlit · Jupyter
