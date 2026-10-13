# Counterfactual Explanations for Insurance

This repository contains the code and supplementary materials for the paper **"From Explaining to Enabling: Actionable Transparency in
Algorithmic Insurance Pricing"**, which investigates counterfactual explanations for insurance pricing models.

Counterfactual explanations (CEs) provide actionable information about how the input characteristics of an individual policyholder could be changed to obtain a desired prediction from a machine learning (ML) model. In insurance, such explanations can help communicate model predictions to policyholders to enhance transparency and identify potential actions to reduce predicted insurance costs.

## Overview

The repository implements and evaluates four CE methods (WhatIf, NICE, MOC and DiCE) for insurance pricing models, with a focus on generating **actionable, plausible and reliable counterfactuals**.

We consider two insurance applications:

1. **Flood insurance** - a severity model applied to NFIP flood insurance claims.
2. **Automobile insurance** - a claim frequency model applied to telematics data.

The CEs are generated for individual observations and evaluated according to several criteria:

* **Actionability**: recommendations involve only mutable features;
* **Proximity**: how close the counterfactual is to the original observation (i.e., the query instance);
* **Plausibility**: how similar the counterfactual is to observations in the data;
* **Reliability**: whether the counterfactual lies in a region of low predictive uncertainty, measured using conformal prediction intervals based on LACP.

## Methods

The repository compares four CE approaches:

* [**WhatIf**](https://arxiv.org/pdf/1907.04135),
* [**NICE**](https://arxiv.org/pdf/2104.07411), 
* [**MOC**](https://arxiv.org/pdf/2004.11165),
* [**DiCE**](https://arxiv.org/abs/1905.07697).

For the insurance applications, the methods are adapted to account for the criteria mentioned above and to distinguish between actionable and non-actionable features. Conformal prediction is used to quantify predictive uncertainty and to assess whether a counterfactual achieves the desired target while accounting for model uncertainty.

## Case Studies

### 1. Flood Insurance

The first case study uses data from the **FEMA National Flood Insurance Program (NFIP)**.

The analysis focuses on flood claim severity conditional on a claim occurring. The dataset is restricted to selected flood events in Florida in 2023-2024. The predictive model is an XGBoost model with a Gamma objective. CEs are generated for individual claims with the objective of reducing the predicted claim severity. The actionable features considered in this case study include:

* basement level;
* elevation;
* contents location.

CEs aim to identify individual mitigation measures that could reduce the predicted flood severity.
  
### 2. Automobile Telematics

The second case study uses the `catelematic13` dataset from the CASdatasets collection. A Poisson GLM is used to predict claim frequency. The actionable features considered include:

* percentage of driving during rush hours;
* acceleration events;
* braking events.

CEs aim to identify changes in driving behaviour that could reduce the predicted claim frequency.

## Authors

**Agathe Fernandes Machado**
Université du Québec à Montréal (UQAM)

**Fei Huang**
UNSW Sydney
