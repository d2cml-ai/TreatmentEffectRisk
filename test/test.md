

# TreatmentEffectRisk

This repository is an implementation of the Treatment Effect Risk:
Bounds and Inference package, based on the replication by Kallus (2024).

# References

- Paper: [Treatment Effect Risk: Bounds and
  Inference](https://arxiv.org/abs/2201.05893)
- Original Repository:
  [CausalML/TreatmentEffectRisk](https://github.com/CausalML/TreatmentEffectRisk)

# License

This project is licensed under the MIT License - see the LICENSE file
for details.

<!-- ## MAP
&#10;![](./original_pkg/functions.png) -->

# Contact

For questions or inquiries about the package, please contact
\[fr.jhonk@gmail.com\].

# Usage

``` python
# pip install TrER==0.1.41
```

``` python
import numpy as np
import pandas as pd
url_csv = 'https://raw.githubusercontent.com/CausalML/TreatmentEffectRisk/main/data/behaghel.csv'
import warnings 

warnings.filterwarnings("ignore")
```

``` python
pd.read_csv(url_csv).head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | Unnamed: 0 | sw | A_public | A_private | A_standard | Y | College_education | nivetude2 | Vocational | High_school_dropout | ... | Sensitive_suburban_area | Insertion | Interim | Conseil | age | Number_of_children | exper | salaire.num | mois_saisie_occ | ndem |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 1 | 0.831709 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | ... | 0 | 1 | 0 | 0 | 48 | 0 | 10 | 2 | 12 | 3 |
| 1 | 2 | 0.827225 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | ... | 0 | 1 | 0 | 0 | 48 | 2 | 5 | 3 | 5 | 1 |
| 2 | 3 | 0.827225 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | ... | 0 | 1 | 0 | 0 | 42 | 2 | 25 | 4 | 6 | 3 |
| 3 | 4 | 0.957266 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | ... | 0 | 1 | 0 | 0 | 40 | 1 | 15 | 4 | 3 | 1 |
| 4 | 5 | 0.957266 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | ... | 0 | 1 | 0 | 0 | 38 | 0 | 6 | 4 | 3 | 7 |

<p>5 rows × 46 columns</p>
</div>
