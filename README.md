# MGTAB
A Multi-Relational Graph-Based Twitter Account Detection Benchmark
## Introduction

MGTAB is the first standardized graph-based benchmark for stance and bot detection. MGTAB contains 10,199 expert-annotated users
and 7 types of relationships, ensuring high-quality annotation and diversified relations. For more details, please refer to the MGTAB paper.
 
### Distribution of Labels in annotations.

<table>
    <thead>
        <tr>
            <td colspan=3>Stance</td>
            <td colspan=3>Bot</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan=1>Lable</td>
            <td colspan=1>Count</td>
            <td colspan=1>Percentage</td>
            <td colspan=1>Lable</td>
            <td colspan=1>Count</td>
            <td colspan=1>Percentage</td>
        </tr>
        <tr>
            <td colspan=1>neutral</td>
            <td colspan=1>3,776</td>
            <td colspan=1>37.02</td>
            <td colspan=1>human</td>
            <td colspan=1>7,451</td>
            <td colspan=1>73.06</td>
        </tr>
        <tr>
            <td colspan=1>against</td>
            <td colspan=1>3,637</td>
            <td colspan=1>35.66</td>
            <td colspan=1>bot</td>
            <td colspan=1>2,748</td>
            <td colspan=1>26.94</td>
        </tr>
        <tr>
            <td colspan=1>support</td>
            <td colspan=1>2,786</td>
            <td colspan=1>27.32</td>
            <td colspan=3> </td>
        </tr>
    </tbody>
</table>
MGTAB contains 10,199 expert-annotated users, and 400,000 additional unlabelled users in MGTAB-large compared to  MGTAB.

## Enviromment

```
python 3.7
scikit-learn 1.0.2
torch 1.8.1+cu111
torch_cluster-1.5.9
torch_scatter-2.0.6
torch_sparse-0.6.9
torch_spline_conv-1.2.1
torch-geometric 2.0.4
pytorch-lightning 1.5.0
```

## Licensing
The MGTAB dataset uses the CC BY-NC-ND 4.0 license. Implemented code in the
MGTAB evaluation framework uses the MIT license.

##  Datasets download
For SemEval-2016 T6, visit the [SemEval2016 repository](https://alt.qcri.org/semeval2016/task6/)
For SemEval-2019 T7, visit the [SemEval2019 github repository](https://github.com/kochkinaelena/RumourEval2019)
For TwiBot-20, visit the [TwiBot-20 github repository](https://github.com/BunsenFeng/TwiBot-20).
For TwiBot-22, visit the [TwiBot-22 github repository](https://github.com/LuoUndergradXJTU/TwiBot-22).
For other bot detection datasets, please visit the [Bot Repository](https://botometer.osome.iu.edu/bot-repository/datasets.html).

MGTAB is available at [Google Drive](https://drive.google.com/uc?export=download&id=1gbWNOoU1JB8RrTu2a5j9KMNVa9wX72Fe).
MGTAB-large (contains 400,000 unlabeled users) is available at [Google Drive](https://drive.google.com/uc?export=download&id=1Gkq83o9uIjldOU2VZdvbferKASpVI8ul).
We also offer the standardized Cresci-15 at [Google Drive](https://drive.google.com/uc?export=download&id=1AzMUNt70we5G2DShS8hk5qH95VR9HfD3). 
After downloading these datasets, please unzip it into path "./Dataset".

