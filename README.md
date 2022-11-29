# MGTAB
MGTAB: A Multi-Relational Graph-Based Twitter Account Detection Benchmark

## Introduction
MGTAB is the first standardized graph-based benchmark for stance and bot detection. MGTAB contains 10,199 expert-annotated users
and 7 types of relationships, ensuring high-quality annotation and diversified relations. For more details, please refer to the MGTAB paper.
 
### Distribution of labels in annotations.
<table>
    <thead>
        <tr>
            <td colspan=3 align="center"><b>Stance</b></td>
            <td colspan=3 align="center"><b>Bot</b></td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan=1 align="center">Lable</td>
            <td colspan=1 align="center">Count</td>
            <td colspan=1 align="center">Percentage</td>
            <td colspan=1 align="center">Lable</td>
            <td colspan=1 align="center">Count</td>
            <td colspan=1 align="center">Percentage</td>
        </tr>
        <tr>
            <td colspan=1 align="center">neutral</td>
            <td colspan=1 align="center">3,776</td>
            <td colspan=1 align="center">37.02</td>
            <td colspan=1 align="center">human</td>
            <td colspan=1 align="center">7,451</td>
            <td colspan=1 align="center">73.06</td>
        </tr>
        <tr>
            <td colspan=1 align="center">against</td>
            <td colspan=1 align="center">3,637</td>
            <td colspan=1 align="center">35.66</td>
            <td colspan=1 align="center">bot</td>
            <td colspan=1 align="center">2,748</td>
            <td colspan=1 align="center">26.94</td>
        </tr>
        <tr>
            <td colspan=1 align="center">support</td>
            <td colspan=1 align="center">2,786</td>
            <td colspan=1 align="center">27.32</td>
            <td colspan=3 align="center"> </td>
        </tr>
    </tbody>
</table>
MGTAB contains 10,199 expert-annotated users, and 400,000 additional unlabelled users in MGTAB-large compared to  MGTAB.

### Multiple relations in the MGTAB.
Our proposed dataset has seven types of user relationships.

<table>
    <thead>
        <tr>
            <td colspan=8 align="center"><b>MGTAB</b></td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan=1 align="center">Edge type</td>
            <td colspan=1 align="center">followers</td>
            <td colspan=1 align="center">friends</td>
            <td colspan=1 align="center">mention</td>
            <td colspan=1 align="center">reply</td>
            <td colspan=1 align="center">quoted</td>
            <td colspan=1 align="center">URL</td>
            <td colspan=1 align="center">hashtag</td>
        </tr>
        <tr>
            <td colspan=1 align="center">Numbers</td>
            <td colspan=1 align="center">308,120</td>
            <td colspan=1 align="center">412,575</td>
            <td colspan=1 align="center">114,516</td>
            <td colspan=1 align="center">223,466</td>
            <td colspan=1 align="center">77,631</td>
            <td colspan=1 align="center">263,800</td>
            <td colspan=1 align="center">300,000</td>
        </tr>
     <thead>
        <tr>
            <td colspan=8 align="center"><b>MGTAB-large</b></td>
        </tr>
    </thead>
         <tr>
            <td colspan=1 align="center">Edge type</td>
            <td colspan=1 align="center">followers</td>
            <td colspan=1 align="center">friends</td>
            <td colspan=1 align="center">mention</td>
            <td colspan=1 align="center">reply</td>
            <td colspan=1 align="center">quoted</td>
            <td colspan=1 align="center">URL</td>
            <td colspan=1 align="center">hashtag</td>
        </tr>
        <tr>
            <td colspan=1 align="center">Numbers</td>
            <td colspan=1 align="center">31,990,488</td>
            <td colspan=1 align="center">49,668,723</td>
            <td colspan=1 align="center">7,135,192</td>
            <td colspan=1 align="center">1,018,834</td>
            <td colspan=1 align="center">182,296</td>
            <td colspan=1 align="center">51,281</td>
            <td colspan=1 align="center">7,950,896</td>
        </tr>
    </tbody>
</table>


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


## Baseline performance
### Stance detection performance on MGTAB
| methods             | type | accuracy              | precision            | recall               | f1-score             |
| ------------------- | ---- | --------------------- | -------------------- | -------------------- | -------------------- |
| AdaBoost            | F    | 74.59</br> $_{1.41}$  | 74.60</br> $_{1.35}$ | 74.02</br> $_{1.61}$ | 73.88</br> $_{1.47}$ |
| Random Forest       | F    | 79.62</br> $_{0.68}$  | 80.04</br> $_{0.43}$ | 78.83</br> $_{0.98}$ | 79.04</br> $_{0.82}$ |
| Decision Tree       | F    | 66.92</br> $_{0.93}$  | 66.34</br> $_{1.02}$ | 66.23</br> $_{1.06}$ | 66.03</br> $_{0.84}$ |
| SVM                 | F    | 81.23</br> $_{0.66}$  | 81.40</br> $_{0.71}$ | 80.86</br> $_{1.09}$ | 80.71</br> $_{0.78}$ |
| KNN                 | F    | 76.25</br> $_{1.32}$  | 75.54</br> $_{1.41}$ | 75.70</br> $_{1.37}$ | 75.48</br> $_{1.37}$ |
| Logistic Regression | F    | 79.51</br> $_{1.00}$  | 79.33</br> $_{0.98}$ | 78.83</br> $_{1.17}$ | 78.98</br> $_{1.11}$ |
| GCN                 | G    | 81.35</br> $_{0.58}$  | 81.08</br> $_{0.30}$ | 80.19</br> $_{0.56}$ | 80.08</br> $_{0.56}$ |
| GrapgSAGE           | G    | 83.33</br> $_{1.22}$  | 82.52</br> $_{1.63}$ | 83.45</br> $_{0.63}$ | 82.72</br> $_{1.34}$ |
| GAT                 | G    | 82.19</br> $_{1.23}$  | 81.72</br> $_{1.19}$ | 81.68</br> $_{1.16}$ | 81.04</br> $_{1.24}$ |
| HGT                 | G    | 83.29</br> $_{0.44}$  | 81.63</br> $_{0.58}$ | 81.51</br> $_{0.76}$ | 81.82</br> $_{0.34}$ |
| S-HGN               | G    | 85.32</br> $_{0.53}$  | 83.93</br> $_{0.67}$ | 83.65</br> $_{0.65}$ | 84.42</br> $_{0.43}$ |
| BotRGCN             | G    | 84.71</br> $_{1.43}$  | 83.43</br> $_{1.23}$ | 84.08</br> $_{0.94}$ | 84.30</br> $_{1.44}$ |
| RGT                 | G    | 87.78</br> $_{0.43}$  | 85.22</br> $_{0.89}$ | 84.40</br> $_{0.74}$ | 86.86</br> $_{0.43}$ |


### Bot detection performance on MGTAB
| methods             | type | accuracy             | precision            | recall               | f1-score             |
| ------------------- | ---- | -------------------- | -------------------- | -------------------- | -------------------- |
| AdaBoost            |  F   | 90.12</br> $_{0.92}$ | 88.51</br> $_{1.33}$ | 89.10</br> $_{0.92}$ | 87.71</br> $_{1.10}$ |
| Random Forest       |  F   | 89.52</br> $_{0.44}$ | 88.92</br> $_{0.49}$ | 86.72</br> $_{1.15}$ | 86.83</br> $_{0.53}$ |
| Decision Tree       |  F   | 87.13</br> $_{0.51}$ | 83.81</br> $_{0.72}$ | 83.39</br> $_{1.06}$ | 83.70</br> $_{0.74}$ |
| SVM                 |  F   | 88.68</br> $_{1.40}$ | 85.73</br> $_{1.84}$ | 85.73</br> $_{1.84}$ | 85.31</br> $_{1.73}$ |
| KNN                 |  F   | 85.78</br> $_{0.84}$ | 82.28</br> $_{1.22}$ | 80.49</br> $_{0.64}$ | 81.28</br> $_{0.66}$ |
| Logistic Regression |  F   | 88.49</br> $_{1.31}$ | 85.69</br> $_{1.69}$ | 84.41</br> $_{1.96}$ | 84.97</br> $_{1.67}$ |
| GCN                 |  G   | 85.81</br> $_{1.32}$ | 77.40</br> $_{2.12}$ | 84.37</br> $_{1.73}$ | 78.33</br> $_{1.67}$ |
| GrapgSAGE           |  G   | 88.71</br> $_{1.24}$ | 85.33</br> $_{1.83}$ | 86.15</br> $_{2.55}$ | 85.44</br> $_{1.08}$ |
| GAT                 |  G   | 86.96</br> $_{1.28}$ | 79.71</br> $_{2.96}$ | 84.88</br> $_{1.52}$ | 82.33</br> $_{2.12}$ |
| HGT                 |  G   | 90.28</br> $_{0.29}$ | 85.35</br> $_{0.33}$ | 85.97</br> $_{0.41}$ | 87.52</br> $_{0.37}$ |
| S-HGN               |  G   | 91.42</br> $_{0.43}$ | 87.40</br> $_{0.67}$ | 86.73</br> $_{0.64}$ | 88.72</br> $_{0.58}$ |
| BotRGCN             |  G   | 89.60</br> $_{0.82}$ | 85.21</br> $_{1.81}$ | 87.07</br> $_{1.38}$ | 87.16</br> $_{0.74}$ |
| RGT                 |  G   | 92.12</br> $_{0.37}$ | 88.08</br> $_{0.43}$ | 86.64</br> $_{0.25}$ | 90.41</br> $_{0.47}$ |



## Licensing
The MGTAB dataset uses the CC BY-NC-ND 4.0 license. Implemented code in the
MGTAB evaluation framework uses the MIT license.

##  Datasets download
For SemEval-2016 T6, visit the [SemEval2016 repository](https://alt.qcri.org/semeval2016/task6/).
For SemEval-2019 T7, visit the [SemEval2019 github repository](https://github.com/kochkinaelena/RumourEval2019).
For TwiBot-20, visit the [TwiBot-20 github repository](https://github.com/BunsenFeng/TwiBot-20).
For TwiBot-22, visit the [TwiBot-22 github repository](https://github.com/LuoUndergradXJTU/TwiBot-22).
For other bot detection datasets, please visit the [Bot Repository](https://botometer.osome.iu.edu/bot-repository/datasets.html).

MGTAB is available at [Google Drive](https://drive.google.com/uc?export=download&id=1gbWNOoU1JB8RrTu2a5j9KMNVa9wX72Fe).
MGTAB-large (contains 400,000 unlabeled users) is available at [Google Drive](https://drive.google.com/uc?export=download&id=17XuXbklLJgZI3uISaB6I4Kw39FTw8r1i).
We also offer the standardized Cresci-15 at [Google Drive](https://drive.google.com/uc?export=download&id=1AzMUNt70we5G2DShS8hk5qH95VR9HfD3). 
After downloading these datasets, please unzip it into path "./Dataset".

