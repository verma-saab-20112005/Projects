---
# Analysis Output & Data Profile

## 1. Dataset Summary

| Property | Value |
| :--- | :--- |
| **Total Rows** | 20,000 |
| **Total Columns** | 17 |
| **Target Variable** | `letter` |
| **Memory Usage** | ~2.6 MB |
| **Missing Values** | 0 |

---

## 2. Feature Schema & Description

| Column Index | Column Name | Data Type | Description |
| :---: | :--- | :---: | :--- |
| **0** | `letter` | `object` | Target uppercase character (A–Z) |
| **1** | `xbox` | `int64` | Horizontal position of bounding box |
| **2** | `ybox` | `int64` | Vertical position of bounding box |
| **3** | `width` | `int64` | Width of bounding box |
| **4** | `height` | `int64` | Height of bounding box |
| **5** | `onpix` | `int64` | Total number of ON pixels |
| **6** | `xbar` | `int64` | Mean x of ON pixels in box |
| **7** | `ybar` | `int64` | Mean y of ON pixels in box |
| **8** | `x2bar` | `int64` | Mean variance of x |
| **9** | `y2bar` | `int64` | Mean variance of y |
| **10** | `xybar` | `int64` | Mean correlation of x and y |
| **11** | `x2ybar` | `int64` | Mean of $x^2 y$ |
| **12** | `xy2bar` | `int64` | Mean of $x y^2$ |
| **13** | `xedge` | `int64` | Mean edge count from left to right |
| **14** | `xedgey` | `int64` | Correlation of x-edge with y-position |
| **15** | `yedge` | `int64` | Mean edge count from bottom to top |
| **16** | `yedgex` | `int64` | Correlation of y-edge with x-position |

---

## 3. Class Distribution (`letter`)

Below are the top and bottom class counts across all 26 letters:

| Letter | Count | Letter | Count |
| :---: | :---: | :---: | :---: |
| **U** | 813 | **S** | 748 |
| **D** | 805 | **J** | 747 |
| **P** | 803 | **K** | 739 |
| **T** | 796 | **C** | 736 |
| **M** | 792 | **H** | 734 |
| **A** | 789 | **Z** | 734 |