---
# Code Execution Output Log

This file details the output logs generated during the execution of the dataset parsing notebook cell outputs.

---

## 1. Unique Age Distribution in Dataset (`set(age)`)

Executing `set(age)` returns all distinct ages present within the parsed image dataset:

```python
{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 
 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 
 93, 95, 96, 99, 100, 101, 103, 105, 110, 111, 115, 116}
Minimum Age: 1

Maximum Age: 116

2. Sample Gender Labels List (gender[:... ])
A preview of extracted gender labels (0 for Male, 1 for Female):

Python
[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, ...]
3. Sample File Path Preview (img_path[:... ])
Parsed image filenames extracted from the directory loop:

Python
[
 '100_0_0_20170112213500903.jpg.chip.jpg',
 '100_0_0_20170112215240346.jpg.chip.jpg',
 '100_1_0_20170110183726390.jpg.chip.jpg',
 '100_1_0_20170112213001988.jpg.chip.jpg',
 '100_1_0_20170112213303693.jpg.chip.jpg',
 '100_1_0_20170112215032192.jpg.chip.jpg',
 '101_0_0_20170112213500903.jpg.chip.jpg',
 '105_1_0_20170112213001988.jpg.chip.jpg',
 '10_0_0_20161220222308131.jpg.chip.jpg',
 ...
]