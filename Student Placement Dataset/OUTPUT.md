# Model Evaluation & Testing Output

## Performance Summary
- **Test Dataset Size**: 5,000 records
- **Evaluation Metric**: Accuracy
- **Target Variable**: `Placement_Status`

## Sample Predictions Output

| Student_ID | Age | Gender | Degree | Branch | CGPA | Coding_Skills | Aptitude_Test_Score | Placement_Status | Predicted_placement_status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **15202** | 23 | Male | B.Tech | Civil | 7.32 | 5 | 56 | Placed | **Placed** |
| **4573** | 24 | Female | MCA | ME | 4.76 | 1 | 37 | Not Placed | **Not Placed** |
| **34424** | 20 | Male | BCA | ME | 6.16 | 3 | 68 | Not Placed | **Not Placed** |
| **38881** | 19 | Male | B.Sc | IT | 8.77 | 8 | 83 | Placed | **Placed** |
| **30191** | 23 | Male | B.Tech | ME | 7.63 | 4 | 66 | Not Placed | **Not Placed** |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| **40640** | 19 | Female | MCA | Civil | 4.91 | 1 | 47 | Not Placed | **Not Placed** |
| **46999** | 20 | Male | B.Tech | ECE | 6.82 | 5 | 76 | Placed | **Placed** |
| **3816** | 20 | Male | BCA | IT | 5.80 | 6 | 56 | Not Placed | **Not Placed** |
| **41368** | 23 | Female | B.Tech | IT | 7.67 | 7 | 67 | Placed | **Placed** |
| **44562** | 24 | Male | B.Tech | CSE | 6.68 | 7 | 85 | Not Placed | **Not Placed** |

---

## Verification Result
```text
The model is trained on the dataset!
[5000 rows x 16 columns]
The accuracy of this model is 100.0%