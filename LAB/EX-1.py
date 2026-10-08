import numpy as np

# 4x4 matrix: Math, Science, English, History
student_scores = np.array([
    [85, 90, 78, 88],
    [92, 85, 80, 75],
    [78, 95, 88, 82],
    [88, 80, 92, 90]
])

subjects = ["Math", "Science", "English", "History"]

# Calculate average marks for each subject
averages = np.mean(student_scores, axis=0)

# Display average marks
for i in range(len(subjects)):
    print(subjects[i], "Average:", averages[i])

# Find subject with highest average
highest_index = np.argmax(averages)
print("Subject with highest average:", subjects[highest_index])
print("Highest average score:", averages[highest_index])
