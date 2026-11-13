import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv("data/crop_data.csv")

print("=" * 60)
print("AI-BASED CROP ADVISORY SYSTEM")
print("=" * 60)


# ============================================================
# STEP 3 - DATA PREPROCESSING
# ============================================================

print("\nSTEP 3 - DATA PREPROCESSING")
print("=" * 60)

print("\nDataset Shape:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate Records:")
print(data.duplicated().sum())

# Remove duplicate records
data = data.drop_duplicates()

print("\nDataset after removing duplicates:")
print(data.shape)

# Separate input and target
X = data.drop("Recommended_Crop", axis=1)
y = data["Recommended_Crop"]


# Categorical features
categorical_features = [
    "District",
    "Region",
    "Soil_Type",
    "Crop_History",
    "Fertilizer_Used",
    "Pest_Incidents",
    "Resource_Level"
]


# Numerical features
numerical_features = [
    "Rainfall_mm",
    "Temperature_C",
    "Humidity_Percent"
]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# STEP 4 - EXPLORATORY DATA ANALYSIS
# ============================================================

print("\nSTEP 4 - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nCrop Distribution:")
print(data["Recommended_Crop"].value_counts())


# Crop distribution
plt.figure(figsize=(8, 5))

data["Recommended_Crop"].value_counts().plot(kind="bar")

plt.title("Crop Distribution")
plt.xlabel("Recommended Crop")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("results/crop_distribution.png")
plt.close()


# Rainfall distribution
plt.figure(figsize=(9, 5))

sns.boxplot(
    data=data,
    x="Recommended_Crop",
    y="Rainfall_mm"
)

plt.title("Rainfall Distribution by Recommended Crop")
plt.xlabel("Recommended Crop")
plt.ylabel("Rainfall (mm)")
plt.tight_layout()

plt.savefig("results/rainfall_by_crop.png")
plt.close()


# Temperature distribution
plt.figure(figsize=(9, 5))

sns.boxplot(
    data=data,
    x="Recommended_Crop",
    y="Temperature_C"
)

plt.title("Temperature Distribution by Recommended Crop")
plt.xlabel("Recommended Crop")
plt.ylabel("Temperature (°C)")
plt.tight_layout()

plt.savefig("results/temperature_by_crop.png")
plt.close()


# Region-wise crop distribution
region_crop = pd.crosstab(
    data["Region"],
    data["Recommended_Crop"]
)

print("\nRegion-wise Crop Distribution:")
print(region_crop)

region_crop.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Region-wise Crop Distribution")
plt.xlabel("Region")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.legend(title="Crop")
plt.tight_layout()

plt.savefig("results/region_wise_crop_distribution.png")
plt.close()


# Resource-level distribution
resource_distribution = data["Resource_Level"].value_counts()

plt.figure(figsize=(8, 5))

resource_distribution.plot(kind="bar")

plt.title("Resource Level Distribution")
plt.xlabel("Resource Level")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("results/resource_level_distribution.png")
plt.close()


# Soil-type distribution
soil_distribution = data["Soil_Type"].value_counts()

plt.figure(figsize=(8, 5))

soil_distribution.plot(kind="bar")

plt.title("Soil Type Distribution")
plt.xlabel("Soil Type")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("results/soil_type_distribution.png")
plt.close()


print("\nEDA completed successfully!")


# ============================================================
# STEP 5 - MULTIPLE ML MODEL DEVELOPMENT
# ============================================================

print("\nSTEP 5 - MULTIPLE ML MODEL DEVELOPMENT")
print("=" * 60)


# Logistic Regression
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=2000)
        )
    ]
)


# Decision Tree
decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(random_state=42)
        )
    ]
)


# Random Forest
random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)


# Store models
models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}


results = []

trained_models = {}


# Train and evaluate models
for model_name, model in models.items():

    print("\n" + "-" * 60)
    print(f"TRAINING: {model_name}")
    print("-" * 60)

    # Train model
    model.fit(X_train, y_train)

    # Store trained model
    trained_models[model_name] = model

    # Prediction
    y_pred = model.predict(X_test)


    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    # Precision
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # Recall
    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # F1-score
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })


    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")


# Convert results to DataFrame
results_df = pd.DataFrame(results)


# Model comparison
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# Save model comparison
results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)


# Model comparison graph
results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Model Performance Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()

plt.savefig("results/model_comparison.png")
plt.close()


# Select best model using F1-score
best_model_name = results_df.loc[
    results_df["F1-Score"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]


print("\nBest Model:")
print(best_model_name)


# ============================================================
# STEP 6 - REGION-WISE RELIABILITY
# ============================================================

print("\n")
print("=" * 60)
print("STEP 6 - REGION-WISE RELIABILITY EVALUATION")
print("=" * 60)

print("\nSelected Model:")
print(best_model_name)


regions = data["Region"].unique()

region_results = []


for region in regions:

    print("\n" + "-" * 60)
    print(f"REGION: {region}")
    print("-" * 60)


    # Select data belonging to the region
    region_data = data[
        data["Region"] == region
    ]


    X_region = region_data.drop(
        "Recommended_Crop",
        axis=1
    )


    y_region = region_data[
        "Recommended_Crop"
    ]


    # Predict using selected model
    y_region_pred = best_model.predict(
        X_region
    )


    # Accuracy
    region_accuracy = accuracy_score(
        y_region,
        y_region_pred
    )


    # Precision
    region_precision = precision_score(
        y_region,
        y_region_pred,
        average="weighted",
        zero_division=0
    )


    # Recall
    region_recall = recall_score(
        y_region,
        y_region_pred,
        average="weighted",
        zero_division=0
    )


    # F1-score
    region_f1 = f1_score(
        y_region,
        y_region_pred,
        average="weighted",
        zero_division=0
    )


    region_results.append({
        "Region": region,
        "Records": len(region_data),
        "Accuracy": region_accuracy,
        "Precision": region_precision,
        "Recall": region_recall,
        "F1-Score": region_f1
    })


    print(f"Records   : {len(region_data)}")
    print(f"Accuracy  : {region_accuracy:.4f}")
    print(f"Precision : {region_precision:.4f}")
    print(f"Recall    : {region_recall:.4f}")
    print(f"F1-Score  : {region_f1:.4f}")


# ============================================================
# REGION-WISE COMPARISON
# ============================================================

region_results_df = pd.DataFrame(
    region_results
)


print("\n")
print("=" * 60)
print("REGION-WISE PERFORMANCE")
print("=" * 60)

print(
    region_results_df.to_string(
        index=False
    )
)


# Save region results
region_results_df.to_csv(
    "results/region_wise_performance.csv",
    index=False
)


# ============================================================
# REGION-WISE PERFORMANCE GRAPH
# ============================================================

region_plot = region_results_df.set_index(
    "Region"
)


region_plot[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].plot(
    kind="bar",
    figsize=(11, 6)
)


plt.title(
    "Region-wise Model Performance"
)

plt.xlabel("Region")
plt.ylabel("Score")

plt.ylim(0, 1.1)

plt.xticks(rotation=0)

plt.legend()

plt.tight_layout()


plt.savefig(
    "results/region_wise_performance.png"
)

plt.close()


# ============================================================
# RELIABILITY ANALYSIS
# ============================================================

best_region = region_results_df.loc[
    region_results_df["F1-Score"].idxmax()
]


worst_region = region_results_df.loc[
    region_results_df["F1-Score"].idxmin()
]


print("\n")
print("=" * 60)
print("RELIABILITY ANALYSIS")
print("=" * 60)


print(
    f"Best Performing Region: "
    f"{best_region['Region']}"
)


print(
    f"Best Region F1-Score: "
    f"{best_region['F1-Score']:.4f}"
)


print(
    f"Worst Performing Region: "
    f"{worst_region['Region']}"
)


print(
    f"Worst Region F1-Score: "
    f"{worst_region['F1-Score']:.4f}"
)


performance_difference = (
    best_region["F1-Score"]
    - worst_region["F1-Score"]
)


print(
    f"Performance Difference: "
    f"{performance_difference:.4f}"
)


if performance_difference <= 0.10:

    print(
        "\nConclusion: The selected model "
        "shows relatively consistent performance "
        "across the evaluated regions."
    )

else:

    print(
        "\nConclusion: The selected model "
        "shows noticeable performance differences "
        "across regions and requires further evaluation."
    )


print("\n")
print("=" * 60)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 60)


# ============================================================
# STEP 7 - BIAS ANALYSIS BY RESOURCE LEVEL
# ============================================================

print("\n")
print("=" * 70)
print("STEP 7 - BIAS ANALYSIS BY RESOURCE LEVEL")
print("=" * 70)


# Predict crop recommendations for the complete dataset
all_predictions = best_model.predict(X)


# Create analysis dataframe
bias_data = data[
    ["Resource_Level", "Recommended_Crop"]
].copy()


# IMPORTANT:
# The model already predicts crop names directly.
# No LabelEncoder is required.

predicted_crop_names = all_predictions


bias_data["Predicted_Crop"] = predicted_crop_names


# Check whether prediction is correct
bias_data["Correct"] = (
    bias_data["Recommended_Crop"]
    == bias_data["Predicted_Crop"]
)


# ============================================================
# RESOURCE-LEVEL PERFORMANCE
# ============================================================

resource_results = []


for level in data["Resource_Level"].unique():

    subset = bias_data[
        bias_data["Resource_Level"] == level
    ]


    accuracy = subset["Correct"].mean()


    resource_results.append({
        "Resource_Level": level,
        "Samples": len(subset),
        "Correct_Predictions": int(
            subset["Correct"].sum()
        ),
        "Accuracy": accuracy
    })


resource_performance = pd.DataFrame(
    resource_results
)


# Sort by accuracy
resource_performance = resource_performance.sort_values(
    by="Accuracy",
    ascending=False
)


print("\nRESOURCE-LEVEL PERFORMANCE")
print("-" * 70)

print(
    resource_performance.to_string(
        index=False
    )
)


# ============================================================
# SAVE BIAS ANALYSIS
# ============================================================

resource_performance.to_csv(
    "results/resource_level_bias_analysis.csv",
    index=False
)


# ============================================================
# RESOURCE-LEVEL BIAS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    resource_performance["Resource_Level"],
    resource_performance["Accuracy"]
)

plt.title(
    "Model Accuracy by Resource Level"
)

plt.xlabel("Resource Level")

plt.ylabel("Accuracy")

plt.ylim(0, 1.1)

plt.tight_layout()

plt.savefig(
    "results/resource_level_bias_analysis.png"
)

plt.close()


# ============================================================
# BIAS INTERPRETATION
# ============================================================

best_resource = resource_performance.iloc[0]

worst_resource = resource_performance.iloc[-1]


performance_gap = (
    best_resource["Accuracy"]
    - worst_resource["Accuracy"]
)


print("\nBIAS ANALYSIS")
print("-" * 70)


print(
    f"Best performing resource level : "
    f"{best_resource['Resource_Level']}"
)


print(
    f"Worst performing resource level: "
    f"{worst_resource['Resource_Level']}"
)


print(
    f"Accuracy gap                  : "
    f"{performance_gap:.2f}"
)


# ============================================================
# BIAS CONCLUSION
# ============================================================

if performance_gap <= 0.10:

    print(
        "\nConclusion: The model shows relatively low "
        "performance variation across resource levels "
        "and does not show strong evidence of bias "
        "toward high-resource farming areas."
    )

else:

    print(
        "\nConclusion: Noticeable performance differences "
        "exist across resource levels. The model may be "
        "biased toward better-resourced farming areas "
        "and should be reviewed before deployment."
    )


print("\nBias analysis saved to:")
print(
    "results/resource_level_bias_analysis.csv"
)

print("\nBias graph saved to:")
print(
    "results/resource_level_bias_analysis.png"
)


print("\n")
print("=" * 70)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 70)
# ============================================================
# STEP 8 - HUMAN REVIEW AND DEPLOYMENT DECISION
# ============================================================

print("\n")
print("=" * 70)
print("STEP 8 - HUMAN REVIEW AND DEPLOYMENT DECISION")
print("=" * 70)


# Get prediction probabilities from the selected model
prediction_probabilities = best_model.predict_proba(X)

# Get maximum probability for each prediction
prediction_confidence = prediction_probabilities.max(axis=1)

# Get predicted crop
predicted_crops = best_model.predict(X)


# Create deployment decision dataframe
deployment_data = data[
    [
        "District",
        "Region",
        "Resource_Level",
        "Recommended_Crop"
    ]
].copy()


deployment_data["Predicted_Crop"] = predicted_crops

deployment_data["Confidence"] = prediction_confidence


# ============================================================
# HUMAN REVIEW RULES
# ============================================================

def review_decision(confidence):

    if confidence >= 0.80:
        return "AUTO-ACCEPT"

    elif confidence >= 0.60:
        return "OFFICER REVIEW"

    else:
        return "MANDATORY OFFICER REVIEW"


deployment_data["Review_Decision"] = (
    deployment_data["Confidence"]
    .apply(review_decision)
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nDEPLOYMENT DECISIONS")
print("-" * 70)

display_columns = [
    "District",
    "Region",
    "Resource_Level",
    "Recommended_Crop",
    "Predicted_Crop",
    "Confidence",
    "Review_Decision"
]

print(
    deployment_data[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# REVIEW SUMMARY
# ============================================================

review_summary = (
    deployment_data["Review_Decision"]
    .value_counts()
    .reset_index()
)

review_summary.columns = [
    "Review_Decision",
    "Number_of_Cases"
]


print("\n")
print("=" * 70)
print("HUMAN REVIEW SUMMARY")
print("=" * 70)

print(
    review_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

deployment_data.to_csv(
    "results/human_review_decisions.csv",
    index=False
)


review_summary.to_csv(
    "results/human_review_summary.csv",
    index=False
)


# ============================================================
# REVIEW RATE
# ============================================================

total_cases = len(deployment_data)

review_cases = len(
    deployment_data[
        deployment_data["Review_Decision"] != "AUTO-ACCEPT"
    ]
)

review_rate = (
    review_cases / total_cases
)


print("\n")
print("Human Review Statistics")
print("-" * 70)

print(
    f"Total recommendations : {total_cases}"
)

print(
    f"Cases requiring review: {review_cases}"
)

print(
    f"Review rate            : {review_rate:.2%}"
)


# ============================================================
# DEPLOYMENT RECOMMENDATION
# ============================================================

print("\n")
print("=" * 70)
print("DEPLOYMENT RECOMMENDATION")
print("=" * 70)


if review_rate <= 0.30:

    print(
        "The model can be considered for assisted deployment. "
        "High-confidence recommendations may be provided "
        "automatically, while uncertain recommendations "
        "should be reviewed by agricultural officers."
    )

else:

    print(
        "The model should be deployed only with strong "
        "human oversight because a substantial proportion "
        "of recommendations require review."
    )


print("\nAgricultural officers should review recommendations when:")
print("1. Model confidence is below 80%.")
print("2. The prediction is uncertain between multiple crops.")
print("3. The farmer belongs to a resource group with lower model performance.")
print("4. Soil, rainfall, temperature, or pest conditions are unusual.")
print("5. The recommendation conflicts with local agricultural knowledge.")


print("\nResults saved to:")
print("results/human_review_decisions.csv")
print("results/human_review_summary.csv")


print("\n")
print("=" * 70)
print("STEP 8 COMPLETED SUCCESSFULLY")
print("=" * 70)
# ============================================================
# STEP 9 - FINAL EVALUATION AND OVERALL CONCLUSION
# ============================================================

print("\n")
print("=" * 70)
print("STEP 9 - FINAL EVALUATION AND OVERALL CONCLUSION")
print("=" * 70)


# ============================================================
# 1. BEST MODEL PERFORMANCE
# ============================================================

best_model_row = results_df.loc[
    results_df["F1-Score"].idxmax()
]

best_model_name_final = best_model_row["Model"]

best_model_accuracy = best_model_row["Accuracy"]

best_model_precision = best_model_row["Precision"]

best_model_recall = best_model_row["Recall"]

best_model_f1 = best_model_row["F1-Score"]


print("\nBEST MODEL")
print("-" * 70)

print(
    f"Model     : {best_model_name_final}"
)

print(
    f"Accuracy  : {best_model_accuracy:.4f}"
)

print(
    f"Precision : {best_model_precision:.4f}"
)

print(
    f"Recall    : {best_model_recall:.4f}"
)

print(
    f"F1-Score  : {best_model_f1:.4f}"
)


# ============================================================
# 2. REGION-WISE RELIABILITY
# ============================================================

best_region_final = region_results_df.loc[
    region_results_df["F1-Score"].idxmax()
]

worst_region_final = region_results_df.loc[
    region_results_df["F1-Score"].idxmin()
]


region_performance_gap = (
    best_region_final["F1-Score"]
    - worst_region_final["F1-Score"]
)


print("\nREGION-WISE RELIABILITY")
print("-" * 70)

print(
    f"Best Region  : "
    f"{best_region_final['Region']}"
)

print(
    f"Best F1-Score: "
    f"{best_region_final['F1-Score']:.4f}"
)

print(
    f"Worst Region : "
    f"{worst_region_final['Region']}"
)

print(
    f"Worst F1-Score: "
    f"{worst_region_final['F1-Score']:.4f}"
)

print(
    f"Regional F1 Gap: "
    f"{region_performance_gap:.4f}"
)


# ============================================================
# 3. RESOURCE-LEVEL BIAS
# ============================================================

best_resource_final = resource_performance.iloc[0]

worst_resource_final = resource_performance.iloc[-1]


resource_performance_gap = (
    best_resource_final["Accuracy"]
    - worst_resource_final["Accuracy"]
)


print("\nRESOURCE-LEVEL BIAS")
print("-" * 70)

print(
    f"Best Resource Level : "
    f"{best_resource_final['Resource_Level']}"
)

print(
    f"Best Accuracy       : "
    f"{best_resource_final['Accuracy']:.4f}"
)

print(
    f"Worst Resource Level: "
    f"{worst_resource_final['Resource_Level']}"
)

print(
    f"Worst Accuracy      : "
    f"{worst_resource_final['Accuracy']:.4f}"
)

print(
    f"Resource Accuracy Gap: "
    f"{resource_performance_gap:.4f}"
)


# ============================================================
# 4. HUMAN REVIEW
# ============================================================

total_recommendations = len(
    deployment_data
)

review_cases_final = len(
    deployment_data[
        deployment_data["Review_Decision"] != "AUTO-ACCEPT"
    ]
)

review_rate_final = (
    review_cases_final /
    total_recommendations
)


print("\nHUMAN REVIEW")
print("-" * 70)

print(
    f"Total Recommendations: "
    f"{total_recommendations}"
)

print(
    f"Cases Requiring Review: "
    f"{review_cases_final}"
)

print(
    f"Review Rate           : "
    f"{review_rate_final:.2%}"
)


# ============================================================
# 5. OVERALL RELIABILITY DECISION
# ============================================================

if (
    best_model_f1 >= 0.80
    and region_performance_gap <= 0.20
    and resource_performance_gap <= 0.20
):

    overall_status = "SUITABLE FOR ASSISTED DEPLOYMENT"

else:

    overall_status = "REQUIRES FURTHER VALIDATION"


print("\n")
print("=" * 70)
print("OVERALL DEPLOYMENT STATUS")
print("=" * 70)

print(overall_status)


# ============================================================
# 6. FINAL CONCLUSION
# ============================================================

final_conclusion = (
    "Multiple crop-advisory machine learning models were "
    "developed and evaluated using Accuracy, Precision, "
    "Recall, and F1-score. The best-performing model was "
    f"{best_model_name_final}. Region-wise analysis was "
    "performed to determine whether the model remained "
    "reliable across different geographical conditions. "
    "Resource-level analysis was also performed to identify "
    "possible performance differences between high, medium, "
    "and low-resource farming areas. A human-in-the-loop "
    "mechanism was implemented so that uncertain or unusual "
    "recommendations can be reviewed by agricultural officers. "
    "The results support assisted deployment, but the model "
    "should not replace agricultural expertise. Further "
    "validation using a larger real-world dataset is required "
    "before full-scale deployment."
)


print("\nFINAL CONCLUSION")
print("-" * 70)

print(final_conclusion)


# ============================================================
# 7. CREATE FINAL EVALUATION TABLE
# ============================================================

final_evaluation = pd.DataFrame({
    "Evaluation_Metric": [
        "Best Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "Best Region",
        "Worst Region",
        "Regional F1 Gap",
        "Best Resource Level",
        "Worst Resource Level",
        "Resource Accuracy Gap",
        "Total Recommendations",
        "Cases Requiring Review",
        "Review Rate",
        "Overall Deployment Status"
    ],

    "Result": [
        best_model_name_final,
        round(best_model_accuracy, 4),
        round(best_model_precision, 4),
        round(best_model_recall, 4),
        round(best_model_f1, 4),
        best_region_final["Region"],
        worst_region_final["Region"],
        round(region_performance_gap, 4),
        best_resource_final["Resource_Level"],
        worst_resource_final["Resource_Level"],
        round(resource_performance_gap, 4),
        total_recommendations,
        review_cases_final,
        f"{review_rate_final:.2%}",
        overall_status
    ]
})


# ============================================================
# 8. SAVE FINAL RESULTS
# ============================================================

final_evaluation.to_csv(
    "results/final_evaluation.csv",
    index=False
)


# Save final conclusion as text
with open(
    "results/final_conclusion.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AI-BASED CROP ADVISORY SYSTEM\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        "BEST MODEL\n"
    )

    file.write(
        f"Model: {best_model_name_final}\n"
    )

    file.write(
        f"Accuracy: {best_model_accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {best_model_precision:.4f}\n"
    )

    file.write(
        f"Recall: {best_model_recall:.4f}\n"
    )

    file.write(
        f"F1-Score: {best_model_f1:.4f}\n\n"
    )

    file.write(
        "REGION-WISE RELIABILITY\n"
    )

    file.write(
        f"Best Region: {best_region_final['Region']}\n"
    )

    file.write(
        f"Worst Region: {worst_region_final['Region']}\n"
    )

    file.write(
        f"Regional F1 Gap: {region_performance_gap:.4f}\n\n"
    )

    file.write(
        "RESOURCE-LEVEL BIAS\n"
    )

    file.write(
        f"Best Resource Level: "
        f"{best_resource_final['Resource_Level']}\n"
    )

    file.write(
        f"Worst Resource Level: "
        f"{worst_resource_final['Resource_Level']}\n"
    )

    file.write(
        f"Resource Accuracy Gap: "
        f"{resource_performance_gap:.4f}\n\n"
    )

    file.write(
        "HUMAN REVIEW\n"
    )

    file.write(
        f"Total Recommendations: "
        f"{total_recommendations}\n"
    )

    file.write(
        f"Cases Requiring Review: "
        f"{review_cases_final}\n"
    )

    file.write(
        f"Review Rate: "
        f"{review_rate_final:.2%}\n\n"
    )

    file.write(
        "OVERALL DEPLOYMENT STATUS\n"
    )

    file.write(
        f"{overall_status}\n\n"
    )

    file.write(
        "FINAL CONCLUSION\n"
    )

    file.write(
        final_conclusion
    )


# ============================================================
# 9. COMPLETION MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RESULTS SAVED")
print("=" * 70)

print(
    "results/final_evaluation.csv"
)

print(
    "results/final_conclusion.txt"
)

print("\n")
print("=" * 70)
print("STEP 9 COMPLETED SUCCESSFULLY")
print("=" * 70)