"""Various functions to assist throughout a data science project.

It includes functionality for:
- Data cleaning: Handling missing values and plotting univariate distributions.
- EDA: Visualizing numeric/categorical relationships and running statistical tests.
- Machine learning: Plotting confusion matrices for model evaluation.

Typical usage examples:
    my_module.number_of_missing_values(df)
    my_module.plot_numeric_distributions(df)
    my_module.numeric_relationships(df, target_feature)
    my_module.plot_confusion(y_test, y_pred)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import chi2_contingency
from matplotlib.colors import LinearSegmentedColormap
from phik import phik_matrix
from sklearn.metrics import confusion_matrix

def number_of_missing_values(df: pd.DataFrame) -> None:
    """
    Explores how much missing information is in the dataframe.

    Args:
        df (pd.DataFrame): The DataFrame to analyze for missing values.

    Returns:
        None
    """
    missing_cells = df.isnull()
    print(missing_cells.sum())
    print(
        f"Percentage of rows with missing entries: {(missing_cells.any(axis=1).mean())*100:.2f}%"
    )
    rows_with_multiple_missing = (missing_cells.sum(axis=1) > 1).sum()
    print(f"Number of rows with more than one missing value: {rows_with_multiple_missing}")

def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """
    Plots the distribution of all numeric features in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the data to be plotted.

    Returns:
        None
    """
    numeric_features = df.select_dtypes(include=["number"])
    num_features = len(numeric_features.columns)
    plots_per_row = 3
    num_rows = math.ceil(num_features / plots_per_row)
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(15, 4 * num_rows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_features.columns):
        sns.histplot(df[col], ax=axes[i], zorder=2, edgecolor="white", linewidth=0.5)
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(df: pd.DataFrame) -> None:
    """
    Plots the distribution of all categorical features in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the data to be plotted.

    Returns:
        None
    """
    categorical_features = df.select_dtypes(include=["category", "object"])
    num_features = len(categorical_features.columns)
    plots_per_row = 3
    num_rows = math.ceil(num_features / plots_per_row)
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(15, 4 * num_rows))
    axes = axes.flatten()
    for i, col in enumerate(categorical_features.columns):
        sns.countplot(x=df[col], ax=axes[i], zorder=2)
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def bootstrap_median_diff(df: pd.DataFrame, target_feature: str, predictor: str, num_samples: int = 10000) -> float:
    """
    Perform a bootstrap test to compare the difference in medians between two groups.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_feature (str): The name of the target feature column.
        predictor (str): The name of the predictor feature column.
        num_samples (int): The number of bootstrap samples to generate (default is 10,000).

    Returns:
        float: The p-value of the bootstrap test.
    """
    categories = df[target_feature].unique()
    data_cat1 = df[df[target_feature] == categories[0]][predictor]
    data_cat2 = df[df[target_feature] == categories[1]][predictor]
    observed_diff = np.median(data_cat1) - np.median(data_cat2)
    combined = np.concatenate([data_cat1, data_cat2])
    boot_diffs = []
    for _ in range(num_samples):
        boot_cat1 = np.random.choice(combined, size=len(data_cat1), replace=True)
        boot_cat2 = np.random.choice(combined, size=len(data_cat2), replace=True)
        boot_diff = np.median(boot_cat1) - np.median(boot_cat2)
        boot_diffs.append(boot_diff)
    p_value = np.sum(np.abs(boot_diffs) >= np.abs(observed_diff)) / num_samples
    return p_value


def numeric_relationships(df: pd.DataFrame, target_feature: str) -> None:
    """
    Analyze the relationship between numeric predictors and a target feature.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_feature (str): The name of the target feature column.

    The function will:
        1. Generate histogram and boxplot visuals for each numeric predictor by target feature status.
        2. Perform a bootstrap hypothesis test for the difference in medians.

    Returns:
        None
    """
    selected_columns = df.select_dtypes(include=["number"]).columns.tolist()
    selected_columns.append(target_feature)
    temp_df = df[selected_columns].copy()
    for predictor in selected_columns:
        if predictor != target_feature:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(
                f"Distribution of {predictor} by {target_feature} Status",
                fontsize=13,
            )
            sns.histplot(
                data=temp_df,
                x=predictor,
                hue=target_feature,
                multiple="stack",
                edgecolor="white",
                ax=axs[0],
                zorder=2,
                linewidth=0.5,
                alpha=0.98,
            )
            axs[0].set_xlabel(predictor)
            axs[0].set_ylabel("Frequency")
            sns.boxplot(
                data=temp_df,
                y=predictor,
                x=target_feature,
                ax=axs[1],
                zorder=2,
            )
            axs[1].set_ylabel(predictor)
            axs[1].set_xlabel(target_feature)
            plt.tight_layout()
            plt.show()
            p_value = bootstrap_median_diff(df, target_feature, predictor)
            print(f"{predictor} - {target_feature}:")
            print(f"Bootstrap hypothesis test p-value: {p_value:.2f}")

def categorical_relationships(df: pd.DataFrame, target_feature: str) -> None:
    """
    Analyzes the relationship between categorical predictors and a target feature.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_feature (str): The name of the target feature column.

    The function will:
        1. Generate bar plots for each categorical predictor by target feature status.
        2. Perform Chi-square hypothesis test.

    Returns:
        None
    """
    selected_columns = df.select_dtypes(include=["category", "object"]).columns.tolist()
    if target_feature not in selected_columns:
        selected_columns.append(target_feature)
    temp_df = df[selected_columns].copy()
    for predictor_name in temp_df.columns:
        if predictor_name != target_feature:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(
                f"Distribution of {predictor_name} by {target_feature} Status",
                fontsize=13,
            )
            crosstab = pd.crosstab(temp_df[predictor_name], temp_df[target_feature])
            crosstab.plot(
                kind="bar",
                stacked=True,
                ax=axs[0],
                edgecolor="white",
                linewidth=0.5,
            )
            axs[0].grid(axis="x")
            axs[0].set_xlabel(predictor_name)
            axs[0].set_ylabel("Count")
            axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=0)
            for patch in axs[0].patches:
                patch.set_zorder(2)
            crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0)
            crosstab_normalized.plot(
                kind="bar", stacked=True, ax=axs[1], edgecolor="white", linewidth=0.5
            )
            axs[1].grid(axis="x")
            axs[1].set_ylabel("Proportion")
            axs[1].set_xlabel(predictor_name)
            axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=0)
            axs[1].legend().set_visible(False)
            for patch in axs[1].patches:
                patch.set_zorder(2)
            plt.tight_layout()
            plt.show()
            _, p_value, _, _ = chi2_contingency(crosstab)
            print(
                f"{predictor_name} - {target_feature}:\nChi-Square test p-value: {p_value:.2f}\n"
            )

def ordinal_relationships(df: pd.DataFrame, target_feature: str) -> None:
    """
    Analyze the relationship between ordinal predictors and a target feature.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_feature (str): The name of the target feature column.

    The function will:
        1. Generate histogram and boxplot visuals for each ordinal predictor by target feature status.
        2. Perform a bootstrap hypothesis test for the difference in medians.

    Returns:
        None
    """
    selected_columns = df.select_dtypes(include=["number"]).columns.tolist()
    selected_columns.append(target_feature)
    temp_df = df[selected_columns].copy()
    temp_df = temp_df.dropna()
    for predictor in selected_columns:
        if predictor != target_feature:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(
                f"Distribution of {predictor} by {target_feature} Status",
                fontsize=13,
            )
            temp_df_counts = temp_df.groupby([predictor, target_feature]).size().unstack()
            temp_df_counts.index = temp_df_counts.index.astype(int)
            temp_df_counts.plot(kind='bar', stacked=True, ax=axs[0], zorder=2)
            axs[0].grid(axis='x')
            axs[0].set_xlabel(predictor)
            axs[0].set_ylabel("Frequency")
            axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=0)
            temp_df[predictor] = temp_df[predictor].astype(int)
            crosstab = pd.crosstab(temp_df[predictor], temp_df[target_feature])
            crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0)
            crosstab_normalized.plot(
                kind="bar", stacked=True, ax=axs[1], edgecolor="white", linewidth=0.5, zorder=2
            )
            axs[1].grid(axis='x')
            axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=0)
            axs[1].set_xlabel(predictor)
            axs[1].set_ylabel(target_feature)
            plt.tight_layout()
            plt.show()
            p_value = bootstrap_median_diff(df, target_feature, predictor)
            print(f"{predictor} - {target_feature}:")
            print(f"Bootstrap hypothesis test p-value: {p_value:.2f}")

def phik_heatmap(df: pd.DataFrame, size: float = 1) -> None:
    """
    Generate a heatmap of the Phik correlation matrix for the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        size (float, optional): Scaling factor for the figure size. Default is 1.

    Returns:
        None
    """
    colors = ["#4C72B0", "#DD8452"]
    n_bins = [0, 1]
    cmap = LinearSegmentedColormap.from_list("custom_blue", list(zip(n_bins, colors)))
    numeric_features = df.select_dtypes(include=["float64"])
    phik_matrix = df.phik_matrix(interval_cols=numeric_features)
    mask = np.triu(np.ones_like(phik_matrix, dtype=bool))
    plt.figure(figsize=(7 * size, 5 * size))
    sns.heatmap(
        phik_matrix,
        mask=mask,
        annot=True,
        cmap=cmap,
        fmt=".2f",
        linewidths=0.5,
        cbar=False,
    )
    plt.grid(False)
    plt.title(r"$\phi_K$ Correlation Heatmap Between All Variables")

def plot_confusion(
    y_test: pd.Series,
    y_test_pred: np.ndarray,
) -> None:
    """
    Plots the confusion matrix for test dataset.

    Args:
        y_test (pd.Series): Actual test data.
        y_test_pred (np.ndarray): Predicted values for testing set.

    Returns:
        None
    """
    colors = ["#FFFFFF", "#DD8452"]
    n_bins = [0, 1]
    cmap = LinearSegmentedColormap.from_list("custom_blue", list(zip(n_bins, colors)))
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(3, 3))
    sns.heatmap(
        conf_matrix_test,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        linewidths=0.7,
    )
    plt.grid(False)
    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
