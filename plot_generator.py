# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class PlotGenerator:
    def __init__(self, csv_file):
        """
        Initialize the PlotGenerator with a CSV file containing experiment results.

        :param csv_file: Path to the CSV file with experiment results.
        """
        self.csv_file = csv_file
        self.data = None

    def load_data(self):
        """
        Load experiment data from the CSV file.
        """
        try:
            self.data = pd.read_csv(self.csv_file)
            print(self.data)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def generate_lineplot(self, x_column, y_columns, hue=None, title="Plot", output=""):
        """
        Generate a plot based on the provided x column and multiple y columns.

        :param x_column: The column to be used as the x-axis.
        :param y_columns: A list of columns to be used as the y-axis.
        :param hue: Optional. Column to differentiate data with color (e.g., Alteration Type).
        :param title: Title of the plot.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return

        plt.figure(figsize=(10, 6))

        for y_column in y_columns:
            sns.lineplot(data=self.data, x=x_column, y=y_column, label=y_column, marker="o")

        plt.title(title)
        plt.xlabel(x_column)
        plt.ylabel("Values")
        plt.legend(title="Metrics")
        plt.savefig(output)
        plt.close()

    def generate_regplot(self, x_column, y_columns, title="Plot", output=""):
        """
        Generate a plot based on the provided x and y columns.

        :param x_column: The column to be used as the x-axis.
        :param y_column: The column to be used as the y-axis.
        :param hue: Optional. Column to differentiate data with color (e.g., Alteration Type).
        :param title: Title of the plot.
        """
        if self.data is None:
            print("Data not loaded. Please load the data first.")
            return
        

        plt.figure(figsize=(10,6))
        plt.title(title)
        for y_column in y_columns:
            sns.regplot(data=self.data, x=x_column, y=y_column, label=y_column)
        plt.xlabel(x_column)
        plt.ylabel("Similarity/Score (%)")
        plt.legend(title="Metrics")
        plt.savefig(output)
        plt.close()
        

    


if __name__ == "__main__":
    # Example usage
    plot_generator = PlotGenerator("test2.csv")
    plot_generator.load_data()
    # plot_generator.generate_lineplot(x_column="Alteration Count",
    #                                 y_columns=["F1-Score"],
    #                                 hue="Alteration Type", 
    #                                 title="Metrics vs Alteration Count")
    
    # plot_generator.generate_lineplot(x_column="Alteration Count", 
    #                                  y_columns=["Node Similarity", "Structural Similarity", "Behavioral Similarity"],
    #                                  hue="Alteration Type", 
    #                                  title="Node/Structural/Behavioral Similarity vs Alteration Count")

    plot_generator.generate_regplot(x_column="Alteration Count", 
                                     y_columns=["Node Similarity", "Structural Similarity", "Behavioral Similarity"],
                                     title="Node/Structural/Behavioral Similarity vs Alteration Count (ChangeLabel)",
                                     output="output_plot1.png")

    plot_generator.generate_regplot(x_column="Alteration Count", 
                                     y_columns=["Behavioral Similarity", "F1-Score"],
                                     title="Metrics vs Alteration Count (ChangeLabel)",
                                     output="output_plot2.png")                                 
    
    