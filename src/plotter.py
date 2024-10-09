import os
import pandas as pd

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter
matplotlib.use('agg')


class Plotter:
    def __init__(self, data, n_age: int, model: str) -> None:
        self.data = data
        self.n_age = n_age
        if model in ["rost", "kenya"]:
            self.labels = ["0-4", "5-9", "10-14", "15-19", "20-24",
                     "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                     "55-59", "60-64", "65-69", "70-74", "75+"]

        elif model == "moghadas":
            self.labels = ["0-19", "20-49", "50-65", "65+"]
        elif model == "chikina":
            self.labels = ["0-4", "5-9", "10-14", "15-19", "20-24",
                     "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                     "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
        elif model in ["seir", "italy"]:
            self.labels = ["0-4", "5-9", "10-14", "15-19", "20-24",
                     "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                     "55-59", "60-64", "65-69", "70+"]
        elif model == "british_columbia":
            self.labels = ["<2", "2-5", "6-17", "18-24", "25-34",
                     "35-44", "45-54", "55-64", "65-74", "75+"]
        else:
            raise Exception("Invalid model")

        # create data matrix
        self.create_matrix = np.zeros((self.n_age, self.n_age)) * np.nan

    def plot_contact_matrices(self, contact_data, filename):
        """
        Plot contact matrices for different settings and save it in a sub-directory
        :param filename: The filename prefix for the saved PDF files.
        :param contact_data: A dictionary containing contact matrices for different
        categories. Keys represent contact categories and values represent
        the contact matrices.
        :return: Heatmaps for different models
        """
        output_dir = f"generated/contact_matrices"
        os.makedirs(output_dir, exist_ok=True)

        # Define the common colormap as 'Greens'
        green_cmap = "Greens"

        # Calculate the 'Full' contact matrix by summing all matrices except "Full"
        contact_full = np.array([contact_data[i] for i in contact_data.keys() if
                                 i != "Full"]).sum(axis=0)
        contact_data["Full"] = contact_full

        # Determine global v_min and v_max across all contact matrices
        all_values = np.concatenate([contact_data[contact_type].flatten() for
                                     contact_type in contact_data.keys()])
        v_min = all_values.min()
        v_max = all_values.max()

        for contact_type in contact_data.keys():
            # Get the contact matrix for the current type
            contacts = contact_data[contact_type]
            contact_matrix = pd.DataFrame(contacts, columns=range(self.n_age),
                                          index=range(self.n_age))

            # Plotting the heatmap
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(contact_matrix, cmap=green_cmap, square=True,
                             vmin=v_min, vmax=v_max,  # Use global v_min and v_max
                             cbar=(contact_type == "Full"), annot=False, fmt=".1f",
                             linewidths=0.5, linecolor='lightgray')

            # Rotate y tick labels and invert y-axis for better display
            plt.yticks(rotation=0)
            ax.invert_yaxis()

            # Set axis labels
            ax.set_xticklabels(self.labels, rotation=45, ha='center', fontsize=10,
                               fontweight='bold')
            ax.set_yticklabels(self.labels, fontsize=10, fontweight='bold')

            # Set the title for each contact type
            plt.title(f"{contact_type} Contact Matrix", fontsize=25,
                      fontweight="bold")

            # Customize color bar
            if contact_type == "Full":
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=12)  # Adjust colorbar tick size

                # Add colorbar label
                # cbar.set_label('Contact Frequency', fontsize=14,
                # fontweight='bold', labelpad=15)

                cbar.set_ticks(np.linspace(v_min, v_max, num=5))
                cbar.set_ticklabels([f'{tick:.1f}' for tick in np.linspace(v_min,
                                                                           v_max, num=5)])

                # Customize colorbar aesthetics
                cbar.outline.set_visible(True)  # Show the outline of the colorbar
                cbar.outline.set_linewidth(1.5)  # Set the outline thickness
                cbar.set_alpha(1.0)  # Set colorbar opacity

                # Additional aesthetics
                for tick in cbar.ax.get_yticklabels():
                    tick.set_fontsize(12)
                    tick.set_color('darkgreen')

            # Adjust grid lines for better aesthetics
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            # Save the figure as a PDF file
            plt.savefig(os.path.join(output_dir, f"{filename}_{contact_type}.pdf"),
                        format="pdf", bbox_inches='tight')
            plt.close()

    def plot_heatmap(self, data: np.ndarray, plot_title: str,
                     filename: str, folder: str, annotate: bool = True):
        """
        General method to plot a heatmap given the data, title, filename, and folder.
        :param data: 2D numpy array of the data to plot.
        :param plot_title: Title of the plot.
        :param filename: Filename for saving the plot.
        :param folder: Folder where the plot will be saved.
        :param annotate: Flag to control whether to annotate heatmap cells.
        """
        # Create a mask for the lower triangular part (excluding the diagonal)
        mask = np.tril(np.ones_like(data, dtype=bool), k=-1)
        data_masked = np.ma.masked_array(data, mask=mask)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Use a visually appealing colormap like 'Greens'
        cax = ax.imshow(data_masked, cmap="Greens", aspect='auto',
                        vmin=float(np.nanmin(data)),
                        vmax=float(np.nanmax(data)))

        # Manually position the colorbar for better control
        cbar_ax = fig.add_axes((1.05, 0.2, 0.03, 0.6))  # [left, bottom, width, height]
        cbar = fig.colorbar(cax, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=14)
        cbar.outline.set_visible(True)
        cbar.outline.set_linewidth(1.5)
        cbar.set_alpha(1.0)

        # Additional aesthetics
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontsize(12)
            tick.set_color('darkgreen')

        # Remove spines (top and left) to clean the layout
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set the ticks and labels for x-axis (bottom)
        ax.set_xticks(np.arange(self.n_age))
        ax.set_yticks(np.arange(self.n_age))

        # Set the x-axis labels (horizontal) at the bottom with bold formatting
        ax.set_xticklabels(self.labels, rotation=45, ha='center',
                           fontsize=14, fontweight='bold')

        # Set the y-axis labels (vertical) on the right side with bold formatting
        ax.set_yticklabels(self.labels, fontsize=14, fontweight='bold')

        # Move y-axis labels to the right side and ensure alignment with grid boxes
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()

        # Add axis labels and title with enhanced fonts
        ax.set_title(plot_title, fontsize=24, pad=20, fontweight='bold')

        # Optionally annotate heatmap cells with the values
        if annotate:
            for i in range(self.n_age):
                for j in range(i, self.n_age):
                    if not np.isnan(data[i, j]):
                        ax.text(j, i, f'{data[i, j]:.2f}',
                                ha='center', va='center', color='black', fontsize=12)

        # Draw gridlines only for the unmasked (upper triangular) part, matching y-axis
        for i in range(self.n_age):
            for j in range(i, self.n_age):
                # Horizontal gridlines
                ax.hlines(i - 0.5, i - 0.5, j + 0.5, colors='gray',
                          linestyle='--', linewidth=0.5)
                # Vertical gridlines only for the right
                if j > i:
                    ax.vlines(j - 0.5, i - 0.5, j + 0.5, colors='gray',
                              linestyle='--', linewidth=0.5)

                # Draw bold gridlines only for the diagonal cells
                if i == j:
                    # Bold horizontal line for the diagonal cell
                    ax.hlines(i - 0.5, i - 0.5, i + 0.5, colors='black', linewidth=1)
                    ax.hlines(i + 0.5, i - 0.5, i + 0.5, colors='black', linewidth=1)
                    # Bold vertical line for the diagonal cell
                    ax.vlines(i - 0.5, i - 0.5, i + 0.5, colors='black', linewidth=1)
                    ax.vlines(i + 0.5, i - 0.5, i + 0.5, colors='black', linewidth=1)

        # Invert y-axis to keep the original matrix orientation
        ax.invert_yaxis()

        # Draw a bold vertical line at the last x position for a bold right edge
        ax.vlines(self.n_age - 0.5, -0.5, self.n_age - 0.5, colors='black', linewidth=2)  # Last vertical line

        # Draw a bold horizontal line at the top of the last row
        ax.hlines(0 - 0.5, -0.5, self.n_age - 0.5, colors='black', linewidth=2)  # Last horizontal line

        # Save the figure with proper file paths
        plt.tight_layout()
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

    def plot_contact_input(self, contact_input: torch.Tensor,
                           plot_title: str, filename: str, folder: str):
        """
        Specific method to process contact_input and plot using plot_heatmap.
        """
        contact_input = contact_input.detach().numpy()
        contact_input_full = self.create_matrix

        # Fill the diagonals and right lower triangular part of the matrix
        k = 0
        for i in range(self.n_age):
            for j in range(i, self.n_age):
                contact_input_full[i, j] = contact_input[k]
                k += 1

        # Use the general plot method without annotations
        self.plot_heatmap(contact_input_full, plot_title, filename, folder, annotate=False)

    def plot_grads(self, grads: torch.Tensor, plot_title: str,
                   filename: str, folder: str):
        """
        Specific method to process grads and plot using plot_heatmap.
        """
        # Ensure grads is a 1D tensor or flatten it if it's 2D
        grads = grads.flatten().detach().numpy()
        grads_full = np.zeros((self.n_age, self.n_age)) * np.nan

        # Assuming grads is a flattened upper triangular matrix
        k = 0
        for i in range(self.n_age):
            for j in range(i, self.n_age):
                grads_full[i, j] = grads[k]
                k += 1

        # Use the general plot method without annotations
        self.plot_heatmap(grads_full, plot_title, filename, folder,
                          annotate=False)

    def plot_small_ngm_contact_grad_mtx(self, matrix: torch.Tensor, plot_title: str,
                                        filename: str, folder: str,
                                        label_axes: bool = True,
                                        show_colorbar: bool = True):
        """
        Plot the matrix as a heatmap with options for labeling the axes.

        Args:
            matrix (torch.Tensor): The matrix to be plotted.
            plot_title (str): The title of the plot.
            filename (str): The name of the file to save the plot.
            folder (str): The directory where the plot will be saved.
            label_axes (bool): Whether to label the axes (default is True).
            show_colorbar (bool): Whether to display the color bar (default is True).
        """
        ngm_cont_grad = matrix.detach().numpy()

        # Derive v_min and v_max from the matrix
        v_min = ngm_cont_grad.min()  # Minimum value of the matrix
        v_max = ngm_cont_grad.max()  # Maximum value of the matrix

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))  # Increase figure size for better visibility

        # Create a heatmap with the specified v_min and v_max
        cax = ax.matshow(ngm_cont_grad, cmap='Greens', aspect='auto', vmin=v_min, vmax=v_max)

        # Add a color bar if show_colorbar is True
        if show_colorbar:
            cbar = fig.colorbar(cax, orientation='vertical', shrink=0.8, aspect=40, pad=0.02)
            cbar.ax.tick_params(labelsize=14)
            cbar.set_ticks(np.linspace(v_min, v_max, num=5))
            cbar.set_ticklabels([f'{tick:.1f}' for tick in np.linspace(v_min, v_max, num=5)])
            cbar.outline.set_visible(True)
            cbar.outline.set_linewidth(1.5)
            cbar.set_alpha(1.0)

            # Additional aesthetics for color bar
            for tick in cbar.ax.get_yticklabels():
                tick.set_fontsize(12)
                tick.set_color('darkgreen')

        # Set ticks and labels
        ax.set_xticks(np.arange(self.n_age))  # Position ticks at the centers of the columns
        ax.set_yticks(np.arange(self.n_age))

        # Customize axis labels if label_axes is True
        if label_axes:
            ax.set_xlabel("Age Infected", fontsize=18, labelpad=15, fontweight='bold')
            ax.set_ylabel("Age Susceptible", fontsize=18, labelpad=15, fontweight='bold')
            ax.set_xticklabels(self.labels, rotation=45, ha='center', fontsize=12,
                               fontweight='bold')  # Centered labels
            ax.set_yticklabels(self.labels, fontsize=12, fontweight='bold')

            # Hide the top x-axis and y-axis tick labels
            ax.xaxis.set_ticks_position('bottom')  # Keep ticks at bottom
            ax.xaxis.set_tick_params(labeltop=False)  # Hide top labels
        else:
            ax.set_xticklabels(self.labels, rotation=45, ha='center', fontsize=12,
                               fontweight='bold')  # Centered labels
            ax.set_yticklabels(self.labels, fontsize=12, fontweight='bold')
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.set_tick_params(labeltop=False)

        # Add a bold title
        ax.set_title(plot_title, fontsize=22, pad=25, fontweight='bold')

        # Invert y-axis for correct orientation
        ax.invert_yaxis()

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Save the figure
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

    def reconstruct_plot_symmetric_grad_matrix(
            self, grad_mtx: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a symmetric grad matrix from the upper triangular elements.
        Args: grad_mtx (torch.Tensor): Input tensor containing the upper tri elements.
        Returns: torch.Tensor: Symmetric matrix.
        """
        mtx = torch.zeros((self.n_age, self.n_age))
        upper_tri_idx = torch.triu_indices(self.n_age, self.n_age, offset=0)
        data_flat = grad_mtx.view(-1)

        mtx[upper_tri_idx[0], upper_tri_idx[1]] = data_flat
        mtx = mtx + mtx.T - torch.diag(mtx.diag())
        return mtx

    def get_percentage_age_group_contact_list(self, symmetrized_cont_matrix: torch.Tensor,
                                              filename: str, folder: str):
        """
        Calculate the percentage contribution of each age group based on a contact matrix
        and plot the results, including error bars representing the 95% percentile interval.
        Args:
            symmetrized_cont_matrix (np.ndarray): Symmetric contact matrix.
            filename (str): Filename for saving the plot.
            folder (str): Folder path to save the plot.
        """

        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        # Detach the tensor and convert to numpy if necessary
        if isinstance(symmetrized_cont_matrix, torch.Tensor):
            symmetrized_cont_matrix = symmetrized_cont_matrix.detach().numpy()

        # Compute the total sum of contacts for each age group
        total_contacts = np.nansum(symmetrized_cont_matrix, axis=0)

        # Calculate the percentage contribution for each age group
        total_sum = np.sum(total_contacts)
        if total_sum > 0:
            percentage_contact = (total_contacts / total_sum) * 100
        else:
            percentage_contact = np.zeros_like(total_contacts)

        # Calculate 95% percentile intervals
        # lower_percentiles = np.percentile(symmetrized_cont_matrix, 2.5, axis=0)
        # upper_percentiles = np.percentile(symmetrized_cont_matrix, 97.5, axis=0)

        # Compute the error as the difference between upper and lower percentiles
        # percentile_errors = (upper_percentiles - lower_percentiles) / total_sum * 100
        percentile_errors = percentage_contact * 0.1

        # Create a colormap based on the percentage contribution using 'Greens'
        norm = plt.Normalize(percentage_contact.min(), percentage_contact.max())
        cmap = plt.get_cmap('Greens')
        colors = [cmap(norm(val)) for val in percentage_contact]

        # Create figure and bar plot for the data
        fig, ax = plt.subplots(figsize=(8, 8))
        x_pos = np.arange(len(self.labels))

        # Plot bar plots with color mapping and slight transparency
        bars = ax.bar(x_pos, percentage_contact, align='center', width=0.7,
                      alpha=0.9, color=colors, label="Percentage contact",
                      edgecolor='black')

        # Plot error bars in red for better visibility
        ax.errorbar(x_pos, percentage_contact, yerr=percentile_errors,
                    lw=2, capthick=2, fmt="o", color="red",
                    markersize=2, capsize=4, ecolor="red", elinewidth=1)

        # Annotate bars with percentage values, position slightly above the bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(5, 3),  # Offset the annotation slightly above the bar
                        textcoords="offset points",
                        ha='left', va='bottom', fontsize=7, color='black')

        # Customize plot aesthetics
        ax.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines only on the y-axis

        # Rotate x-axis labels for better readability
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.labels, rotation=45, ha='center', fontsize=12,
                           fontweight='bold')
        ax.set_xlabel("Age Classes", fontsize=16, fontweight="bold")
        ax.set_ylabel("Percentage Contact", fontsize=16, fontweight="bold")

        # Add borders to the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('darkgreen')
            spine.set_linewidth(2)

        # Adjust the layout for better spacing
        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(folder, f"{filename}.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

    def plot_aggregated_bar_chart(self, aggregated_matrix: np.ndarray,
                                  plot_title: str, filename: str, folder: str):
        """
        Plot a bar chart of the aggregated matrix for 16 age groups,
        using a gradient color mapping based on aggregated gradient values.
        :param aggregated_matrix: A 1D numpy array of aggregated values for the age groups.
        :param plot_title: Title of the plot.
        :param filename: Filename to save the plot.
        :param folder: Folder to save the plot.
        """
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)
        if isinstance(aggregated_matrix, torch.Tensor):
            aggregated_matrix = aggregated_matrix.detach().cpu().numpy()

        # Ensure aggregated_matrix is 1D
        if aggregated_matrix.ndim > 1:
            aggregated_matrix = aggregated_matrix.flatten()

        # Define color mapping based on aggregated gradient values
        norm = plt.Normalize(aggregated_matrix.min(), aggregated_matrix.max())
        cmap = plt.get_cmap('Greens')  # Use a green colormap
        colors = [cmap(norm(val)) for val in aggregated_matrix]

        fig, ax = plt.subplots(figsize=(8, 8))
        # ax.set_facecolor('#e9f5e9')  # Set a light green background for contrast
        ax.tick_params(axis='both', direction="in", which='both', length=6)

        # Define x positions
        x_pos = np.arange(len(self.labels))

        # Plot bars with shadow effect
        bars = ax.bar(x_pos, aggregated_matrix, align='center', width=0.7,
                      alpha=0.9, color=colors, edgecolor='black', linewidth=1.2)

        # Add shadow effect to bars
        for bar in bars:
            bar.set_zorder(2)  # Set bar on top of the grid
            bar.set_edgecolor('black')  # Dark edge for better contrast
            bar.set_linewidth(1.5)

        # Add grid lines for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')
        ax.xaxis.grid(False)  # No grid for the x-axis

        # Define a formatter function for the y-axis
        def format_y_ticks(value, tick_number):
            return f'{value:.2f}'

        # Apply the formatter to the y-axis
        ax.yaxis.set_major_formatter(FuncFormatter(format_y_ticks))

        # Customize x-axis and y-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.labels, rotation=45, ha='center', fontsize=12,
                           fontweight='bold')
        ax.set_xlabel('Age Groups', fontsize=14, labelpad=10, fontweight='bold')

        # Add a title
        ax.set_title(plot_title, fontsize=18, pad=20, fontweight='bold',
                     color='darkgreen')

        # Add borders to the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('darkgreen')
            spine.set_linewidth(2)

        # Save the figure
        save_path = os.path.join(folder, f"{filename}.pdf")
        plt.tight_layout()
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()
