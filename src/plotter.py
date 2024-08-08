import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
matplotlib.use('agg')


class Plotter:
    def __init__(self, n_age: int) -> None:
        self.n_age = n_age

    def plot_contact_input(
            self, contact_input: torch.Tensor, plot_title: str,
            filename: str, folder: str):
        contact_input = contact_input.detach().numpy()
        contact_input_full = np.zeros((self.n_age, self.n_age)) * np.nan

        # Fill the diagonals and right lower tri part of the matrix
        k = 0
        for i in range(self.n_age):
            for j in range(i, self.n_age):
                contact_input_full[i, j] = contact_input[k]
                k += 1

        # Create a mask for the upper triangular part (excluding the diagonal)
        mask = np.triu(np.ones_like(contact_input_full, dtype=bool), k=1)
        contact_input_masked = np.ma.masked_array(contact_input_full, mask=mask)

        fig, ax = plt.subplots(figsize=(12, 10))

        # Create a heatmap with the 'Greens' colormap and apply the mask
        cax = ax.imshow(contact_input_masked, cmap='Greens', aspect='auto',
                        vmin=float(np.nanmin(contact_input_full)),
                        vmax=float(np.nanmax(contact_input_full))
                        )

        # Manually position the colorbar
        cbar_ax = fig.add_axes((1.05, 0.2, 0.02, 0.6))  # [left, bottom, width, height]
        cbar = fig.colorbar(cax, cax=cbar_ax)

        # Remove upper and left axis spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Set x and y limits to fit only the lower triangular part
        ax.set_xlim(-0.5, self.n_age - 0.5)
        ax.set_ylim(self.n_age - 0.5, -0.5)

        # Set ticks and labels dynamically based on self.n_age
        ax.set_xticks(np.arange(self.n_age))
        ax.set_yticks(np.arange(self.n_age))
        ax.set_xticklabels(np.arange(1, self.n_age + 1), fontsize=12)
        ax.set_yticklabels(np.arange(1, self.n_age + 1), fontsize=12)

        # Add axis labels and title
        labels = ["0-4", "5-9", "10-14", "15-19", "20-24",
                  "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                  "55-59", "60-64", "65-69", "70-74", "75+"]
        ax.set_xlabel("Age", fontsize=16)
        ax.set_ylabel("Age", fontsize=16)
        ax.set_xticklabels(labels, rotation=90, ha='center')
        ax.set_yticklabels(labels)
        ax.set_title(plot_title, fontsize=20, pad=20)

        # Annotate heatmap cells with the values
        for i in range(self.n_age):
            for j in range(i, self.n_age):
                if not np.isnan(contact_input_full[i, j]):
                    text = ax.text(j, i, f'{contact_input_full[i, j]:.2f}',
                                   ha='center', va='center', color='black')

        # Draw a line on the right side of the plot
        line_x = [self.n_age - 0.5, self.n_age - 0.5]
        line_y = [-0.5, self.n_age - 0.5]
        ax.plot(line_x, line_y, color='black', lw=3)

        # Adjust tick positions for the y-axis to the right side
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()

        # Adjust tick line widths for both axes
        ax.tick_params(axis='both', width=3)

        # Invert y-axis to keep the original matrix orientation
        ax.invert_yaxis()
        plt.tight_layout()

        # Save the figure
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

    def plot_small_ngm_contact_grad_mtx(
            self, matrix: torch.Tensor, plot_title: str,
            filename: str, folder: str
    ):
        fig, ax = plt.subplots(figsize=(10, 10))
        ngm_cont_grad = matrix.detach().numpy()  # Detach the tensor before converting to numpy

        # Create a heatmap with the 'Greens' colormap
        cax = ax.matshow(ngm_cont_grad, cmap='Greens', aspect='auto',
                         vmin=ngm_cont_grad.min(),
                         vmax=ngm_cont_grad.max())

        # Add color bar
        cbar = fig.colorbar(cax, orientation='vertical', shrink=1.0, aspect=50)

        # Set ticks and labels dynamically based on self.n_age
        ax.set_xticks(np.arange(self.n_age))
        ax.set_yticks(np.arange(self.n_age))
        ax.set_xticklabels(np.arange(1, self.n_age + 1), fontsize=12)
        ax.set_yticklabels(np.arange(1, self.n_age + 1), fontsize=12)

        ax.xaxis.set_label_position('bottom')  # Move x-axis labels to the bottom
        ax.yaxis.set_label_position('left')  # Ensure y-axis labels are on the left
        ax.xaxis.tick_bottom()  # Ensure x-axis ticks are on the bottom
        ax.yaxis.tick_left()
        labels = ["0-4", "5-9", "10-14", "15-19", "20-24",
                  "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                  "55-59", "60-64", "65-69", "70-74", "75+"]
        ax.set_xlabel("Age Infected", fontsize=16)
        ax.set_ylabel("Age Susceptible", fontsize=16)
        ax.set_xticklabels(labels, rotation=90, ha='center')
        ax.set_yticklabels(labels)
        ax.set_title(plot_title, fontsize=20, pad=20)
        # Annotate heatmap cells with the values
        for i in range(self.n_age):
            for j in range(self.n_age):
                text = ax.text(j, i, f'{ngm_cont_grad[i, j]:.2f}', ha='center',
                               va='center', color='black')
        ax.invert_yaxis()
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
