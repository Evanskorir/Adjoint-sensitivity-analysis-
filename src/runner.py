import os
import torch
from src.gradient.sensitivity_calculator import SensitivityCalculator
from src.plotter import Plotter
from src.static.dataloader import DataLoader


class Runner:
    def __init__(self, data: DataLoader, model):
        """
        Initialize the simulation with the provided data.
        Args: data (DataLoader): DataLoader object containing age data and model params.
        """

        self.data = data
        self.population = self.data.age_data
        self.n_age = len(self.data.age_data)
        self.params = self.data.model_parameters_data
        self.model = model

        self.sensitivity_calc = SensitivityCalculator(data=self.data, model=model)
        self.r0_cm_grad = None

        # User-defined parameters
        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.2, 1.8, 2.5]
        # self.scales = ["pop_sum", "contact_sum", "no_scale"]
        self.scales = ["pop_sum", "contact_sum"]

    def run(self):
        """
        Run the simulation to perform contact matrix manipulations,
        calculate eigenvalues, and compute gradients.
        """
        for scale in self.scales:
            susceptibility = torch.ones(self.n_age)
            for susc in self.susc_choices:
                # Update susceptibility values in the model parameters
                susceptibility[:4] = susc
                self.params.update({"susc": susceptibility})

                # Run sensitivity calculation for the current scale and parameters
                self.sensitivity_calc.run(scale=scale, params=self.params)

                # Create plots and process the results
                self.create_plots(scale=scale, susc=susc)

    def create_plots(self, scale: str, susc: float):
        """
        Generate plots for different R0 values and susceptibility choices,
        after calculating gradients and applying projection.
        """
        for base_r0 in self.r0_choices:
            # Calculate projected gradients for the current base R0
            self.calculate_projected_gradients(base_r0=base_r0)

            # Create folder structure for saving plots
            folder = f"generated/results_base_r0_{base_r0:.1f}_susc_{susc:.1f}"
            os.makedirs(folder, exist_ok=True)

            # Create subfolder for each scale
            scale_folder = os.path.join(folder, scale)
            os.makedirs(scale_folder, exist_ok=True)

            # Generate plots for contact input, gradients, NGM matrix, etc.
            self.generate_plots(scale_folder=scale_folder, base_r0=base_r0)

    def calculate_projected_gradients(self, base_r0: float):
        """
        Calculate the projected R0 gradients using the dominant eigenvalue and
        eigenvalue gradient, then apply scaling by beta (base R0 / eigenvalue).
        """
        # Compute beta as base R0 divided by the dominant eigenvalue
        beta = base_r0 / self.sensitivity_calc.eigen_value
        self.params.update({"beta": beta})

        # Scale the eigenvalue gradient by beta to get r0_cm_grad
        self.r0_cm_grad = beta * \
                          self.sensitivity_calc.eigen_value_gradient_136.eig_val_cm_grad

    def generate_plots(self, scale_folder: str, base_r0: float):
        """
        Generate various plots such as contact matrix, gradients, NGM matrix, etc.,
        and save them in the specified folder.
        Args:
            scale_folder (str): Path to the folder where plots will be saved.
            base_r0 (float): Base R0 value used for the simulation.
        """
        plot = Plotter(data=self.data, n_age=self.n_age, model=self.model)

        # Plot contact matrices
        plot.plot_contact_matrices(contact_data=self.data.contact_data,
                                   filename="contact_matrices")

        # Plot contact input matrix
        plot.plot_contact_input(
            contact_input=self.sensitivity_calc.contact_input_136,
            plot_title="",
            filename="contact_input.pdf",
            folder=scale_folder
        )

        # Plot R0 gradient matrix
        plot.plot_grads(
            grads=self.r0_cm_grad,
            plot_title=f"$\\overline{{\\mathcal{{R}}}}_0={base_r0}$",
            filename="Grads_tri.pdf",
            folder=scale_folder
        )

        matrices = [
            (self.sensitivity_calc.ngm_small_tensor,
             None, "ngm_heatmap.pdf", True),
            (self.sensitivity_calc.symmetric_contact_matrix,
             None, "CM.pdf", False),
            (plot.reconstruct_plot_symmetric_grad_matrix(self.r0_cm_grad),
             "Gradient values",
             "Grads.pdf", True)
        ]

        # Plot each matrix and save in the folder
        for matrix, plot_title, filename, label_axes in matrices:
            show_colorbar = True  # Always show color bar
            plot.plot_small_ngm_contact_grad_mtx(
                matrix=matrix,
                plot_title=plot_title,
                filename=filename,
                folder=scale_folder,
                label_axes=label_axes,
                show_colorbar=show_colorbar
            )

        # Generate and plot the percentage age group contribution bar plot
        plot.get_percentage_age_group_contact_list(
            symmetrized_cont_matrix=self.sensitivity_calc.symmetric_contact_matrix,
            filename="age_group_percentage_bar_plot",
            folder=scale_folder
        )

        # plot agg based on pca
        plot.plot_aggregated_bar_chart(
            aggregated_matrix=self.sensitivity_calc.eigen_value_gradient_16.eig_val_cm_grad,
            plot_title=f"$\\overline{{\\mathcal{{R}}}}_0={base_r0}$",
            filename="agg",
            folder=scale_folder
        )