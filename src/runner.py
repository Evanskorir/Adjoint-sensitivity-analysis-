import os

import torch

from src.gradient.sensitivity_calculator import SensitivityCalculator
from src.plotter import Plotter
from src.static.dataloader import DataLoader


class Runner:
    def __init__(self, data: DataLoader):
        """
        Initialize the simulation with the provided data.
        Args: data (DataLoader): DataLoader object containing age data and model params.
        """
        self.data = data
        self.population = self.data.age_data
        self.n_age = len(self.data.age_data)
        self.params = self.data.model_parameters_data

        self.sensitivity_calc = SensitivityCalculator(data=self.data)
        self.r0_cm_grad = None

        # User-defined parameters
        self.susc_choices = [0.5, 1.0]
        self.r0_choices = [1.2, 1.8, 2.5]
        self.scales = ["pop_sum", "contact_sum", "no_scale"]

    def run(self):
        """
        Run the simulation to perform contact matrix manipulations,
        calculate eigenvalues and gradients.
        """
        for scale in self.scales:
            susceptibility = torch.ones(self.n_age)
            for susc in self.susc_choices:
                susceptibility[:4] = susc
                self.params.update({"susc": susceptibility})

                self.sensitivity_calc.run(scale=scale, params=self.params)

                self.create_plots(scale=scale, susc=susc)

    def create_plots(self, scale: str, susc: float):
        for base_r0 in self.r0_choices:
            beta = base_r0 / self.sensitivity_calc.eigen_value
            # Scale the gradients with beta
            self.r0_cm_grad = beta * self.sensitivity_calc.eigen_value_gradient.eig_val_cm_grad

            # 6. Generate the plots from plotter to visualize the gradients with
            # different base r0's and susc
            # Create folder with base R0 and susc
            folder = f"generated/results_base_r0_{base_r0:.1f}_susc_{susc:.1f}"
            os.makedirs(folder, exist_ok=True)

            # Create sub folder for the scale
            scale_folder = os.path.join(folder, scale)
            os.makedirs(scale_folder, exist_ok=True)

            plot = Plotter(n_age=self.n_age)
            plot.plot_contact_input(contact_input=self.sensitivity_calc.contact_input,
                                    plot_title=f"Contact input "
                                               f"using {scale} scaling",
                                    filename="contact_input.pdf",
                                    folder=scale_folder)

            matrices = [
                (self.sensitivity_calc.ngm_small_tensor, "NGM with Small Domain", "ngm_heatmap.pdf"),
                (self.sensitivity_calc.symmetric_contact_matrix,
                 "Symmetrized Contact " "Input Matrix", "CM.pdf"),
                (plot.reconstruct_plot_symmetric_grad_matrix(self.r0_cm_grad),
                 "Gradient values", "Grads.pdf")
            ]

            for matrix, plot_title, filename in matrices:
                plot.plot_small_ngm_contact_grad_mtx(
                    matrix=matrix,
                    plot_title=plot_title,
                    filename=filename,
                    folder=scale_folder)
