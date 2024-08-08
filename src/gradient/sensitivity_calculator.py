from src.gradient.eigen_value_gradient import EigenValueGradient
from src.comp_graph.cm_creator import CMCreator
from src.comp_graph.cm_elements_cg_leaf import CMElementsCGLeaf
from src.comp_graph.ngm_calculator import NGMCalculator
from src.gradient.ngm_gradient import NGMGradient
from src.static.cm_leaf_preparator import CGLeafPreparator
from src.static.dataloader import DataLoader
from src.static.eigen_calculator import EigenCalculator


class SensitivityCalculator:
    def __init__(self, data: DataLoader):
        self.data = data
        self.population = self.data.age_data
        self.n_age = len(self.data.age_data)

        self.ngm_calculator = None
        self.ngm_small_tensor = None
        self.ngm_small_grads = None
        self.eigen_value_gradient = None
        self.eigen_value = None
        self.eigen_vector = None
        self.contact_input = None
        self.scale_value = None
        self.symmetric_contact_matrix = None

    def run(self, scale: str, params: dict):
        # Starting initialization
        self.ngm_calculator = NGMCalculator(n_age=self.n_age,
                                            param=params)
        # 1. Create leaf of the computation graph
        self._create_leaf(scale)
        # 2. Perform contact matrix manipulations
        self._contact_matrix_manipulation(scale)
        # 3. Compute the next generation matrix (NGM)
        self.ngm_calculator.run(
            symmetric_contact_mtx=self.symmetric_contact_matrix)
        self.ngm_small_tensor = self.ngm_calculator.ngm_small_tensor
        # 4.a. Calculate gradients of the NGM
        ngm_grad = NGMGradient(ngm_small_tensor=self.ngm_small_tensor,
                               contact_input=self.contact_input)
        ngm_grad.run()
        self.ngm_small_grads = ngm_grad.ngm_small_grads
        # 4.b. Calculate eigenvectors
        self._calculate_eigenvectors()
        # 5. Calculate the gradient of R0 with respect to the contact matrix
        self.eigen_value_gradient = EigenValueGradient(
            ngm_small_tensor=self.ngm_small_tensor,
            dominant_eig_vec=self.eigen_vector)
        self.eigen_value_gradient.run(ngm_small_grads=self.ngm_small_grads)

    def _create_leaf(self, scale: str):
        # Original contact matrix symmetrization
        cg_leaf_preparator = CGLeafPreparator(data=self.data)
        cg_leaf_preparator.run()
        transformed_total_orig_cm = cg_leaf_preparator.transformed_total_orig_cm

        # Extract upper triangular elements of the contact matrix
        cm_elements_cg_leaf = CMElementsCGLeaf(
            n_age=self.n_age,
            transformed_total_orig_cm=transformed_total_orig_cm,
            pop=self.population)
        cm_elements_cg_leaf.run(scale=scale)
        self.contact_input = cm_elements_cg_leaf.contact_input.requires_grad_(True)
        self.scale_value = cm_elements_cg_leaf.scale_value

    def _contact_matrix_manipulation(self, scale: str):
        """
        Perform manipulations on contact matrices to generate the necessary inputs.
        """
        # Create a new symmetric contact matrix
        cm_creator = CMCreator(n_age=self.n_age,
                               pop=self.population.reshape((-1, 1)))
        cm_creator.run(contact_matrix=self.contact_input,
                       scale_value=self.scale_value,
                       scale=scale)
        self.symmetric_contact_matrix = cm_creator.cm

    def _calculate_eigenvectors(self):
        """
        Calculate eigenvalues and gradients for the simulation.
        """
        # Calculate the dominant eigenvalue and eigenvector
        eigen_calculator = EigenCalculator(ngm_small_tensor=self.ngm_small_tensor)
        eigen_calculator.run()
        self.eigen_vector = eigen_calculator.dominant_eig_vec
        self.eigen_value = eigen_calculator.dominant_eig_val
