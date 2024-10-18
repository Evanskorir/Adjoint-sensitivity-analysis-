from src.gradient.eigen_value_gradient import EigenValueGradient
from src.comp_graph.cm_creator import CMCreator
from src.comp_graph.cm_elements_cg_leaf import CMElementsCGLeaf
from src.gradient.ngm_gradient import NGMGradient
from src.static.cm_leaf_preparator import CGLeafPreparator
from src.static.dataloader import DataLoader
from src.static.eigen_calculator import EigenCalculator

# Import NGMCalculators from different models
from src.models.chikina.ngm_calculator import NGMCalculator as ChikinaNGMCalculator
from src.models.moghadas.ngm_calculator import NGMCalculator as MoghadasNGMCalculator
from src.models.seir.ngm_calculator import NGMCalculator as SeirNGMCalculator
from src.models.rost.ngm_calculator import NGMCalculator as RostNGMCalculator
from src.models.italy.ngm_calculator import NGMCalculator as ItalyNGMCalculator
from src.models.kenya.ngm_calculator import NGMCalculator as KenyaNGMCalculator
from src.models.british_columbia.ngm_calculator import NGMCalculator as BCNGMCalculator


class SensitivityCalculator:
    def __init__(self, data: DataLoader, model: str):
        self.data = data
        self.population = self.data.age_data
        self.n_age = len(self.data.age_data)
        self.model = model

        # Initialize variables to store calculated values
        self.ngm_calculator = None
        self.ngm_small_tensor = None
        self.ngm_small_grads = None
        self.eigen_value_gradient = None
        self.eigen_value = None
        self.eigen_vector = None
        self.contact_input = None
        self.scale_value = None
        self.symmetric_contact_matrix = None

        self._select_ngm_calculator()

    def _select_ngm_calculator(self):
        """
        Select the appropriate NGMCalculator based on the model name
        """
        if self.model == "chikina":
            self.ngm_calculator_class = ChikinaNGMCalculator
        elif self.model == "moghadas":
            self.ngm_calculator_class = MoghadasNGMCalculator
        elif self.model == "seir":
            self.ngm_calculator_class = SeirNGMCalculator
        elif self.model == "rost":
            self.ngm_calculator_class = RostNGMCalculator
        elif self.model == "italy":
            self.ngm_calculator_class = ItalyNGMCalculator
        elif self.model == "kenya":
            self.ngm_calculator_class = KenyaNGMCalculator
        elif self.model == "british_columbia":
            self.ngm_calculator_class = BCNGMCalculator
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def run(self, scale: str, params: dict):
        """
        Main function to calculate sensitivity using various steps
        """
        # 1. Initialize NGM calculator with parameters
        self.ngm_calculator = self.ngm_calculator_class(n_age=self.n_age, param=params)

        # 2. Create leaf of the computation graph
        self._create_leaf(scale)

        # 3. Perform contact matrix manipulations
        self._contact_matrix_manipulation(scale)

        # 4. Compute the next generation matrix (NGM)
        self.ngm_calculator.run(symmetric_contact_mtx=self.symmetric_contact_matrix)
        self.ngm_small_tensor = self.ngm_calculator.ngm_small_tensor

        # 5. Calculate gradients of the NGM
        self._calculate_ngm_gradients()

        # 6. Calculate eigenvalue and eigenvalue gradients for R0
        self._calculate_eigenvectors()

    def _create_leaf(self, scale: str):
        """
        Create the computation graph leaf (CG leaf) from the original contact matrix
        """
        cg_leaf_preparator = CGLeafPreparator(data=self.data, model=self.model)
        cg_leaf_preparator.run()
        transformed_total_orig_cm = cg_leaf_preparator.transformed_total_orig_cm

        cm_elements_cg_leaf = CMElementsCGLeaf(
            n_age=self.n_age,
            transformed_total_orig_cm=transformed_total_orig_cm,
            pop=self.population
        )
        cm_elements_cg_leaf.run(scale=scale)
        self.contact_input = cm_elements_cg_leaf.contact_input.requires_grad_(True)
        self.scale_value = cm_elements_cg_leaf.scale_value

    def _contact_matrix_manipulation(self, scale: str):
        """
        Perform manipulations on the contact matrix to produce necessary inputs
        """
        cm_creator = CMCreator(
            n_age=self.n_age,
            pop=self.population.reshape((-1, 1))
        )
        cm_creator.run(
            contact_matrix=self.contact_input,
            scale_value=self.scale_value,
            scale=scale
        )
        self.symmetric_contact_matrix = cm_creator.cm

    def _calculate_ngm_gradients(self):
        """
        Calculate the gradients of the next generation matrix (NGM)
        """
        ngm_grad = NGMGradient(
            ngm_small_tensor=self.ngm_small_tensor,
            contact_input=self.contact_input
        )
        ngm_grad.run()
        self.ngm_small_grads = ngm_grad.ngm_small_grads

    def _calculate_eigenvectors(self):
        """
        Calculate the dominant eigenvalue and eigenvector, and gradients
        """
        eigen_calculator = EigenCalculator(ngm_small_tensor=self.ngm_small_tensor)
        eigen_calculator.run()

        self.eigen_vector = eigen_calculator.dominant_eig_vec
        self.eigen_value = eigen_calculator.dominant_eig_val

        # Calculate eigenvalue gradient
        eigen_value_grad = EigenValueGradient(
            ngm_small_tensor=self.ngm_small_tensor,
            dominant_eig_vec=self.eigen_vector
        )
        eigen_value_grad.run(ngm_small_grads=self.ngm_small_grads)
        self.eigen_value_gradient = eigen_value_grad






