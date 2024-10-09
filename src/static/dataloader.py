import json
import os
import torch
import xlrd

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class DataLoader:
    def __init__(self, model: str):
        self.model = model.lower()  # Standardize country string

        # Dictionary mapping for the models and corresponding data files
        model_data_files = {
            "rost": {
                "model_parameters": "rost_model_parameters.json",
                "contact_matrices": "rost_contact_matrices.xls",
                "age_distribution": "rost_age_distribution.xls"
            },
            "chikina": {
                "model_parameters": "chikina_model_parameters.json",
                "contact_matrices": "chikina_contact_matrices.xls",
                "age_distribution": "chikina_age_distribution.xls"
            },
            "moghadas": {
                "model_parameters": "moghadas_model_parameters.json",
                "contact_matrices": "moghadas_contact_matrices.xls",
                "age_distribution": "moghadas_age_distribution.xls"
            },
            "seir": {
                "model_parameters": "seir_model_parameters.json",
                "contact_matrices": "seir_contact_matrices.xls",
                "age_distribution": "seir_age_distribution.xls"
            },
            "italy": {
                "model_parameters": "italy_model_parameters.json",
                "contact_matrices": "italy_contact_matrices.xls",
                "age_distribution": "italy_age_distribution.xls"
            },
            "kenya": {
                "model_parameters": "kenya_model_parameters.json",
                "contact_matrices": "kenya_contact_matrices.xls",
                "age_distribution": "kenya_age_distribution.xls"
            },
            "british_columbia": {
                "model_parameters": "British_Columbia_model_parameters.json",
                "contact_matrices": "British_Columbia_contact_matrices.xls",
                "age_distribution": "British_Columbia_age_distribution.xls"
            }

        }

        if self.model not in model_data_files:
            raise ValueError(f"model '{model}' is not supported.")

        # Set file paths for the selected country
        data_files = model_data_files[self.model]
        self._model_parameters_data_file = os.path.join(PROJECT_PATH, "../data",
                                                        data_files["model_parameters"])
        self._contact_data_file = os.path.join(PROJECT_PATH, "../data",
                                               data_files["contact_matrices"])
        self._age_data_file = os.path.join(PROJECT_PATH, "../data",
                                           data_files["age_distribution"])

        # Load the data
        self._get_age_data()
        self._get_model_parameters_data()
        self._get_contact_mtx()

    def _get_age_data(self):
        wb = xlrd.open_workbook(self._age_data_file)
        sheet = wb.sheet_by_index(0)
        datalist = [[sheet.cell_value(row, col) for col in range(sheet.ncols)] for
                    row in range(sheet.nrows)]
        wb.unload_sheet(0)

        # Convert to torch tensor and move to the specified device
        self.age_data = torch.tensor(datalist, dtype=torch.float32).view(-1)

    def _get_model_parameters_data(self):
        # Load model parameters from JSON
        with open(self._model_parameters_data_file) as f:
            parameters = json.load(f)

        # Parse parameters, convert lists to tensors where necessary
        self.model_parameters_data = {
            param: torch.tensor(param_data["value"],
                                dtype=torch.float32)
            if isinstance(param_data["value"], list)
            else torch.tensor([param_data["value"]],
                              dtype=torch.float32).item()
            for param, param_data in parameters.items()
        }

    def _get_contact_mtx(self):
        wb = xlrd.open_workbook(self._contact_data_file)
        contact_matrices = {}
        num_sheets = 1 if self.model == "british_columbia" else (
            2 if self.model in ["moghadas", "seir", "italy"] else 4)

        for idx in range(num_sheets):
            sheet = wb.sheet_by_index(idx)
            datalist = [[sheet.cell_value(row, col) for col in range(sheet.ncols)] for
                        row in range(sheet.nrows)]
            cm_type = wb.sheet_names()[idx]
            wb.unload_sheet(idx)

            # Convert to torch tensor
            matrix = torch.tensor(datalist, dtype=torch.float32)
            contact_matrices[cm_type] = matrix

        self.contact_data = contact_matrices
