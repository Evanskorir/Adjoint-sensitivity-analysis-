import json
import os

import torch
import xlrd

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class DataLoader:
    def __init__(self):
        self._model_parameters_data_file = os.path.join(PROJECT_PATH, "../data",
                                                        "model_parameters.json")
        self._contact_data_file = os.path.join(PROJECT_PATH, "../data",
                                               "contact_matrices.xls")
        self._age_data_file = os.path.join(PROJECT_PATH, "../data",
                                           "age_distribution.xls")

        self._get_age_data()
        self._get_model_parameters_data()
        self._get_contact_mtx()

    def _get_age_data(self):
        wb = xlrd.open_workbook(self._age_data_file)
        sheet = wb.sheet_by_index(0)
        datalist = torch.tensor([sheet.row_values(i) for i in range(0, sheet.nrows)],
                                dtype=torch.float32)
        wb.unload_sheet(0)
        self.age_data = datalist

    def _get_model_parameters_data(self):
        # Load model parameters
        with open(self._model_parameters_data_file) as f:
            parameters = json.load(f)
        self.model_parameters_data = dict()
        for param in parameters.keys():
            param_value = parameters[param]["value"]
            if isinstance(param_value, list):
                self.model_parameters_data[param] = torch.tensor(param_value,
                                                                 dtype=torch.float32)
            else:
                self.model_parameters_data[param] = param_value

    def _get_contact_mtx(self):
        wb = xlrd.open_workbook(self._contact_data_file)
        contact_matrices = dict()
        for idx in range(4):
            sheet = wb.sheet_by_index(idx)
            datalist = torch.tensor([sheet.row_values(i) for i in range(0, sheet.nrows)],
                                    dtype=torch.float32)
            cm_type = wb.sheet_names()[idx]
            wb.unload_sheet(0)
            # datalist = self.transform_matrix(datalist)
            contact_matrices[cm_type] = datalist

        self.contact_data = contact_matrices
