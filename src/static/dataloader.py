import json
import os
import torch
import xlrd
from gdown import download

# Google Drive file links for various models
model_ids = {
    "rost": {
        "model_parameters": "1wzwVGzZPpfQbOy3fJGEHgWivfwYqBAjt",
        "contact_matrices": "1g1CudF3RdJdEcz1T_MnSYJdRBpFIeSAz",
        "age_distribution": "1M64F15FTN44dpjhglPY-bcNSARqpbnmT"
    },
    "chikina": {
        "model_parameters": "1XDf5_Jd6Z1ao0Y6Lxeidqmha12ixCDxK",
        "contact_matrices": "1SvSxAhbYamx_wK4ug8lK7QZsu_8TX2Dg",
        "age_distribution": "1ULrvitaTHS73RtZQx5_I781av040yFdS"
    },
    "moghadas": {
        "model_parameters": "1Ae_TLyIHsRIKEZYLuMePZshftrgYvI2T",
        "contact_matrices": "1pLfqqUn1FGo4MBN8gCXDYrxQXHCGBfXl",
        "age_distribution": "1bgvP9W08ZOFyjz1FN9Mj3e3G-4YzXJyZ"
    },
    "seir": {
        "model_parameters": "1296mBAN2fy_R8kuXQNvvc7HEv-dSuC1S",
        "contact_matrices": "1ryvN37HHCxNidFLbxTtW_4wFU_3owDS5",
        "age_distribution": "1UPtjEUJPax1f6Pr1np2Y2aLz0SG2nw8n"
    },
    "italy": {
        "model_parameters": "1a-kb9EpSSl0dyH6TZRXsXtomRIM1BEHH",
        "contact_matrices": "1er8pEI1Q2gnG07PJYRxkz8PHoGC0_1_W",
        "age_distribution": "1hvfOY5gPvvtBbfQuSmPr3FDyRnMR3LVN"
    },
    "kenya": {
        "model_parameters": "1tY03EqrmcxI6iWypWZQpYyrHSxkBPJ6x",
        "contact_matrices": "1E7ebxcj-V2aj6PPvGiEPVi_cgw1Ot2Y0",
        "age_distribution": "11-jdcccXeHC3nfuJQ1vTL-b8fs2vKgcC"
    },
    "british_columbia": {
        "model_parameters": "1NRnLBlzswN6I0xiRA97jA3yPDp04nFae",
        "contact_matrices": "1_ju10AS0VPRDcUpmMMUYr30rkjtA7kRl",
        "age_distribution": "11bsQqIywcEIV34zl5N1FKSRqxQ108dXC"
    }
}

labels_file_id = "1Qe7poWcPwol0xfVIYP23Xk9SUnzHZpwI"  # Shared labels file ID

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class DataLoader:
    def __init__(self, model: str):
        self.model = model.lower()
        self._validate_model()

        # Set file paths and create data directory
        self._set_file_paths()
        self._create_data_directory()

        # Download the files from Google Drive
        self._download_data()

        # Load the data
        self.age_data = self._get_age_data()
        self.model_parameters_data = self._get_model_parameters_data()
        self.contact_data = self._get_contact_mtx()
        self.labels = self._get_labels()

    def _validate_model(self):
        """Validate if the model is supported."""
        if self.model not in model_ids:
            raise ValueError(f"Model '{self.model}' is not supported.")

    def _set_file_paths(self):
        """Set file paths for the selected model."""
        data_files = {
            "model_parameters": f"{self.model}_model_parameters.json",
            "contact_matrices": f"{self.model}_contact_matrices.xls",
            "age_distribution": f"{self.model}_age_distribution.xls",
            "labels": "labels.json"
        }

        self._model_parameters_data_file = os.path.join(PROJECT_PATH, "../data",
                                                        data_files["model_parameters"])
        self._contact_data_file = os.path.join(PROJECT_PATH, "../data",
                                               data_files["contact_matrices"])
        self._age_data_file = os.path.join(PROJECT_PATH, "../data",
                                           data_files["age_distribution"])
        self._labels_file = os.path.join(PROJECT_PATH, "../data",
                                         data_files["labels"])

    def _create_data_directory(self):
        """Create the data directory if it doesn't exist."""
        data_directory = os.path.join(PROJECT_PATH, "../data")
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)

    def _download_data(self):
        """Download each file from Google Drive if not already present."""
        for file_type, file_info in model_ids[self.model].items():
            filename = self._get_filename(file_type)

            # Check if the file already exists
            if not os.path.isfile(filename):
                print(f"Downloading {file_type} for model {self.model}...")
                self._download_file(file_type, file_info, filename)
            else:
                print(f"{file_type} for model {self.model} already "
                      f"exists. Skipping download.")

        # Download the shared labels file
        if not os.path.isfile(self._labels_file):
            print(f"Downloading labels for model {self.model}...")
            download(f"https://drive.google.com/uc?id={labels_file_id}",
                     self._labels_file)
        else:
            print(f"Labels for model {self.model} already "
                  f"exists. Skipping download.")

    def _get_filename(self, file_type):
        """Get the filename based on the file type."""
        if file_type == "labels":
            return self._labels_file
        return os.path.join(PROJECT_PATH,
                            "../data", f"{self.model}_{file_type}.json" if
                            file_type == "model_parameters" else
                            f"{self.model}_{file_type}.xls")

    def _download_file(self, file_type, file_id, filename):
        """Download a file from Google Drive."""
        if file_type == "age_distribution":
            download(f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xls",
                     filename)
        else:
            download(f"https://drive.google.com/uc?id={file_id}", filename)

    def _get_age_data(self):
        wb = xlrd.open_workbook(self._age_data_file)
        sheet = wb.sheet_by_index(0)
        datalist = [sheet.cell_value(row, 0) for row in range(sheet.nrows)]
        wb.unload_sheet(0)

        return torch.tensor(datalist)

    def _get_model_parameters_data(self):
        """Load model parameters from JSON."""
        with open(self._model_parameters_data_file) as f:
            parameters = json.load(f)

        return {
            param: torch.tensor(param_data["value"], dtype=torch.float32)
            if isinstance(param_data["value"], list)
            else torch.tensor([param_data["value"]], dtype=torch.float32).item()
            for param, param_data in parameters.items()
        }

    def _get_contact_mtx(self):
        """Load contact matrices from the specified file."""
        wb = xlrd.open_workbook(self._contact_data_file)
        contact_matrices = {}
        num_sheets = 1 if self.model == "british_columbia" else (
            2 if self.model in ["moghadas", "seir", "italy"] else 4)

        for idx in range(num_sheets):
            sheet = wb.sheet_by_index(idx)
            datalist = [[sheet.cell_value(row, col) for
                         col in range(sheet.ncols)] for row in range(sheet.nrows)]
            cm_type = wb.sheet_names()[idx]
            wb.unload_sheet(idx)
            matrix = torch.tensor(datalist, dtype=torch.float32)
            contact_matrices[cm_type] = matrix

        return contact_matrices

    def _get_labels(self):
        """Load labels from the labels.json file based on the model."""
        with open(self._labels_file, 'r') as f:
            labels_data = json.load(f)

        if self.model not in labels_data:
            raise ValueError(f"Model '{self.model}' not found in labels.json")

        return labels_data[self.model]
