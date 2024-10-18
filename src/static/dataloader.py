import json
import os
import torch
import xlrd
from gdown import download

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class DataLoader:
    def __init__(self, model: str):
        # Ensure the 'data' directory exists
        self._create_data_directory()

        # Load the configuration JSON from Google Drive and set up necessary paths
        self.model = model.lower()
        self.model_data = self._load_model_config()
        self.models = self.model_data["models"]
        self.labels_file_id = self.model_data["shared_files"]["labels_file_id"]

        # Validate model
        self._validate_model()

        # Set file paths
        self._set_file_paths()

        # Download the necessary files
        self._download_data()

        # Load the data into memory
        self.age_data = self._get_age_data()
        self.model_parameters_data = self._get_model_parameters_data()
        self.contact_data = self._get_contact_mtx()
        self.labels = self._get_labels()

    def _create_data_directory(self):
        """Create the data directory if it doesn't exist."""
        data_directory = os.path.join(PROJECT_PATH, "../data")
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)

    def _load_model_config(self):
        """Download and load the model configuration file from Google Drive."""
        model_config_url = "https://drive.google.com/uc?id=18ztwRVy4qW2NMs8OKUbkEDS_1rAxpRt0"
        config_file_path = os.path.join(PROJECT_PATH, "../data", "model_config.json")

        # Download the model configuration if it doesn't exist
        if not os.path.isfile(config_file_path):
            print("Downloading model configuration from Google Drive...")
            download(model_config_url, config_file_path)
        else:
            print("Model configuration already exists. Skipping download.")

        # Load the configuration file into a dictionary
        with open(config_file_path, 'r') as f:
            return json.load(f)

    def _validate_model(self):
        """Validate if the model is supported."""
        if self.model not in self.models:
            raise ValueError(f"Model '{self.model}' is not supported.")

    def _set_file_paths(self):
        """Set file paths for the selected model."""
        data_files = {
            "model_parameters": f"{self.model}_model_parameters.json",
            "contact_matrices": f"{self.model}_contact_matrices.xls",
            "age_distribution": f"{self.model}_age_distribution.xls",
            "labels": "labels.json"
        }

        self._model_parameters_data_file = os.path.join(
            PROJECT_PATH, "../data", data_files["model_parameters"])
        self._contact_data_file = os.path.join(
            PROJECT_PATH, "../data", data_files["contact_matrices"])
        self._age_data_file = os.path.join(
            PROJECT_PATH, "../data", data_files["age_distribution"])
        self._labels_file = os.path.join(
            PROJECT_PATH, "../data", data_files["labels"])

    def _download_data(self):
        """Download each file from Google Drive if not already present."""
        # Use model information from the new JSON file
        for file_type, file_info in self.models[self.model].items():
            filename = self._get_filename(file_type)

            # Check if the file already exists
            if not os.path.isfile(filename):
                print(f"Downloading {file_type} for model {self.model}...")
                self._download_file(file_type, file_info, filename)
            else:
                print(f"{file_type} for model {self.model} "
                      f"already exists. Skipping download.")

        # Download the shared labels file
        if not os.path.isfile(self._labels_file):
            print(f"Downloading labels for model {self.model}...")
            download(f"https://drive.google.com/uc?id={self.labels_file_id}",
                     self._labels_file)
        else:
            print(f"Labels for model {self.model} already exists. "
                  f"Skipping download.")

    def _get_filename(self, file_type):
        """Get the filename based on the file type."""
        if file_type == "labels":
            return self._labels_file
        return os.path.join(PROJECT_PATH, "../data",
                            f"{self.model}_{file_type}.json" if
                            file_type == "model_parameters" else f"{self.model}_{file_type}.xls")

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
            datalist = [[sheet.cell_value(row, col) for col in
                         range(sheet.ncols)] for row in range(sheet.nrows)]
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
