import requests
import os
import re
import gzip
import shutil

class Import_nii():
    def __init__(self, dataset_id, version, save_path):
        """
          @Description
            The Import_nii class provides a way to download NIfTI files from the OpenNeuro dataset.
            It fetches the required dataset based on the dataset ID and version, filters the relevant files,
            and downloads them to the specified local path. The downloaded files are uncompressed and saved
            as .nii files.


          @Inputs
            - dataset_id: The dataset ID. This is the name of the folder where the file is saved as well
            - version: The version of the dataset.
            - save_path: The local path to save the downloaded files. Parent folder for dataset_id
            - group: Dataset group
            - save_path_tensor: Final path where to store the .pt
            - bound: Range where I am going to slice the .nii
        """

        self.dataset_id = dataset_id
        self.version = version
        self.save_path = save_path
        self.api_url = 'https://openneuro.org/crn/graphql'


        # Define the GraphQL query to get the dataset snapshot files
        query = f'''
    {{
      snapshot(datasetId: "{self.dataset_id}", tag: "{self.version}") {{
        id
        tag
        files {{
          id
          filename
          size
          urls
        }}
      }}
    }}
    '''

        # Get the URL info from the query
        response = requests.post(self.api_url, json={'query': query})
        response_data = response.json()

        # Get for each subject the URL of the directory "anat"
        subjects = {i['filename']: i['id'] for i in response_data['data']['snapshot']['files'] if
                    "sub" in i['filename']}
        Sub_anat = {k: self.RecQuery(v) for k, v in subjects.items()}

        # Save the files  in a dictionary such as  'sub-001_T2w.nii': URL
        toDownload = {}
        for k, v in Sub_anat.items():
            toDownload.update(self.GetURL(v))

        # Download the files
        for k, v in toDownload.items():
            self.Download(k, v)

    def RecQuery(self, id):
        """
        @Description
          Perform a recursive GraphQL query for the given file ID to find the corresponding 'anat' file.

        @Inputs
          - id (str): The file ID to perform the recursive query on.

        @output
            - file_id (str): The file ID of the corresponding 'anat' file.
        """

        query = f'''
    {{
      snapshot(datasetId: "{self.dataset_id}", tag: "{self.version}") {{
        files(tree: "{id}") {{
          id
          filename
          size
          urls
        }}
      }}
    }}
    '''

        response = requests.post(self.api_url, json={'query': query})
        response_data = response.json()
        response_data = response_data['data']['snapshot']['files']
        file_id = None

        for i in response_data:
            if 'anat' in i['filename']:
                file_id = i['id']

        return file_id

    def extract_subject_modality(self, url):
        """
        @Description
          Extract subject number and modality from the given URL.

        @Inputs
          - url (str): The URL to extract the subject number and modality from.

        @output
          - result (str): The formatted string containing subject number and modality, e.g., 'sub-001_T2w.nii'.
        """

        # Extract subject number and modality from the URL
        subject_match = re.search(r'sub-(\d+)', url)
        modality_match = re.search(r'sub-\d+_([a-zA-Z0-9]+)', url)
        result = None

        if subject_match and modality_match:
            subject_number = int(subject_match.group(1))
            subject = f'sub-{subject_number:03d}'
            modality = modality_match.group(1)
            result = f'{subject}_{modality}.nii'

        return result

    def GetURL(self, ant_id):
        """
        @Description
          Perform a GraphQL query for the given 'anat' file ID to find the corresponding NIfTI file URLs.

        @Inputs
          - ant_id (str): The 'anat' file ID.

        @output
          - url_dict (dict): A dictionary containing the formatted string of subject and modality as keys,
                            and corresponding NIfTI file URLs as values.
        """

        query = f'''
      {{
        snapshot(datasetId: "{dataset_id}", tag: "{self.version}") {{
          files(tree: "{ant_id}") {{
            id
            filename
            size
            urls
          }}
        }}
      }}
      '''

        response = requests.post(self.api_url, json={'query': query})
        response_data = response.json()
        response_data = response_data['data']['snapshot']['files']

        return {self.extract_subject_modality(i["urls"][0]): i["urls"][0] for i in response_data if
                ".nii.gz" in i["urls"][0]}

    def Download(self, filename, url):
        """
        @Description
          Download the NIfTI file from the given URL, uncompress it, and save it in the specified path.

        @Inputs
          - filename (str): The filename of the NIfTI file to be saved.
          - url (str): The URL to download the NIfTI file from.

        @output
          - Downloads the NIfTI file, uncompresses it, and saves it in the specified path.
          - Removes the compressed .gz file after uncompressing it.
        """

        # Create a directory for the downloaded files
        download_directory = os.path.join(self.save_path, f'{self.dataset_id}_files')
        os.makedirs(download_directory, exist_ok=True)

        local_file_path_nii = os.path.join(download_directory, filename)
        local_file_path = local_file_path_nii + '.gz'

        response = requests.get(url)

        with open(local_file_path, 'wb') as f:
            f.write(response.content)
        print(f'Downloaded {filename} to {local_file_path}')

        # Uncompress the .gz file and save it as a .nii file
        with gzip.open(local_file_path, 'rb') as f_in:
            with open(local_file_path_nii, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f'Uncompressed {filename} to {local_file_path_nii}')

        # Remove the .gz file
        os.remove(local_file_path)
        print(f'Removed {local_file_path}')
