from setuptools import find_packages, setup  # Package discovery and setup tools
from typing import List  # Type hints support

# Constant for editable installation marker
HYPHEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    """
    Read and parse requirements from a text file.

    Args:
        file_path (str): Path to requirements.txt file

    Returns:
        List[str]: Clean list of requirements excluding empty lines and -e .
    """
    requirements = []
    with open(file_path) as file_obj:
        # Read all lines and remove whitespace/newline characters
        requirements = [req.strip() for req in file_obj.readlines()]

        # Remove editable installation marker if present
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


# Package configuration
setup(
    name="performance_prediction",  # Package name ( PEP 8 recommends lowercase with underscores )
    version='0.0.1',  # Initial version (semantic versioning recommended)
    author='Ved',  # Author name
    author_email='vihast@gmail.com',  # not real gmail
    packages=find_packages(),  # Automatically discover Python packages in directory
    install_requires=get_requirements('requirements.txt'),  # Runtime dependencies
)

