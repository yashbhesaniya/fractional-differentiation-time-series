# ML-in-Finance

## About
This repository is in the context of the course MAP 5922 MAP 4006 at IME-USP. The main goal is to structure a complete pipeline of machine learning applications with finance and related techniques, focusing on the development and implementation of trading strategies.

The pipeline encompasses various stages, including data preparation, feature engineering, model training and evaluation, and strategy execution. By following this pipeline, we aim to provide a comprehensive framework for developing and testing machine learning-based trading strategies.

## Repository Organization
Please observe the project structure. The modules are exclusively .py scripts and each directory contains an init.py file that imports the functions/classes used in that directory.

All code must be placed in the /src directory as follows:

    └── src
        ├── data_preparation
        │   ├── __init__.py
        │   └── data_preparation.py
        ├── fractional_differentation
        │   ├── __init__.py
        │   ├── fractional_differentation.py
        │   └── some_other_module.py
        └── ...


## Specifying function/method inputs and outputs
For all functions, it is necessary to specify:

- The variable types of the arguments
- The output type
- Function description, args, and returns

        '''
        Short description

        Args:
        -----
        arg1 : data type
            Short description of the variable arg1.
        arg2 : data type
            Short description of the variable arg2.
        ...

        Returns:
        -----
        output1: data type
            Short description of the output.
        ...
        '''
## Testing
There is a folder named tests where tests must be conducted to guarantee the consistency and reliability of each module. Please use the nomenclature test_module.py, as shown in the example.

## Requirements
We recommend that each user should create a virtual environment. This can be easily done by running the following command in a terminal:

    python -m venv financeml_env

To activate the environment:

    .\financeml_env\Scripts\activate.bat

After that, the user can install all package dependencies by running the command:
pip install -r requirements.txt

If you used a package that is not already mentioned in requirements.txt, please add the package and the correct distribution version. Do not change the already specified packages.

## Git
Some general rules:
- Always create branches
- Do not push directly to the main branch
- Commit messages should be short but precise

If you don't have git installed on your computer, you must install it in order to use the resources. Another recommendation is to use Visual Studio Code (VSC).
Important commands:

        1) git checkout -b branch_name (create and switch to a branch named branch_name)
        2) git add . (add all files from the working directory to the staging area)
        3) git commit -m "short description" (commit files from the staging area to the local repository)
        4) git push origin branch_name (send changes from the local repository to the remote repository (GitHub))

**Please note that all information in this repository must be provided in English.**




