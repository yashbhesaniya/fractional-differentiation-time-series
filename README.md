# ML-in-Finance

## About
This repository is created as part of the MAP 5922 MAP 4006 course at IME-USP. Its primary objective is to establish a complete pipeline for machine learning applications in finance, specifically focusing on the development and implementation of trading strategies.

The pipeline encompasses various stages, including data preparation, feature engineering, model training and evaluation, and strategy execution. Our aim is to provide a comprehensive framework that covers the entire process of developing and testing machine learning-based trading strategies.

This README serves as a reference for contributors to the shared GitHub repository used for the course. When adding your code, please consult this README for guidance and feel free to make suggestions to further improve the repository.

## Style guide
**All contributions should be fully in English.** This includes both code, such as variable names, and comments. Naming should follow the Python style guide. Variables should also be given descriptive names, even if they're longer. For example, it is better to name a variable 'bovespa_data_loader' than 'bovdl'. It reduces the cognitive strain of reading a fellow contributor's code.

Notable naming conventions include:

- Class names as CamelCase (e.g., StockDataProcessor, PortfolioManager)
- Functions as lower_case_with_underscores (e.g., calculate_returns, fetch_stock_prices)
- Variables as lower_case_with_underscores (e.g., closing_price, stock_symbol)
- Class methods as lower_case_with_underscores (e.g., compute_portfolio_returns, validate_input_data)
- Constants as UPPERCASE (e.g., MAX_RETRIES, DEFAULT_BATCH_SIZE)
- Inner use objects have their name prefixed with an underscore (e.g., _data_processor, _helper_function)

## Repository Organization
Please observe the project structure. The modules **are exclusively .py scripts** and each directory contains an __init__.py file that imports the functions/classes used in that directory.

All code must be placed in the /src/finance_ml directory as follows:

    └── src/finance_ml
        ├── data_preparation
        │   ├── __init__.py
        │   └── data_preparation.py
        ├── fractional_differentation
        │   ├── __init__.py
        │   ├── fractional_differentiation.py
        │   └── some_other_module.py
        └── ...
        
## Jupyter notebooks (.ipynb)
While Jupyter Notebooks (.ipynb) are useful for prototyping, data visualization, and writing usage examples, they are not appropriate for building code pipelines. Therefore, please ensure that you do not add .ipynb files to the \src folder.

However, if you believe it's a good idea to share a use case to illustrate the usage of your implemented module, you can utilize the \notebooks folder. Just make sure to name the file according to your module.

    └── notebooks
        ├── data_preparation.ipynb
        ├── fractional_differentiation.ipynb
        └──  some_other_module.ipynb

## Specifying function/method inputs and outputs
For all functions, it is necessary to specify:

- A brief description of the function, encompassing the primary goal and highlights
- The variable types of the arguments and a short description of each variable
- The output type and a short description


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
            Short description of the output1.
        ...
        '''
        
## About using Classes
Using classes isn't mandatory, but it's recommended whenever it meets certain criteria, such as:

- The module performs data transformation and relies on hyperparameters that are used across multiple functions.
- The module relies on multiple functions and follows a logical flow with multiple steps.
- When it's possible to utilize classical methods like fit, transform, or predict

   
## Unit testing
There will be a folder in the repository called 'tests'. Whenever you make a contribution, add a file (or edit an existing one) with functions designed for testing the code you just built. The file should follow the standards of the pytest library, which means it should be named 'test_.py' ( being the rest of the file name). The functions to be tested should also be named 'test_' and should include one or more assert statements to verify that your code works as intended. Please avoid using 'test_.py' or '*_test.py' names in other files.

Unit testing is crucial to ensure that version differences leading to errors are detected. They also help in scenarios where function B depends on function A, as any intended changes in function A should not silently affect the behavior of function B. While bug fixes and other changes are inevitable, we want to ensure that any downstream effects are detected. Before submitting a pull request, run all tests, including both the ones you added and those already present in the codebase. If any test fails, make appropriate edits to your code.


## Using virtual environment and updating requirements.txt
We recommend that each user create a virtual environment. This can be easily done by running the following command in a terminal:

    python -m venv financeml_env

To activate the environment, use the following command:

    .\financeml_env\Scripts\activate.bat

After activating the environment, install all package dependencies by running the following command:
    
    pip install -r requirements.txt

**If you have used a package that is not already listed in requirements.txt, please add the package along with its corresponding distribution version.** However, please try to avoid modifying the packages that are already specified.

## Git
Some general rules:
- **Do not push directly to the main branch.**
- Commit messages should be short but precise.
- Pulling changes from the remote repository before creating a new branch to ensure you have the latest updates: git pull origin main.
- Create a branch using the name of the branch you're working with
- Regularly updating your local main branch with the latest changes from the remote main branch: git pull origin main (check that before  start updating/creating your code).
- Using descriptive branch names that reflect the purpose or feature being worked on for better clarity and organization.
- Reviewing and testing your code locally before pushing changes to the remote repository.
- If you don't have Git installed on your computer, you must install it in order to utilize the available resources. Additionally, we recommend using Visual Studio Code (VSC).

Important commands:

        0. git clone https://github.com/Christian-Jaekel/ML-in-Finance.git (clone the project)
        1. git pull origin main (pull the latest changes from the remote main branch to your local main branch -  to make sure you have the latest version of main)
        2. git checkout -b branch_name (create and switch to a branch named 'branch_name').
        3. git checkout branch_name (switch to an existing branch named 'branch_name').
        4. git add . (add all files from the working directory to the staging area).
        5. git commit -m "short description" (commit files from the staging area to the local repository).
        6. git push origin branch_name (send changes from the local repository to the remote repository (GitHub)).
        7. **Do not push to the main branch**

For example, if you're writing the code for the module "fractional_differentiation," your 'branch_name' should be 'fractional_differentiation' or an abbreviation like 'frac_diff'. After your update, we'll merge it into the main branch. This way, the next group that presents will have access to all the available modules that were previously presented by other groups





