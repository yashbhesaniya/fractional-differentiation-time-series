# ML-in-Finance

## About
This repository is created as part of the Machine Learning In Finance course. Its primary objective is to establish 
a complete pipeline for machine learning applications in finance, specifically focusing on the development and implementation of trading strategies.

The pipeline encompasses various stages, including data preparation, feature engineering, model training and evaluation, 
and strategy execution. Our aim is to provide a comprehensive framework that covers the entire process of developing 
and testing machine learning-based trading strategies.

This README serves as a reference for contributors to the shared GitHub repository used for the course. When adding your code, 
please consult this README for guidance and feel free to make suggestions to further improve the repository.


## Important

Before submitting a pull request, please ensure the following:

**1. Module Organization and Documentation:**
- Make sure that all the relevant .py modules are placed in their respective folders with proper organization.
- Provide clear and comprehensive documentation for each module, explaining its purpose, functionality, and usage.

**2. Test Implementation:**
- Implement tests for your code in the \test directory.
- Ensure that all tests pass successfully before submitting the pull request.
- Thoroughly test your code to cover different scenarios and edge cases, aiming for comprehensive test coverage.

**3. Usage of Jupyter Notebooks:**
- It is highly recommended to include a use case example in a Jupyter Notebook (.ipynb) format.
- This will help other team members understand and interact with your code more effectively, facilitating collaboration and exploration.

Additional Guidelines:

- Avoid making unnecessary changes to existing code or functions, unless they are aimed at improvements or bug fixes.
- Changing the names of references that could impact other code or tests and potentially cause crashes should be avoided.

By adhering to these guidelines, we can ensure that our code contributions are well-organized, thoroughly tested, and compatible with the existing codebase. This promotes collaboration, readability, and maintainability within the project.

For more detailed information and explanations, please refer to the following sections. These sections provide further guidance and instructions.

**Please review your code and ensure compliance with these guidelines before submitting your pull request.**

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

However, we encourage you to share a use case that demonstrates the usage of your implemented module using a Jupyter Notebook (.ipynb) file. Just make sure to name the file according to your module.

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

## File paths

A common practice is to refer to files in the same directory as the one your code is in simply by their name. This behavior presents some problems when using more complicated file structures, such as when refering to files outside of the current directory. Instead, we'll use **absolute file paths** for importing. This means you have to specify the full path from the root directory, including 'src', and it works if you try to import from any folder - from a module inside 'src', a test file inside 'tests' or an example file inside 'notebooks'.

Example:

    from src.finance_ml.data_preparation.asset_class import Asset

## About using Classes
Using classes isn't mandatory, but it's recommended whenever it meets certain criteria, such as:

- The module performs data transformation and relies on hyperparameters that are used across multiple functions.
- The module relies on multiple functions and follows a logical flow with multiple steps.
- When it's possible to utilize classical methods like fit, transform, or predict


## Unit testing
There will be a folder in the repository called 'tests'. Whenever you make a contribution, add a file (or edit an existing one) with functions designed for testing the code you just built. The file should follow the standards of the [pytest](https://docs.pytest.org/) library, which means it should be named 'test_.py' (_ being the rest of the file name). The functions to be tested should also be named 'test_'. Please avoid using 'test_.py' or '*_test.py' names in other files.

Testing functions, as specified by the `pytest` package, should include one or more statements to verify that your code works as intended. The most basic statement is

    assert <boolean condition>

Examples:

     assert object.atribute == 2
     assert isinstance(object, MyClass)

If you implement errors and warnings in your code, you can test that they're being raised with

    with pytest.raises(MyException):
        code block

    with pytest.warns(MyWarning):
        code block

Example:

    my_dict = {1:"a", 2:"b"}
    with pytest.raises(KeyError):
        my_variable = my_dict[3]

Files for unit testing should be located in the 'tests' folder:

    ├── src/finance_ml
    │   ├── your_directory
    │   └── ...
    ├── tests
    │   ├── test_your_directory.py
    │   └── ...
    └── ...

In order to run your tests, make sure to install the `pytest` package (by running `pip install pytest` in the command prompt) and do the following commands:

    - Open your terminal or command prompt and navigate to the root directory of your local repository (the ML-in-Finance folder).
    - Set the PYTHONPATH environment variable to include the src directory. In Linux, you can do this by typing PYTHONPATH=src. If you're using Windows, use the command set PYTHONPATH=src.
    - Run the command python -m pytest tests in the terminal or command prompt. If you're using Linux, you may need to replace python with python3 depending on your Python installation.

By executing these commands, you will run the pytest framework and it will discover and execute the tests located (all .py files) in the tests directory.These commands were tested in WSL2 (Windows Subsystem for Linux) terminal and in a Command Prompt in Windows. If you have trouble making the commands run, let us know.

Unit testing is crucial to ensure that version differences leading to errors are detected and the code we develop is working as intended. Ideally, they should cover all aspects of your contribution, including instantiation of classes and the behavior of all methods. They also help in scenarios where function B depends on function A, as any intended changes in function A should not silently affect the behavior of function B. While bug fixes and other changes are inevitable, we want to ensure that any downstream effects are detected. Before submitting a pull request, run all tests, including both the ones you added and those already present in the codebase. If any test fails, make appropriate edits to your code.

## Using virtual environment and updating requirements.txt
We recommend that each user create a virtual environment. This can be easily done by running the following command in a terminal:

    python -m venv financeml_env

To activate the environment, use the following command:

    .\financeml_env\Scripts\activate.bat

After activating the environment, install all package dependencies by running the following command:

    pip install -r requirements.txt

**If you have used a package that is not already listed in requirements.txt, please add the package along with its corresponding distribution version.** However, please try to avoid modifying the packages that are already specified.

## Git and Github
Git is a crucial tool for version control, enabling efficient collaboration on software development projects. It consists of a local repository, where changes are made to project files, and a remote repository, such as GitHub, for sharing and synchronizing changes. The workflow involves the working directory for file modifications, the staging area for preparing changes, and the repository for permanently storing committed changes. To track changes, use "git add" to stage modified files and "git commit" to save changes with a descriptive message. Collaborate by using "git pull" to fetch and merge remote changes and "git push" to upload local changes. A diagram illustrating the working directory, staging area, and repository context is available for reference.

![Git Diagram](https://github.com/Christian-Jaekel/ML-in-Finance/assets/54689450/6d4a3b91-17ab-44f4-8634-2ec621b94199)

Some general rules:
- **Do not push directly to the main branch.**
- Commit messages should be short but precise.
- Pull changes from the remote repository before creating a new branch to ensure you have the latest updates: git pull origin main.
- Create a branch using the name of the branch you're working with
- Regularly update your local main branch with the latest changes from the remote main branch: git pull origin main (check that before you start updating/creating your code).
- Usie descriptive branch names that reflect the purpose or feature being worked on for better clarity and organization.
- After making your changes locally, switch to main, pull the remote version of main and merge it into the local branch you worked on before pushing to the remote repository.
- Review and test (manually and with `pytest`) your code locally before pushing changes to the remote repository. Do this again after merging the updated version of main to your branch.
- If you don't have Git installed on your computer, you must install it in order to utilize the available resources. Additionally, we recommend using Visual Studio Code (VSC).

Important commands:

        0. git clone https://github.com/Christian-Jaekel/ML-in-Finance.git (clone the project)
        1. git pull origin main (pull the latest changes from the remote main branch to your local main branch -  to make sure you have the latest version of main)
        2. git checkout -b branch_name (create and switch to a branch named 'branch_name').
        3. git checkout branch_name (switch to an existing branch named 'branch_name').
        4. git add . (add all files from the working directory to the staging area).
        5. git commit -m "short description" (commit files from the staging area to the local repository).
        6. git merge branch_name (merges the changes from branch_name into the current branch, which is the branch you are currently working on)
        7. git push origin branch_name (send changes from the local repository to the remote repository (GitHub)).
        8. **Do not push to the main branch**

For example, if you're writing the code for the module "fractional_differentiation", your 'branch_name' should be 'fractional_differentiation' or an abbreviation like 'frac_diff'.

Once you've completed your contribution and assured:
- you merged an updated version of main before pushing
- your module is in .py files
- your files are properly documented
- you have a test file that asserts your code behaves as intended
- you pass all tests, including those of other test files

you may make a pull resquest (PR) from the remote branch to integrate your code into main. Add one of FabioMMaia or AFumis as reviewers, and if necessary make a short description. Once approved, your code will be available for others to use in their own contributions.



