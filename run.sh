#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
VENV_DIR="venv"                 # Name of the virtual environment directory
REQUIREMENTS_FILE="requirement.txt"
APP_FILE="app.py"
PYTHON_CMD="python3"            # Use python3 explicitly, change if needed
VARS_FILE="vars.py"             # Python file for configuration variables
EXAMPLE_VARS_FILE="vars.py.example" # Example configuration file (committed to Git)

# --- Helper Functions ---
print_info() {
    echo "INFO: $1"
}

print_error() {
    echo "ERROR: $1" >&2
    exit 1
}

# --- Sanity Checks ---

# Check for Python 3 command
if ! command -v $PYTHON_CMD &> /dev/null; then
    print_error "$PYTHON_CMD could not be found. Please install Python 3."
fi
print_info "Using Python interpreter: $(command -v $PYTHON_CMD)"
$PYTHON_CMD --version

# Check for requirements file
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    print_error "$REQUIREMENTS_FILE not found in the current directory ($(pwd))."
fi

# Check for app file
if [ ! -f "$APP_FILE" ]; then
    print_error "$APP_FILE not found in the current directory ($(pwd))."
fi

# Check for vars.py file (IMPORTANT for API keys/config)
if [ ! -f "$VARS_FILE" ]; then
    print_info "----------------Okay, here is the final `run.sh` script tailored for your project, assuming you are using `vars.py` for configuration and secrets (and that you have a `vars.py.example` file in your repository).

**Remember:**
*   Ensure `vars.py` is listed in your `.gitignore` file.
*   Ensure your Python code (`app.py`, etc.) imports variables directly from `vars.py`.
*   Users cloning the repo will need to run `chmod +x run.sh` after cloning.

```bash
#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
VENV_DIR="venv"                     # Name of the virtual environment directory
REQUIREMENTS_FILE="requirements.txt"
APP_FILE="app.py"                   # Your main Streamlit application file
PYTHON_CMD="python3"                # Command to run Python 3
VARS_FILE="vars.py"                 # Your Python configuration file
EXAMPLE_VARS_FILE="vars.py.example" # Example configuration file to be committed

# --- Helper Functions ---
print_info() {
    echo "INFO: $1"
}

print_error() {
    echo "ERROR: $1" >&2
    exit 1
}

# --- Sanity Checks ---

# Check for Python 3 command
if ! command -v $PYTHON_CMD &> /dev/null; then
    print_error "$PYTHON_CMD could not be found. Please install Python 3."
fi
print_info "Using Python interpreter: $(command -v $PYTHON_CMD)"
$PYTHON_CMD --version

# Check for requirements file
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    print_error "'$REQUIREMENTS_FILE' not found in the current directory ($(pwd))."
fi

# Check for app file
if [ ! -f "$APP_FILE" ]; then
    print_error "'$APP_FILE' not found in the current directory ($(pwd))."
fi

# Check for vars.py file (IMPORTANT for API keys/config)
if [ ! -f "$VARS_FILE" ]; then
    print_info "---------------------------------------------------------------------"
    print_info "WARNING: Configuration file '$VARS_FILE' not found."-----------------------------------------------------"
    print_info "WARNING: Configuration file '$VARS_FILE' not found."
    print_info "Your application requires this file to load API keys and settings."
    if [ -f "$EXAMPLE_VARS_FILE" ]; then
       print_info "Please create '$VARS_FILE' by copying '$EXAMPLE_VARS_FILE':"
       print_info "  cp '$EXAMPLE_VARS_FILE' '$VARS_FILE'" # Added quotes for paths with spaces
       print_info "Then, edit '$VARS_FILE' to add your actual API keys and configuration."
    else
       print_info "Please create a '$VARS_FILE' file in the project root directory."
       print_info "Define your API keys and configuration as Python variables inside it."
       print_info "(Refer to README.md or project documentation for the required variables)."
    fi
    print_info "IMPORTANT: Ensure '$VARS_FILE' is added to your .gitignore file!"
    print_info "---------------------------------------------------------------------"
    # You might want to uncomment the next line if the app cannot run without vars.py
    # print_error "Exiting due to missing $VARS_FILE file."
fi


# --- Virtual Environment Setup ---
if [ ! -d "$VENV_DIR" ]; then
    print_info "Creating virtual environment in '$VENV_DIR'..."
    $PYTHON_CMD -m venv "$
    print_info "Your application requires this file to load API keys and settings."
    if [ -f "$EXAMPLE_VARS_FILE" ]; then
       print_info "Please create '$VARS_FILE' by copying '$EXAMPLE_VARS_FILE':"
       print_info "  cp '$EXAMPLE_VARS_FILE' '$VARS_FILE'"
       print_info "Then, edit '$VARS_FILE' to add your actual API keys and configuration."
    else
       print_info "Please create a '$VARS_FILE' file in the project root with your"
       print_info "API keys and configuration variables defined as Python variables."
       print_info "(See README.md or project documentation for required variables)."
       print_info "**IMPORTANT: Make sure '$VARS_FILE' is in your .gitignore and not committed!**"
    fi
    print_info "------------------------------------------------VENV_DIR"
    print_info "Virtual environment created."
---------------------"
    # You might want to exit if the vars file is absolutely essential before any setup
    # print_error "Exiting due to missing '$VARS_FILE'."
fi


# --- Virtual Environment Setup ---
if [ ! -d "$VENV_DIR" ]; then
    print_info "Creating virtual environment in '$Velse
    print_info "Virtual environment '$VENV_DIR' alreadyENV_DIR'..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    print_info "Virtual environment created."
 exists."
fi

# Activate virtual environment
# This assumes bash/zsh.else
    print_info "Virtual environment '$VENV_DIR' already Use `source "$VENV_DIR/bin/activate.fish"` for fish exists."
fi

# Activate virtual environment
# This activation command works for bash/zsh. shell.
print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# --- Dependency Installation ---
print_info "Installing/updating dependencies from $REQUIRE May need adjustment for other shells (e.g., fish).
print_info "Activating virtual environment..."MENTS_FILE..."
# Upgrade pip first within the venv
pip install --upgrade
source "$VENV_DIR/bin/activate"
print_info "Virtual pip
# Install requirements
pip install -r "$REQUIREMENTS_FILE"
print_info environment activated (using Python: $(command -v python))"

# --- Dependency Installation ---
print_info "Installing/updating dependencies from '$REQUIREMENTS_FILE'..."
pip "Dependencies installed."

# --- Run Application ---
print_info "Starting install --upgrade pip # Upgrade pip within the venv
pip install -r "$REQUIRE Streamlit application ($APP_FILE)..."
# Ensure Streamlit uses the variablesMENTS_FILE"
print_info "Dependencies installed."

# --- Run imported within app.py from vars.py
streamlit run "$APP_FILE"

print_info " Application ---
print_info "Starting Streamlit application ('$APP_FILE')..."
printStreamlit application stopped."

# Deactivation typically happens automatically when the script exits._info "Access the application via the URL printed below (usually http://localhost:8501)"
# If needed explicitly:
# deactivate
# print_info "Virtual
print_info "Press Ctrl+C in this terminal to stop the application."
streamlit run "$APP_FILE"

# --- Script End ---
print_info "Streamlit application stopped."

# Deactivation is typically automatic when the script exits or the shell session environment deactivated."

exit 0
```

**Remember:**

1.  Save this code as `run.sh` in your project's root directory.
2 ends.
# If needed manually:
# deactivate
# print_info "Virtual environment deactivated."

exit 0
