# Variables
VENV_DIR = venv

# Creation of the virtual environment and installing dependencies
install:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating the virtual environment..."; \
		pip install --upgrade pip; \
		python3 -m venv $(VENV_DIR); \
	else \
		echo "Virtual environment already exists."; \
	fi
	@echo "Activation of the virtual environment..."
	@. $(VENV_DIR)/Scripts/activate && pip install -r requirements.txtj