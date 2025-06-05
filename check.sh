#!/bin/bash

# Define colors using ANSI escape codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

SOURCES=(graid/)

set -uo pipefail

# Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
if [ -d "$POETRY_ENV_PATH" ]; then
  source "$POETRY_ENV_PATH/bin/activate"
else
  printf "${RED}Error: Poetry environment not found. Please run 'poetry install' first.${NC}\n"
  exit 1
fi

# Define a function for printing a border
print_border() {
  printf "\n====================================\n\n"
}

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for required tools
required_tools=("isort" "black" "flake8" "jupyter" "git")
for tool in "${required_tools[@]}"; do
  if ! command_exists "$tool"; then
    printf "${RED}Error: %s is not installed.${NC}\n" "$tool"
    exit 1
  fi
done

# Arrays to hold tool names and their statuses
tools=()
statuses=()

printf "${GREEN}Checking import order with isort...${NC}\n"
isort --check-only "${SOURCES[@]}"
isort_status=$?
tools+=("isort")
statuses+=("$isort_status")
print_border

printf "${GREEN}Checking code formatting with black...${NC}\n"
black --check "${SOURCES[@]}"
black_status=$?
tools+=("black")
statuses+=("$black_status")
print_border

print_border
printf "${GREEN}Type checking with flake8...${NC}\n"
flake8 "${SOURCES[@]}"
flake8_status=$?
tools+=("flake8")
statuses+=("$flake8_status")
print_border

printf "${GREEN}Checking if Jupyter notebook outputs are cleared...${NC}\n"
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
git diff --exit-code notebooks/*.ipynb
notebook_status=$?
tools+=("notebooks")
statuses+=("$notebook_status")
if [[ $notebook_status -ne 0 ]]; then
    printf "${RED}Notebooks have outputs that were not cleared.${NC}\n"
    printf "${RED}Run ./clear_notebook.sh to clear. First 50 lines of diff.${NC}\n"
    git diff notebooks/*.ipynb | head -n 50
fi
print_border

printf "${GREEN}Status report:${NC}\n"
all_success=true
for i in "${!tools[@]}"; do
  tool="${tools[$i]}"
  status="${statuses[$i]}"
  if [[ $status -ne 0 ]]; then
    printf "${RED}%s failed${NC}\n" "$tool"
    all_success=false
  else
    printf "${GREEN}%s succeeded${NC}\n" "$tool"
  fi
done
print_border

if [ "$all_success" = true ]; then
  printf "${GREEN}All checks succeeded, you're good to go!${NC}\n"
  exit 0
else
  printf "${RED}There were some tool failures. Please check the errors above${NC}\n"
  printf "${RED}and re-commit once the ./check.sh script is happy.${NC}\n\n"
  printf "${RED}You can usually fix formatting errors with ./format.sh, but${NC}\n"
  printf "${RED}type errors might require some manual effort.${NC}\n"
  exit 1
fi
