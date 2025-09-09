# Docs for the Azure Web Apps Deploy action: https://github.com/Azure/webapps-deploy
# More GitHub Actions for Azure: https://github.com/Azure/actions
# More info on Python, GitHub Actions, and Azure App Service: https://aka.ms/python-webapps-actions

name: Build and deploy Python app to Azure Web App - activate-plus-sow-analyzer

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read #This is required for actions/checkout

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python version
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      # Create runtime.txt for Oryx (specifies Python version)
      - name: Create runtime.txt
        run: echo "python-3.10" > runtime.txt

      - name: Install dependencies (for build validation/tests if needed)
        run: pip install -r requirements.txt

      # Zip the code (exclude venv, .git, etc.)
      - name: Zip artifact
        run: |
          zip -r app.zip . -x "venv/*" ".git/*" "__pycache__/*" "*.pyc" "app.zip"

      - name: Upload artifact for deployment jobs
        uses: actions/upload-artifact@v4
        with:
          name: python-app
          path: app.zip

  deploy:
    runs-on: ubuntu-latest
    needs: build
    
    steps:
      - name: Download artifact from build job
        uses: actions/download-artifact@v4
        with:
          name: python-app
          path: .

      - name: Unzip for deploy
        run: unzip app.zip

      - name: 'Deploy to Azure Web App'
        uses: azure/webapps-deploy@v3
        id: deploy-to-webapp
        with:
          app-name: 'activate-plus-sow-analyzer'
          publish-profile: ${{ secrets.AZUREAPPSERVICE_PUBLISHPROFILE_2DA3C67FD5474884AF503EE734DB880C }}
          package: './app.zip'  # Explicit zip for reliability
