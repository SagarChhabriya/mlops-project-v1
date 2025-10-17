1. Understand the Project
2. What tools are available
3. Project Structure
    - Create and Clone repo
    - Create venv
    - install libraries
    - Train the model
    - Track the experiments using wandb
    - Export the models
    - Create the app
    - Run locally
    - define requirements.txt
    - define Dockerfile
    - Build docker image
    - Run image: execute container
    - deploy the app to streamlit cloud
    
4. Next
    - Automated Training
    - 



Build docker image
    - docker build -t my-streamlit-app .
Run Docker image
    - docker run -p 8501:8501 my-streamlit-app


- Automate the model training
    1. Ifyou currently have a Jupyter notebook, convert it once to a script:
        - jupyter nbconvert --to script logistic-regression.ipynb --output src/train.py
    2. Create the GitHub Workflow File
        - Create a new file:
            - .github/workflows/train.yml
    3. Add W&B API Key as a Secret



Plan: Add CI/CD + Model Training in GitHub Actions
    - Here’s what we’ll do:
    1. On every push / on main branch:
        - Checkout code
        - Install dependencies
        - Run training script (e.g. in train.py) → produce a new model.pkl
        - Build a Docker image (that includes the newly trained model)
        - Push the Docker image to Docker Hub (or a registry)
        - Deploy that image to Cloud Run
    2. Modify your Dockerfile / app so that it uses the newly trained model file inside the container.


Part A: Update Repository Structure & Code
    - 1. Add train.py
    - 2. Modify app.py (if needed)
    - 3. Update Dockerfile
        - Modify your Dockerfile to:
            - Copy train.py, code, etc.
            - Run the training script before starting the app (or bake the model)
            - Use CMD or ENTRYPOINT for running the app

Note: This approach builds the model on every push. If training is heavy, this will slow down your build times. 

Part B: Create GitHub Actions Workflow
    - Create a file .github/workflows/deploy.yml with content like:


✨✨✨✨ Project Session 02

1. Docker Image
  - Build the docker image 
  - Push the image to docker hub
  - deploy the image to google cloud run
    - google cloud console > Cloud Run >  
2. CI
  - Convert the notebook file to .py -> train.py
  - define the train.yaml workflow
    - .github/workflows/train.yaml
    - Make sure to add the WANDB_API_KEY secret to github actions secret

3. CD

  - Step 1: Enable required Google Cloud APIs
    In Google Cloud Console:

    1. Go to APIs & Services → Library
    2. Enable the following:
      * Cloud Run Admin API
      * Artifact Registry API
      * Cloud Build API
      * IAM Service Account Credentials API
  
  - Step 2: Create Artifact Registry repository
      We’ll store Docker images here.
  
      1. Go to Artifact Registry → Repositories → Create Repository.
      2. Choose:
        * Name: mlops-project-v1
        * Format: Docker
        * Region: e.g. `southasia delhi
      3. Click Create.
  
  - Step 3: Create Service Account for GitHub Actions
      1. Go to IAM & Admin → Service Accounts → Create Service Account.
        * Name: github-actions-deployer
      2. Click Create and Continue.
      3. Assign roles:
        * Cloud Run Admin
        * Artifact Registry Writer
        * Service Account User
        * Artifact Registry Admin
        * Viewer
      4. Click **Done**.

  - Step 4: Generate and download the service account key
      1. Click the service account you just created.
      2. Go to **Keys → Add key → Create new key**.
      3. Choose **JSON**.
      4. Download the key file — this contains your credentials.
  
  - Step 5: Add GitHub Secrets
    Go to your **GitHub repo → Settings → Secrets and variables → Actions**.
    Create these secrets:

    | Secret Name      | Description                                              |
    | ---------------- | -------------------------------------------------------- |
    | `GCP_PROJECT_ID` | Your Google Cloud Project ID                             |
    | `GCP_REGION`     | e.g. `asiasouth2 delhi`                                       |
    | `GCP_SA_KEY`     | The entire JSON key content from the service account     |
    | `AR_REPO`        | Artifact Registry name (e.g., mlops-project-v1)              |
    | `SERVICE_NAME`   | Cloud Run service name (e.g., mlops-app) |

  - Step 6: Add the deployment workflow
    Create this file: .github/workflows/deploy.yml
  
  - Step 7: Create docker file with proper port handling
  - Step 8: Trigger Deployment
    - Now just push to main
    - GitHub Actions will:
        1. Build your Docker image.
        2. Push it to Artifact Registry.
        3. Deploy it to Cloud Run.
        4. Output your service URL at the end.
  - Step 9: Check your deployed app
      - In **Google Cloud Console → Cloud Run → Services**,
      - find your service name (e.g., `mlops-app`) → click → copy the URL.
      - Your deployed Streamlit app will be live there.

✨✨✨✨

