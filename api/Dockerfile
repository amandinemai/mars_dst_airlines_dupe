# Utilisation de l'image de base python
FROM python:3.9

# les variables d'environnement pour éviter les messages d'erreur de debconf
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1


# Exposer le port sur lequel l'API FastAPI s'exécute
EXPOSE 8000
EXPOSE 5000

# Le répertoire de travail dans le conteneur
WORKDIR /app

COPY ./api/requirements.txt /app/requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

COPY ./api /app

# Commande pour démarrer l'API FastAPI
# CMD ["uvicorn", "fastapi_main:app", "--host", "0.0.0.0", "--port", "8000"]

# Commande pour démarrer mlflow 
# CMD ["mlflow", "run", "-d", "mlruns"]