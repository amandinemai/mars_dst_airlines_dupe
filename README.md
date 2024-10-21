Pour demarrer le container de notre API :
docker run -d --name my-container -p 8000:8000 my-fastapi-app

#Pour constuire et demarrer les container
docker-compose up --build
