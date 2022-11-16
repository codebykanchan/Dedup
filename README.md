Project folder contains the code for pipeline of the prediction model.

Dockerfile contains the instructions to build docker image

requirements.txt contains the required python packages to replicate the working environment for python.
Usage: (linux)

to generate the requirements.txt (this will contain the snapshot of current python environment)
Please generate a requirements.txt every time you intsall a new python package.
This will help us later to replicate the working environment on production/test servers.

pip freeze > requirements.txt

to replicate the environment from given requirements.txt

pip install -r requirements.txt


# How to build Docker image:
docker build -t \<Name of the image\> .

# How to run the Docker image:
docker run -p 8000:8000  \<Name of the image\>

# How to test the JSON api:
* access localhost:8000/docs using web browser
* Paste the Json of the git ticket which we need to find the similar issues for.
* execute
* the closest five matches will be returend in the response JSON.
