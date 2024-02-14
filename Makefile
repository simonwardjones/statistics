
help: # run the help (exit with q)
	poetry run python -m makefile_help

jn: # run jupyuter in pipenv
	poetry run jupyter notebook
