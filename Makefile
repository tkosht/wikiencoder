.PHONY: clean clean-model clean-pyc docs help init init-data init-docker create-container start-container test lint profile clean clean-data clean-docker clean-container clean-image
.DEFAULT_GOAL := help

###########################################################################################################
## SCRIPTS
###########################################################################################################

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
        match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
        if match:
                target, help = match.groups()
                print("%-20s %s" % (target, help))
endef

define START_DOCKER_CONTAINER
if [ `$(DOCKER) inspect -f {{.State.Running}} $(CONTAINER_NAME)` = "false" ] ; then
        $(DOCKER) start $(CONTAINER_NAME)
fi
endef

###########################################################################################################
## VARIABLES
###########################################################################################################


export DOCKER=docker
export PRINT_HELP_PYSCRIPT
export START_DOCKER_CONTAINER
export PYTHONPATH=$(printenv PYTHONPATH):$(PWD)
export PROJECT_NAME=pypj
export PYTHON_MODULE=project
export TEST_MODULE=tests
export IMAGE_NAME=$(PROJECT_NAME)-image
export CONTAINER_NAME=$(PROJECT_NAME)-container
export DATA=s3://aaa
export PYTHON=python3
export DOCKERFILE=docker/Dockerfile
export LOG_DIR=log
export WIKIXMLBZ2=$(PWD)/data/enwiki-latest-pages-articles.xml.bz2

###########################################################################################################
## ADD TARGETS FOR YOUR TASK
###########################################################################################################


###########################################################################################################
## GENERAL TARGETS
###########################################################################################################

help:
	@$(PYTHON) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

init: init-docker init-data ## initialize repository for traning

init-log:
	mkdir -p $(LOG_DIR)

init-data: prep
#	-aws s3 sync $(DATA) ./data/

init-docker: ## initialize docker image
	$(DOCKER) build -t $(IMAGE_NAME) -f $(DOCKERFILE) .

# for module test
logger decorator:
	$(PYTHON) $(PYTHON_MODULE)/$@.py

prep: preprocess

preprocess: data/wikitxt/ 	# after extracted
	sh scripts/sep2doc.sh

data/wikitxt/: $(WIKIXMLBZ2)
	$(eval outdir := $(PWD)/$@)
	sh scripts/wikiextractor.sh $(WIKIXMLBZ2) $(outdir)

$(WIKIXMLBZ2):
	sh scripts/download.sh

create-container: ## create docker container
	$(DOCKER) run -it -v $(PWD):/work --name $(CONTAINER_NAME) $(IMAGE_NAME)

test: init-log
	$(eval logfile := -c log/test_project.log)
	$(eval config := -c config/setup.cfg)
	touch $(logfile)
	truncate --size=0 $(logfile)
	pytest $(config)

run-cov-server:
	cd tests/report && $(PYTHON) -m http.server 8001

lint: ## check style with flake8
	$(eval config := --config=config/setup.cfg)
	flake8 $(PYTHON_MODULE) $(TEST_MODULE) $(config)

doc:
	$(eval config := --config=config/setup.cfg)

profile: ## show profile of the project
	@echo "CONTAINER_NAME: $(CONTAINER_NAME)"
	@echo "IMAGE_NAME: $(IMAGE_NAME)"
#	@echo "DATA: $(DATA)"

clean: clean-model clean-pyc clean-docker ## remove all artifacts

clean-model: ## remove model artifacts
	rm -fr model/*

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

distclean: clean clean-data ## remove all the reproducible resources including Docker images

clean-data: ## remove files under data
	rm -fr data/*

clean-docker: clean-container clean-image ## remove Docker image and container

clean-container: ## remove Docker container
	-$(DOCKER) rm $(CONTAINER_NAME)

clean-image: ## remove Docker image
	-$(DOCKER) image rm $(IMAGE_NAME)
