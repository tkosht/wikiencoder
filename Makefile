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
export PROJECT_NAME=wikiencoder
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
all: preprocess run

run: init-log vectorizer run-visdom-server seq2vec

vectorizer seq2vec encoder_toy :
	$(PYTHON) $(PYTHON_MODULE)/$@.py

run-visdom-server:
	$(eval server := $(shell egrep -A 2 '^\[visdom\]' config/project.cfg | egrep 'server\s*=' | awk '{print $$NF}'))
	$(eval port := $(shell egrep -A 2 '^\[visdom\]' config/project.cfg | egrep 'port\s*=' | awk '{print $$NF}'))
	HOSTNAME=$(server) $(PYTHON) -m visdom.server -port $(port) > $(LOG_DIR)/visdom.log 2>&1 &

kill-visdom-server:
	$(eval visdom := $(shell ps -ef | egrep 'visdom\.server'| awk '{print $$2}'))
	$(eval chk := $(shell [ "$(visdom)" ] && kill $(visdom)))
	@sleep 1

init-log:
	$(eval logfile := $(LOG_DIR)/project.log)
	: > $(logfile)

tags:
	ctags -R $(PYTHON_MODULE) tests

###########################################################################################################
## GENERAL TARGETS
###########################################################################################################

help:
	@$(PYTHON) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

init: init-docker init-data ## initialize repository for traning

init-test-log:
	$(eval logfile := $(LOG_DIR)/test_project.log)
	mkdir -p $(LOG_DIR)
	: > $(logfile)

init-data: prep prepare-testdata
#	-aws s3 sync $(DATA) ./data/

init-docker: docker-build	## initialize docker image

docker-build:
	sh scripts/docker/build.sh

docker-clean:
	sh scripts/docker/clean.sh

docker-run:
	sh scripts/docker/run.sh

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

test: init-test-log init-testdata
	$(eval opts := -c config/setup.cfg)
	pytest $(opts)

init-testdata: data/test/

data/test/:
	sh scripts/pickup.sh	# to sample wiki text files(docs and titles)

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
	rm -rf model/*

clean-pyc: ## remove Python file artifacts
	find $(PYTHON_MODULE) -name '*.pyc' -exec rm -f {} +
	find $(PYTHON_MODULE) -name '*.pyo' -exec rm -f {} +
	find $(PYTHON_MODULE) -name '*~' -exec rm -f {} +
	find $(PYTHON_MODULE) -name '__pycache__' -exec rm -fr {} +

distclean: clean clean-data ## remove all the reproducible resources including Docker images

clean-data: ## remove files under data
	rm -rf data/*

clean-docker: clean-container clean-image ## remove Docker image and container

clean-container: ## remove Docker container
	-$(DOCKER) rm $(CONTAINER_NAME)

clean-image: ## remove Docker image
	-$(DOCKER) image rm $(IMAGE_NAME)
