# Agent implementation for Q-TextWorld

## Prerequisite

    pip install textworld==1.5.1
    pip install "spacy>=3"
    python -m spacy download en_core_web_sm

## Switch among No Query / Query Baseline or AFK:
    change general.settings in config.yaml

## Switch among settings:
### Take 1:
    In config.yaml:
    tw.recipe = 1
    tw.take = 1
    tw.cut = False
### Take 2:
    In config.yaml:
    tw.recipe = 2
    tw.take = 2
    tw.cut = False
### Take 1 Cut:
    In config.yaml:
    tw.recipe = 1
    tw.take = 1
    tw.cut = True
### Take 2 Cut:
    In config.yaml:
    tw.recipe = 2
    tw.take = 2
    tw.cut = True

## Train agent
    python main.py config.yaml
