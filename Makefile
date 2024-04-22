# Makefile for Image Classification Project

# Targets and Rules
.PHONY: all
all: help

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  train          Train the neural network model"
	@echo "  classify       Run the user interaction script"
	@echo "  clean          Remove temporary files and logs"
	@echo "  help           Show this help message"

.PHONY: train
train:
	$python3 classifier.py
	@echo "Pytorch Output..."

.PHONY: classify
classify:
	$python3 userInteraction.py

.PHONY: clean
clean:
	rm -rf __pycache__/
