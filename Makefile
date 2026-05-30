.PHONY: flagship-b flagship-b-ci flagship docker-build docker-flagship docker-cross-validate reproduce cross-validate

flagship-b:
	python3 scripts/flagship_flow_litho.py --seed-sweep

flagship-b-ci:
	python3 scripts/flagship_flow_litho.py --seed-sweep --seed-end 44

flagship: flagship-b ## Alias for flagship-b

docker-build:
	docker build -t diffcfd-flagship .

docker-flagship: docker-build
	docker run --rm diffcfd-flagship

docker-cross-validate: docker-build
	docker run --rm diffcfd-flagship python scripts/cross_validate.py

reproduce: ## One-key reproducibility (local, no docker)
	python3 scripts/flagship_flow_litho.py --seed-sweep --seed-start 42 --seed-end 44

cross-validate: ## Run cross-validation against analytical solutions
	python3 scripts/cross_validate.py
