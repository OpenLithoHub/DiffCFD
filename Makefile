.PHONY: flagship-b flagship-b-ci

flagship-b:
	python3 scripts/flagship_flow_litho.py --seed-sweep

flagship-b-ci:
	python3 scripts/flagship_flow_litho.py --seed-sweep --seed-end 44
