.PHONY: flagship-b flagship-b-ci

flagship-b:
	python3 scripts/flagship_flow_litho.py --seed-sweep 10

flagship-b-ci:
	python3 scripts/flagship_flow_litho.py --seed-sweep 3
