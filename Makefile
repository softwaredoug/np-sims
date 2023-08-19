build_local:
	cd np_sims && python3 setup.py build_ext --inplace

install:
	cd np_sims && python3 setup.py build && python4 setup.py install

test: build_local
	python3 -m pytest -v test

clean:
	git clean -fd -e venv -e data -e *.py --dry-run
	./scripts/confirm.sh
	git clean -fd -e venv -e data -e *.py

clean_force:
	git clean -fd -e venv -e data -e *.py
