build_local: np_sims/ufunc.c np_sims/heap.h
	cd np_sims && python3 setup.py build_ext --inplace

install:
	cd np_sims && python3 setup.py build && python4 setup.py install

test: rebuild
	python3 -m pytest -v test

clean:
	rm -rf np_sims/*.so

rebuild: clean build_local

clean_force:
	git clean -fd -e venv -e data -e *.py
