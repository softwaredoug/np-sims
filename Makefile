build_local:
	cd np_sims && python3 setup.py build_ext --inplace

install:
	cd np_sims && python3 setup.py build && python3 setup.py install

clean:
	git clean -fd -e venv -e data -e *.py --dryrun
	# confirm before proceeding
	input "Are you sure you want to delete all untracked files? [y/N] " answer
	ifeq ($(answer), y)
		git clean -fd -e venv -e data -e *.py
