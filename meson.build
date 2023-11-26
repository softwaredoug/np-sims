project('np_sims', 'c', version : '0.1')

py = import('python').find_installation()

# Check if VIRTUAL_ENV is set
message('VIRTUAL_ENV:')

message('sys.path')
op = run_command(py, ['-c', 'import sys; print(sys.path)'], capture: true, check: true)
message(op.stdout().strip())

# Check which python
message('PRE INSTALL PYTHON HAS PACKAGES:')
op = run_command(py, ['-m', 'pip', 'list'], capture: true, check: false)
message(op.stderr().strip())

message('INSTALLING NUMPY')
op = run_command(py, ['-m', 'pip', 'install', 'numpy'], capture: true,  check: true)
message(op.stdout().strip())

# Check which python
message('POST INSTALL PYTHON HAS PACKAGES:')
op = run_command(py, ['-m', 'pip', 'list'], capture: true, check: true)
message(op.stdout().strip())

err_msg = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  capture: true,
  check: false,
).stderr().strip()

message('USING NUMPY INCLUDE DIR, STDERR:')
message(err_msg)

incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  capture: true,
  check: false,
).stdout().strip()

# incdir_numpy = 'venv/lib/python3.11/site-packages/numpy/core/include'

includes = include_directories('np_sims', incdir_numpy)

srcs = [
  'np_sims/ufunc.c',
]

py.install_sources(
    'np_sims/__init__.py',
    subdir: 'np_sims',
    pure: false,
)

py.extension_module(
    'ufunc',
    srcs,
    install: true,
    subdir: 'npsims',
    include_directories: includes,
)
