project('np_sims', 'c', version : '0.1')

py = import('python').find_installation(pure: true)

incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  capture: true,
  check: false,
).stderr().strip()

message('USING NUMPY INCLUDE DIR:')
message(incdir_numpy)

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
